"""
stage2_pose/tracker.py
======================
Mục đích : Giữ đúng 1 người (học viên) xuyên suốt video khi pose_extractor.py
           phát hiện nhiều người trong khung.

V2.2 - Hỗ trợ ByteTrack:
    - Nếu backend (RTMPoseModel) đã trả track_id (ByteTrack), tracker.py CHỈ
      cần chọn người chính theo track_id ổn định — không cần centroid tracker.
    - Nếu backend không trả track_id (MediaPipe / YoloPoseModel cũ), fallback
      về CentroidTracker như trước để đảm bảo tương thích ngược.

V2.3 - Sửa bug tracker (TR1–TR3):
    - TR1: KHÔNG skip frame không detect được người. Append PoseFrame placeholder
      (xy=NaN, conf=0) với frame_idx gốc → frame_idx liên tục → hết crash
      HLV08 khi manual phases trỏ vào frame bị drop.
    - TR2: ByteTracker giữ _main_track_id ổn định; đếm miss liên tiếp;
      log warning khi mất ID > 5 frame.
    - TR3: Fallback CentroidTracker track TẤT CẢ detections/frame trước,
      rồi mới chọn người chính theo track_id ổn định (không còn update
      1-element list → track_id luôn 0).

Input    : List[List[PoseFrame]] thô của MỖI frame (0..N người).
Output   : List[PoseFrame] 1 người/frame (hoặc placeholder NaN), gắn track_id.
           Độ dài output == độ dài input (mọi frame được giữ).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.pose_schema import COCO17_NAMES, PoseFrame

logger = logging.getLogger(__name__)

# Số frame mất track_id liên tiếp trước khi log warning (TR2)
_BYTE_MISS_WARN_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _make_placeholder(
    frame_idx: int,
    timestamp_s: float = 0.0,
    joint_names: Sequence[str] = COCO17_NAMES,
    track_id: Optional[int] = None,
) -> PoseFrame:
    """
    PoseFrame rỗng (TR1): xy = NaN, conf = 0.
    Giữ frame_idx để chuỗi không bị lỗ hổng — anomaly_detector / quality
    gate sẽ loại frame này ở bước sau.
    """
    n = len(joint_names)
    return PoseFrame(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        xy=np.full((n, 2), np.nan, dtype=np.float32),
        conf=np.zeros(n, dtype=np.float32),
        joint_names=list(joint_names),
        bbox=None,
        track_id=track_id,
    )


def _infer_frame_idx(
    people: List[PoseFrame], list_index: int
) -> int:
    """Lấy frame_idx từ detection có sẵn, fallback về chỉ số list."""
    for p in people:
        if p is not None and hasattr(p, "frame_idx"):
            return int(p.frame_idx)
    return list_index


def _infer_timestamp(people: List[PoseFrame]) -> float:
    for p in people:
        if p is not None and hasattr(p, "timestamp_s"):
            return float(p.timestamp_s)
    return 0.0


def _infer_joint_names(people: List[PoseFrame]) -> Sequence[str]:
    for p in people:
        if p is not None and p.joint_names:
            return p.joint_names
    return COCO17_NAMES


# ---------------------------------------------------------------------------
# Centroid tracker (fallback khi backend KHÔNG trả track_id)
# ---------------------------------------------------------------------------
@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    centroid: np.ndarray
    disappeared: int = 0
    history_len: int = field(default=0)


class CentroidTracker:
    """Gán/duy trì track_id dựa trên khoảng cách centroid + IoU.

    Chỉ dùng khi backend KHÔNG cung cấp track_id (MediaPipe / YOLO cũ).
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        max_centroid_dist: float = 120.0,
        min_iou: float = 0.1,
    ):
        self._next_id = 0
        self._tracks: Dict[int, _Track] = {}
        self.max_disappeared = max_disappeared
        self.max_centroid_dist = max_centroid_dist
        self.min_iou = min_iou

    def _register(self, bbox: np.ndarray) -> int:
        tid = self._next_id
        self._tracks[tid] = _Track(
            track_id=tid, bbox=bbox, centroid=_bbox_center(bbox)
        )
        self._next_id += 1
        return tid

    def _deregister(self, track_id: int) -> None:
        del self._tracks[track_id]

    def update(
        self, detections_bbox: List[Optional[np.ndarray]]
    ) -> List[Optional[int]]:
        valid_idx = [i for i, b in enumerate(detections_bbox) if b is not None]
        assigned: List[Optional[int]] = [None] * len(detections_bbox)
        if not self._tracks:
            for i in valid_idx:
                assigned[i] = self._register(detections_bbox[i])
            return assigned
        if not valid_idx:
            for t in list(self._tracks.values()):
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    self._deregister(t.track_id)
            return assigned

        track_ids = list(self._tracks.keys())
        track_centroids = np.array(
            [self._tracks[t].centroid for t in track_ids]
        )
        det_centroids = np.array(
            [_bbox_center(detections_bbox[i]) for i in valid_idx]
        )
        dist_matrix = np.linalg.norm(
            track_centroids[:, None, :] - det_centroids[None, :, :], axis=2
        )
        used_tracks, used_dets = set(), set()
        pairs = sorted(
            (
                (dist_matrix[r, c], r, c)
                for r in range(dist_matrix.shape[0])
                for c in range(dist_matrix.shape[1])
            ),
            key=lambda x: x[0],
        )
        for dist, r, c in pairs:
            if r in used_tracks or c in used_dets:
                continue
            tid = track_ids[r]
            det_i = valid_idx[c]
            iou = _bbox_iou(self._tracks[tid].bbox, detections_bbox[det_i])
            if dist <= self.max_centroid_dist or iou >= self.min_iou:
                self._tracks[tid].bbox = detections_bbox[det_i]
                self._tracks[tid].centroid = det_centroids[c]
                self._tracks[tid].disappeared = 0
                self._tracks[tid].history_len += 1
                assigned[det_i] = tid
                used_tracks.add(r)
                used_dets.add(c)
        for r, tid in enumerate(track_ids):
            if r not in used_tracks:
                self._tracks[tid].disappeared += 1
                if self._tracks[tid].disappeared > self.max_disappeared:
                    self._deregister(tid)
        for c, det_i in enumerate(valid_idx):
            if c not in used_dets:
                assigned[det_i] = self._register(detections_bbox[det_i])
        return assigned

    def most_stable_track_id(self) -> Optional[int]:
        """Track_id có history_len dài nhất (ổn định nhất)."""
        if not self._tracks:
            return None
        best = max(self._tracks.values(), key=lambda t: t.history_len)
        return best.track_id


# ---------------------------------------------------------------------------
# ByteTrack-aware tracker (dùng khi backend đã gán track_id)
# ---------------------------------------------------------------------------
class ByteTracker:
    """Wrapper mỏng: giữ track_id ổn định do ByteTrack (YOLO) cấp.

    Không tự gán ID mới. Chỉ:
        1. Ưu tiên người có track_id == main_track_id.
        2. Nếu mất → chọn lại theo heuristic, cập nhật main nếu có ID.
        3. (TR2) Đếm miss liên tiếp, log warning khi > ngưỡng.
    """

    def __init__(self, strategy: str = "largest_and_central"):
        self._main_track_id: Optional[int] = None
        self._strategy = strategy
        self._consecutive_misses: int = 0

    def select(
        self, people: List[PoseFrame], frame_shape: Sequence[int]
    ) -> Optional[PoseFrame]:
        if not people:
            self._consecutive_misses += 1
            self._maybe_warn_miss()
            return None

        # TR2: Ưu tiên người có track_id == main (kể cả khi một số
        # detection khác có track_id = None)
        if self._main_track_id is not None:
            for p in people:
                if p.track_id is not None and p.track_id == self._main_track_id:
                    self._consecutive_misses = 0
                    return p
            # Main ID không có trong frame này
            self._consecutive_misses += 1
            self._maybe_warn_miss()

        # Chưa có / mất: chọn theo heuristic
        chosen = select_main_subject(
            people, frame_shape, strategy=self._strategy
        )
        if chosen is None:
            self._consecutive_misses += 1
            self._maybe_warn_miss()
            return None

        if chosen.track_id is not None:
            # Cập nhật / khóa main track_id
            if self._main_track_id is None:
                self._main_track_id = chosen.track_id
            elif chosen.track_id != self._main_track_id:
                # Heuristic chọn người khác có ID → chấp nhận chuyển
                # (main đã mất quá lâu)
                if self._consecutive_misses >= _BYTE_MISS_WARN_THRESHOLD:
                    logger.info(
                        "[tracker] ByteTracker chuyển main_track_id "
                        "%s → %s sau %d frame miss.",
                        self._main_track_id,
                        chosen.track_id,
                        self._consecutive_misses,
                    )
                    self._main_track_id = chosen.track_id
            self._consecutive_misses = 0
        # chosen.track_id is None: giữ _main_track_id cũ, không reset
        return chosen

    def _maybe_warn_miss(self) -> None:
        if (
            self._consecutive_misses == _BYTE_MISS_WARN_THRESHOLD
            and self._main_track_id is not None
        ):
            logger.warning(
                "[tracker] ByteTracker mất track_id=%s liên tiếp %d frame. "
                "Pose có thể nhảy sang người khác nếu heuristic chọn sai.",
                self._main_track_id,
                self._consecutive_misses,
            )


# ---------------------------------------------------------------------------
# Chọn người chính
# ---------------------------------------------------------------------------
def select_main_subject(
    frame_people: List[PoseFrame],
    frame_shape: Sequence[int],
    strategy: str = "largest_and_central",
) -> Optional[PoseFrame]:
    """Chọn 1 người "chính" trong số nhiều người ở CÙNG 1 frame.

    strategy:
        - "largest"             : bbox lớn nhất.
        - "central"             : centroid gần tâm khung nhất.
        - "largest_and_central" : kết hợp 60% area + 40% centrality (mặc định).
    """
    candidates = [p for p in frame_people if p.bbox is not None]
    if not candidates:
        return frame_people[0] if frame_people else None
    if len(candidates) == 1:
        return candidates[0]

    h, w = frame_shape[0], frame_shape[1]
    frame_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    areas = np.array(
        [_bbox_area(p.bbox) for p in candidates], dtype=np.float32
    )
    dists = np.array(
        [
            np.linalg.norm(_bbox_center(p.bbox) - frame_center)
            for p in candidates
        ],
        dtype=np.float32,
    )
    if strategy == "largest":
        best = int(np.argmax(areas))
    elif strategy == "central":
        best = int(np.argmin(dists))
    else:
        area_score = areas / (areas.max() + 1e-6)
        dist_score = 1.0 - dists / (dists.max() + 1e-6)
        combined = 0.6 * area_score + 0.4 * dist_score
        best = int(np.argmax(combined))
    return candidates[best]


# ---------------------------------------------------------------------------
# API chính (giữ nguyên chữ ký cũ để pipeline không phải sửa)
# ---------------------------------------------------------------------------
def track_and_filter(
    per_frame_detections: List[List[PoseFrame]],
    frame_shape: Sequence[int],
) -> List[PoseFrame]:
    """
    Chọn người chính + giữ track_id ổn định.

    V2.3 (TR1): Output có CÙNG số phần tử với input. Frame không detect
    được người → placeholder (xy=NaN, conf=0) với frame_idx gốc.
    → frame_idx liên tục → manual phases không còn KeyError.

    Tự động chọn cơ chế:
        - Nếu backend đã gán track_id (ByteTrack) → ByteTracker.
        - Nếu chưa → CentroidTracker (fallback, TR3 đã sửa).
    """
    if not per_frame_detections:
        return []

    has_bytetrack_ids = any(
        p.track_id is not None
        for people in per_frame_detections[
            : min(10, len(per_frame_detections))
        ]
        for p in people
    )

    n_placeholders = 0

    if has_bytetrack_ids:
        byte_tracker = ByteTracker()
        result: List[PoseFrame] = []
        for i, people in enumerate(per_frame_detections):
            chosen = byte_tracker.select(people, frame_shape)
            if chosen is not None:
                result.append(chosen)
            else:
                # TR1: placeholder thay vì skip
                result.append(
                    _make_placeholder(
                        frame_idx=_infer_frame_idx(people, i),
                        timestamp_s=_infer_timestamp(people),
                        joint_names=_infer_joint_names(people),
                        track_id=byte_tracker._main_track_id,
                    )
                )
                n_placeholders += 1
        if n_placeholders:
            logger.info(
                "[tracker] ByteTrack: %d/%d frame không detect → placeholder NaN.",
                n_placeholders,
                len(per_frame_detections),
            )
        return result

    # ----- Fallback CentroidTracker (TR3: track ALL detections trước) -----
    tracker = CentroidTracker()
    result = []
    main_track_id: Optional[int] = None

    for i, people in enumerate(per_frame_detections):
        if not people:
            result.append(
                _make_placeholder(
                    frame_idx=_infer_frame_idx(people, i),
                    timestamp_s=0.0,
                    joint_names=COCO17_NAMES,
                    track_id=main_track_id,
                )
            )
            n_placeholders += 1
            continue

        # TR3: track TẤT CẢ detections trong frame
        bboxes: List[Optional[np.ndarray]] = [
            p.bbox if p.bbox is not None else None for p in people
        ]
        track_ids = tracker.update(bboxes)

        # Gán track_id vào từng detection
        for p, tid in zip(people, track_ids):
            if tid is not None:
                p.track_id = tid

        # Ưu tiên người có track_id == main_track_id (ổn định)
        chosen: Optional[PoseFrame] = None
        if main_track_id is not None:
            for p in people:
                if p.track_id == main_track_id:
                    chosen = p
                    break

        if chosen is None:
            chosen = select_main_subject(people, frame_shape)
            if chosen is not None and chosen.track_id is not None:
                main_track_id = chosen.track_id

        if chosen is None:
            result.append(
                _make_placeholder(
                    frame_idx=_infer_frame_idx(people, i),
                    timestamp_s=_infer_timestamp(people),
                    joint_names=_infer_joint_names(people),
                    track_id=main_track_id,
                )
            )
            n_placeholders += 1
        else:
            result.append(chosen)

    if n_placeholders:
        logger.info(
            "[tracker] Centroid: %d/%d frame không detect → placeholder NaN.",
            n_placeholders,
            len(per_frame_detections),
        )
    return result