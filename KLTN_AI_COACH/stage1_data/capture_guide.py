"""
stage1_data/capture_guide.py
=============================
Mục đích : Hướng dẫn người dùng quay video đúng chuẩn (kiểm tra khung hình,
           góc quay, khoảng cách) và báo hiệu "sẵn sàng" (khung xanh) khi
           đạt tiêu chí, để tránh ảnh hưởng độ chính xác ở các giai đoạn sau
           (đã bàn ở "Vấn đề 3 - góc quay khác nhau").
Input    : Video stream / webcam (OpenCV VideoCapture).
Output   : Video đã được xác nhận hợp lệ -> đưa vào pose_extractor.py
           (Giai đoạn 2) để trích xuất keypoint chính thức, hoặc trực tiếp
           bắt đầu ghi hình khi đủ điều kiện.

Toàn bộ tiêu chí đều TÍNH ĐƯỢC (không dựa cảm tính "ước lượng 4-5m"):
    - full_body_in_frame : mọi keypoint chính có confidence đủ cao và nằm
      trong khung, có chừa lề.
    - subject_scale_ok   : người chiếm 60-80% chiều cao khung (đại diện cho
      khoảng cách phù hợp, thay cho việc yêu cầu đo mét bằng mắt).
    - camera_level_ok    : thân người không nghiêng bất thường so với
      phương thẳng đứng (ước lượng độ nghiêng camera/tư thế).
    - perpendicular_ok   : tỉ lệ bề rộng vai / chiều dài thân dùng để ước
      lượng góc quay có gần vuông góc với hướng chuyển động hay không.

Khi CẢ 4 tiêu chí đạt trong N frame liên tiếp (ổn định) -> "khung xanh".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from core.pose_schema import PoseFrame

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# 1. Cấu hình ngưỡng (có thể override qua system_config.yaml)
# ---------------------------------------------------------------------------
@dataclass
class CaptureThresholds:
    min_visible_ratio: float = 0.9          # >=90% keypoint chính phải "thấy"
    frame_margin_ratio: float = 0.03        # chừa lề 3% mỗi cạnh khung
    subject_height_ratio_min: float = 0.55  # người chiếm tối thiểu 55% chiều cao khung
    subject_height_ratio_max: float = 0.85  # và tối đa 85% (quá gần cũng không tốt)
    max_trunk_tilt_deg: float = 12.0        # độ nghiêng thân tối đa so với phương thẳng đứng
    shoulder_hip_ratio_min: float = 0.55    # tỉ lệ bề rộng vai/hông dùng ước lượng góc quay
    shoulder_hip_ratio_max: float = 1.6
    stable_frames_required: int = 15        # số frame liên tiếp đạt chuẩn -> "khung xanh"


# Các keypoint bắt buộc phải nhìn thấy để coi là "toàn thân trong khung"
_CORE_JOINTS = [
    "nose", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


@dataclass
class CaptureStatus:
    full_body_in_frame: bool
    subject_scale_ok: bool
    camera_level_ok: bool
    perpendicular_ok: bool
    messages: List[str]

    @property
    def is_ready(self) -> bool:
        return (
            self.full_body_in_frame
            and self.subject_scale_ok
            and self.camera_level_ok
            and self.perpendicular_ok
        )

    def to_dict(self) -> Dict:
        return {
            "full_body_in_frame": self.full_body_in_frame,
            "subject_scale_ok": self.subject_scale_ok,
            "camera_level_ok": self.camera_level_ok,
            "perpendicular_ok": self.perpendicular_ok,
            "is_ready": self.is_ready,
            "messages": self.messages,
        }


class CaptureGuide:
    """Đánh giá 1 PoseFrame theo 4 tiêu chí đặt khung, và theo dõi độ ổn
    định qua nhiều frame liên tiếp để quyết định thời điểm "khung xanh".
    """

    def __init__(self, thresholds: Optional[CaptureThresholds] = None):
        self.th = thresholds or CaptureThresholds()
        self._stable_count = 0

    def reset(self) -> None:
        self._stable_count = 0

    def evaluate_frame(self, pose: PoseFrame, frame_shape) -> CaptureStatus:
        h, w = frame_shape[0], frame_shape[1]
        messages: List[str] = []

        # --- 1. Toàn thân trong khung + đủ lề ----------------------------------
        core_conf = np.array([pose.get_conf(j) for j in _CORE_JOINTS])
        visible_ratio = float(np.mean(core_conf >= 0.35))
        core_xy = np.array([pose.get(j) for j in _CORE_JOINTS])
        margin_x, margin_y = w * self.th.frame_margin_ratio, h * self.th.frame_margin_ratio
        within_margin = bool(
            np.all(core_xy[:, 0] >= margin_x)
            and np.all(core_xy[:, 0] <= w - margin_x)
            and np.all(core_xy[:, 1] >= margin_y)
            and np.all(core_xy[:, 1] <= h - margin_y)
        )
        full_body_in_frame = visible_ratio >= self.th.min_visible_ratio and within_margin
        if not full_body_in_frame:
            messages.append("Chưa thấy đủ toàn thân trong khung — lùi camera ra hoặc chỉnh khung hình.")

        # --- 2. Khoảng cách (qua tỉ lệ chiều cao người / chiều cao khung) ------
        top_y = min(pose.get("nose")[1], core_xy[:, 1].min())
        bottom_y = max(pose.get("left_ankle")[1], pose.get("right_ankle")[1])
        subject_height_ratio = float((bottom_y - top_y) / h) if h > 0 else 0.0
        subject_scale_ok = (
            self.th.subject_height_ratio_min <= subject_height_ratio <= self.th.subject_height_ratio_max
        )
        if subject_height_ratio < self.th.subject_height_ratio_min:
            messages.append("Bạn đang ở quá xa camera — tiến lại gần hơn.")
        elif subject_height_ratio > self.th.subject_height_ratio_max:
            messages.append("Bạn đang ở quá gần camera — lùi ra xa hơn.")

        # --- 3. Camera có bị nghiêng không (độ nghiêng thân so với phương đứng) -
        shoulder_mid = (pose.get("left_shoulder") + pose.get("right_shoulder")) / 2.0
        hip_mid = (pose.get("left_hip") + pose.get("right_hip")) / 2.0
        trunk_vec = shoulder_mid - hip_mid
        tilt_deg = float(np.degrees(np.arctan2(abs(trunk_vec[0]), abs(trunk_vec[1]) + 1e-6)))
        camera_level_ok = tilt_deg <= self.th.max_trunk_tilt_deg
        if not camera_level_ok:
            messages.append(f"Camera hoặc tư thế đang nghiêng ~{tilt_deg:.0f}° — chỉnh lại cho thẳng.")

        # --- 4. Góc quay có gần vuông góc với hướng chuyển động không ----------
        shoulder_w = float(np.linalg.norm(pose.get("left_shoulder") - pose.get("right_shoulder")))
        hip_w = float(np.linalg.norm(pose.get("left_hip") - pose.get("right_hip")))
        ratio = shoulder_w / hip_w if hip_w > 1e-6 else 0.0
        perpendicular_ok = self.th.shoulder_hip_ratio_min <= ratio <= self.th.shoulder_hip_ratio_max
        if not perpendicular_ok:
            messages.append("Góc camera có vẻ chưa vuông góc với hướng chạy — xoay camera lại.")

        status = CaptureStatus(
            full_body_in_frame=full_body_in_frame,
            subject_scale_ok=subject_scale_ok,
            camera_level_ok=camera_level_ok,
            perpendicular_ok=perpendicular_ok,
            messages=messages or ["Đạt chuẩn — giữ nguyên vị trí."],
        )

        self._stable_count = self._stable_count + 1 if status.is_ready else 0
        return status

    @property
    def is_stable_ready(self) -> bool:
        """True khi đã đạt chuẩn liên tục đủ số frame yêu cầu (khung xanh)."""
        return self._stable_count >= self.th.stable_frames_required


# ---------------------------------------------------------------------------
# 2. Vòng lặp trực tiếp (live) dùng webcam/điện thoại qua OpenCV
# ---------------------------------------------------------------------------
def run_live_check(
    video_source,
    pose_extractor,  # instance của stage2_pose.pose_extractor.PoseExtractor
    on_ready: Optional[Callable[[np.ndarray], None]] = None,
    thresholds: Optional[CaptureThresholds] = None,
    show_window: bool = True,
) -> None:
    """Vòng lặp thời gian thực: đọc frame -> pose -> đánh giá -> hiển thị
    phản hồi -> khi ổn định đủ lâu thì gọi on_ready(frame) (ví dụ để bắt đầu
    ghi hình chính thức hoặc phát tín hiệu cho người dùng).

    Đây là bước "guided capture" giải quyết trực tiếp Vấn đề 1 (nghỉ/ngã
    giữa động tác) ở lớp thu thập dữ liệu: chỉ khi khung đạt chuẩn mới bắt
    đầu tính là "bắt đầu quay hợp lệ".
    """
    if cv2 is None:
        raise RuntimeError("Cần cài opencv-python: pip install opencv-python-headless")

    guide = CaptureGuide(thresholds)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {video_source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = pose_extractor._model.infer(frame)  # dùng model thô, 1 frame
            if not detections:
                guide.reset()
                text = "Không phát hiện người trong khung."
                color = (0, 0, 255)
            else:
                det = detections[0]
                pose = PoseFrame(
                    frame_idx=0,
                    timestamp_s=time.time(),
                    xy=det.xy,
                    conf=det.conf,
                    joint_names=pose_extractor._model.joint_names,
                )
                status = guide.evaluate_frame(pose, frame.shape)
                text = " | ".join(status.messages)
                color = (0, 200, 0) if status.is_ready else (0, 165, 255)

                if guide.is_stable_ready:
                    color = (0, 255, 0)
                    text = "SẴN SÀNG — bắt đầu quay!"
                    if on_ready is not None:
                        on_ready(frame)

            if show_window:
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Capture Guide", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
