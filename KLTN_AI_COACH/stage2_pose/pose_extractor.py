"""
stage2_pose/pose_extractor.py
=============================
Mục đích : Trích keypoint từ video (gọi model pose RTMPose-m / YOLO-Pose /
           BlazePose, xuất tọa độ + confidence theo từng frame).
Input    : Video hợp lệ đã được capture_guide.py xác nhận đạt chuẩn.
Output   : Danh sách PoseFrame (xem core/pose_schema.py) -> đưa vào tracker.py.

Thiết kế theo pattern "strategy": PoseExtractor không tự cài đặt thuật toán
pose mà nhận vào một PoseModel (RTMPose, MediaPipe hoặc YOLO-Pose).

Backend mặc định của hệ thống (V2.2): RTMPose-m ONNX + YOLO/ByteTrack.
  - YOLO chỉ dùng để DETECT + TRACK người (ByteTrack built-in ultralytics).
  - RTMPose-m (ONNX Runtime) nhận crop từng người, trả keypoint COCO17
    chính xác cao, robust hơn với occlusion.
  - Không cần mmcv/mmpose/mmdet.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from core.pose_schema import COCO17_NAMES, POSE33_NAMES, PoseFrame

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# 1. Interface chung cho mọi pose model (strategy pattern)
# ---------------------------------------------------------------------------
class PoseModel(ABC):
    """Mọi backend pose (RTMPose, YOLO-Pose, BlazePose, ...) implement lớp này."""

    joint_names: Sequence[str]

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> List["RawDetection"]:
        """Chạy model trên 1 frame, trả về danh sách người phát hiện được.

        Trả về list rỗng nếu không phát hiện người nào. Trả về nhiều phần tử
        nếu có nhiều người trong khung (tracker.py sẽ lọc lấy 1 người chính).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Giải phóng tài nguyên model (override nếu cần)."""
        return None


@dataclass
class RawDetection:
    """Kết quả thô của model cho MỘT người trong MỘT frame."""

    xy: np.ndarray  # (num_joints, 2)
    conf: np.ndarray  # (num_joints,)
    bbox: Optional[np.ndarray] = None  # (4,) x1,y1,x2,y2
    det_score: float = 1.0
    track_id: Optional[int] = None  # ByteTrack gán (nếu có)


# ---------------------------------------------------------------------------
# 2. Backend: YOLO-Pose (ultralytics) - 17 keypoint COCO
# ---------------------------------------------------------------------------
class YoloPoseModel(PoseModel):
    """Backend YOLOv8/YOLO11-Pose (17 keypoint COCO).

    Vẫn được giữ để tương thích ngược và làm fallback. Trong hệ thống V2.2,
    nên dùng RTMPoseONNXModel thay thế.
    """

    joint_names = COCO17_NAMES

    def __init__(self, weights: str = "yolo11n-pose.pt", conf_thres: float = 0.25,
                 device: str = "cpu"):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Cần cài `ultralytics` để dùng YoloPoseModel: pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self._conf_thres = conf_thres
        self._device = device

    def infer(self, frame_bgr: np.ndarray) -> List[RawDetection]:
        results = self._model.predict(
            frame_bgr, conf=self._conf_thres, device=self._device, verbose=False
        )
        detections: List[RawDetection] = []
        if not results:
            return detections
        r = results[0]
        if r.keypoints is None:
            return detections
        kpts_xy = r.keypoints.xy.cpu().numpy()
        kpts_conf = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints.conf is not None
            else np.ones(kpts_xy.shape[:2], dtype=np.float32)
        )
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
        det_scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else None
        for i in range(kpts_xy.shape[0]):
            detections.append(
                RawDetection(
                    xy=kpts_xy[i],
                    conf=kpts_conf[i],
                    bbox=None if boxes is None else boxes[i],
                    det_score=1.0 if det_scores is None else float(det_scores[i]),
                )
            )
        return detections


# ---------------------------------------------------------------------------
# 3. Backend: MediaPipe BlazePose - 33 keypoint (có heel/foot_index)
# ---------------------------------------------------------------------------
class MediaPipePoseModel(PoseModel):
    """Backend MediaPipe Pose (single-person, đủ điểm bàn chân).

    Vẫn giữ để tương thích ngược.
    """

    joint_names = POSE33_NAMES

    _MP_ORDER = [
        "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
        "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_EYE_INNER", "RIGHT_EYE_INNER",
        "LEFT_EYE_OUTER", "RIGHT_EYE_OUTER",
    ]

    def __init__(self, model_complexity: int = 1, min_det_conf: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Cần cài `mediapipe` để dùng MediaPipePoseModel: pip install mediapipe"
            ) from exc
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_det_conf,
        )
        self._landmark_enum = mp.solutions.pose.PoseLandmark

    def infer(self, frame_bgr: np.ndarray) -> List[RawDetection]:
        if cv2 is None:
            raise RuntimeError("Cần cài opencv-python để chuyển đổi màu ảnh")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return []
        h, w = frame_bgr.shape[:2]
        xy = np.zeros((len(self._MP_ORDER), 2), dtype=np.float32)
        conf = np.zeros((len(self._MP_ORDER),), dtype=np.float32)
        for i, name in enumerate(self._MP_ORDER):
            lm = result.pose_landmarks.landmark[getattr(self._landmark_enum, name)]
            xy[i] = [lm.x * w, lm.y * h]
            conf[i] = getattr(lm, "visibility", 1.0)
        xs, ys = xy[:, 0], xy[:, 1]
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        return [RawDetection(xy=xy, conf=conf, bbox=bbox, det_score=1.0)]

    def close(self) -> None:
        self._pose.close()


# ---------------------------------------------------------------------------
# 4. Backend: RTMPose-m ONNX + YOLO/ByteTrack  (khuyến nghị, không cần mmcv)
# ---------------------------------------------------------------------------
class RTMPoseONNXModel(PoseModel):
    """RTMPose-m ONNX Runtime + YOLO/ByteTrack.

    Kiến trúc:
        - YOLO (ultralytics) : detect + track người (ByteTrack built-in).
        - RTMPose-m (ONNX)   : nhận crop từng người, trả keypoint COCO17.

    Không cần mmcv/mmpose/mmdet — chỉ cần `onnxruntime-gpu` + `ultralytics`.
    Yêu cầu file .onnx export từ checkpoint RTMPose-m.
    """

    joint_names = COCO17_NAMES

    # Chuẩn hóa ảnh theo config RTMPose-m
    _MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    _STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    _INPUT_W = 192
    _INPUT_H = 256

    def __init__(
        self,
        yolo_weights: str = "yolo11n-pose.pt",
        yolo_conf: float = 0.25,
        yolo_imgsz: int = 640,
        yolo_device: str = "cuda:0",
        tracker_yaml: str = "bytetrack.yaml",
        rtmpose_onnx: str = "weights/rtmpose-m.onnx",
        rtmpose_device: str = "cuda",
        rtmpose_bbox_pad: float = 0.10,
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("Cần cài `ultralytics`: pip install ultralytics") from exc
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError("Cần cài `onnxruntime-gpu`: pip install onnxruntime-gpu") from exc

        self._yolo = YOLO(yolo_weights)
        # Chuyển YOLO sang device ngay lập tức (bắt buộc cho ultralytics mới)
        try:
            if yolo_device.startswith("cuda"):
                self._yolo.to(yolo_device)
                print(f"[RTMPoseONNXModel] YOLO đã chuyển sang {yolo_device}")
        except Exception as exc:
            print(f"[RTMPoseONNXModel] Không chuyển YOLO sang {yolo_device}: {exc}")

        # Warmup để init CUDA context
        try:
            import numpy as _np
            dummy = _np.zeros((640, 640, 3), dtype=_np.uint8)
            self._yolo.predict(dummy, device=yolo_device, verbose=False)
        except Exception as exc:
            print(f"[RTMPoseONNXModel] Warmup YOLO lỗi: {exc}")

        self._yolo_conf = yolo_conf
        self._yolo_imgsz = yolo_imgsz
        self._yolo_device = yolo_device
        self._tracker_yaml = tracker_yaml

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if rtmpose_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._sess = ort.InferenceSession(rtmpose_onnx, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self._bbox_pad = rtmpose_bbox_pad

    # ------------------------------------------------------------------
    def _pad_bbox(self, bbox: np.ndarray, w: int, h: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * self._bbox_pad, bh * self._bbox_pad
        return np.array(
            [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray, bbox: np.ndarray):
        """Crop + resize + normalize -> tensor (1, 3, H, W)."""
        if cv2 is None:
            raise RuntimeError("Cần opencv-python-headless")
        x1, y1, x2, y2 = bbox.astype(int)
        crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self._INPUT_W, self._INPUT_H))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - self._MEAN) / self._STD
        chw = rgb.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
        return np.ascontiguousarray(chw)

    # ------------------------------------------------------------------
    def _postprocess_simcc(self, outputs, bbox: np.ndarray):
        """Decode SimCC output -> keypoint (K, 2) pixel frame gốc + conf (K,).

        RTMPose ONNX export trả về 2 output dạng LOGIT (giá trị có thể âm):
            simcc_x: (1, K, W*2)
            simcc_y: (1, K, H*2)

        Conf được lấy trực tiếp từ logit max (KHÔNG softmax) — theo đúng
        cách mmpose decode SimCC (xem `mmpose.codecs.SimCCLabelCoder`).
        Softmax trên 384 bins sẽ cho giá trị ~0.005 — sai.
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        simcc_x = simcc_y = None
        for name, val in zip(self._output_names, outputs):
            arr = np.asarray(val)
            if arr.ndim == 3 and "x" in name.lower():
                simcc_x = arr[0]
            elif arr.ndim == 3 and "y" in name.lower():
                simcc_y = arr[0]
        if simcc_x is None or simcc_y is None:
            simcc_x, simcc_y = np.asarray(outputs[0])[0], np.asarray(outputs[1])[0]

        K = simcc_x.shape[0]
        xy = np.zeros((K, 2), dtype=np.float32)
        conf = np.zeros((K,), dtype=np.float32)

        for k in range(K):
            px = int(np.argmax(simcc_x[k]))
            py = int(np.argmax(simcc_y[k]))

            # Conf = max logit giữa 2 trục (KHÔNG softmax)
            # Đây là cách mmpose decode SimCC cho RTMPose
            conf[k] = float(max(simcc_x[k].max(), simcc_y[k].max()))

            # Chuyển bin index -> pixel trong crop
            xy[k, 0] = x1 + (px / (simcc_x.shape[1] - 1)) * bw
            xy[k, 1] = y1 + (py / (simcc_y.shape[1] - 1)) * bh

        return xy, conf

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> List[RawDetection]:
        # 1) YOLO + ByteTrack
        results = self._yolo.track(
            frame_bgr,
            conf=self._yolo_conf,
            imgsz=self._yolo_imgsz,
            device=self._yolo_device,
            tracker=self._tracker_yaml,
            persist=True,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        tids = (
            r.boxes.id.cpu().numpy().astype(int)
            if getattr(r.boxes, "id", None) is not None
            else np.full(len(boxes), -1, dtype=int)
        )

        h, w = frame_bgr.shape[:2]
        detections: List[RawDetection] = []

        # 2) RTMPose ONNX cho từng người
        for box, score, tid in zip(boxes, scores, tids):
            padded = self._pad_bbox(box, w, h)
            inp = self._preprocess(frame_bgr, padded)
            if inp is None:
                continue
            try:
                outs = self._sess.run(self._output_names, {self._input_name: inp})
            except Exception:
                continue
            xy, conf = self._postprocess_simcc(outs, padded)

            if xy.shape[0] > len(COCO17_NAMES):
                xy = xy[: len(COCO17_NAMES)]
                conf = conf[: len(COCO17_NAMES)]

            detections.append(
                RawDetection(
                    xy=xy,
                    conf=conf,
                    bbox=padded,
                    det_score=float(score),
                    track_id=None if tid < 0 else int(tid),
                )
            )
        return detections

    def close(self) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5. PoseExtractor: chạy model trên toàn bộ video
# ---------------------------------------------------------------------------
class PoseExtractor:
    """Bọc một PoseModel để xử lý cả video, sinh ra chuỗi PoseFrame."""

    def __init__(self, model: PoseModel):
        self._model = model

    def extract_video(self, video_path: str) -> Iterator[List[PoseFrame]]:
        if cv2 is None:
            raise RuntimeError("Cần cài opencv-python: pip install opencv-python-headless")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections = self._model.infer(frame)
                pose_frames = [
                    PoseFrame(
                        frame_idx=frame_idx,
                        timestamp_s=frame_idx / fps,
                        xy=det.xy,
                        conf=det.conf,
                        joint_names=self._model.joint_names,
                        bbox=det.bbox,
                        track_id=det.track_id,
                    )
                    for det in detections
                ]
                yield pose_frames
                frame_idx += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._model.close()

    @staticmethod
    def save_sequence(sequence: List[PoseFrame], out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        payload = [pf.to_dict() for pf in sequence]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @staticmethod
    def load_sequence(in_path: str) -> List[PoseFrame]:
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [PoseFrame.from_dict(d) for d in payload]


# ---------------------------------------------------------------------------
# 6. Factory
# ---------------------------------------------------------------------------
def build_pose_extractor(backend: str = "rtmpose", **kwargs) -> PoseExtractor:
    backend = backend.lower()
    if backend in ("rtmpose", "rtmpose-m", "rtm"):
        model: PoseModel = RTMPoseONNXModel(**kwargs)
    elif backend in ("yolo", "yolo-pose", "yolov8", "yolo11"):
        model = YoloPoseModel(**kwargs)
    elif backend in ("mediapipe", "blazepose"):
        model = MediaPipePoseModel(**kwargs)
    else:
        raise ValueError(f"Backend pose không được hỗ trợ: {backend}")
    return PoseExtractor(model)