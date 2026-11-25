"""Trích pose 3D bằng MediaPipe từ video, lưu/đọc bảng pose, và đối tượng PoseData (mảng numpy theo khớp)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
MP_IDX = dict(zip(JOINTS, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]))


FOOT_JOINTS = ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]      # gót và mũi chân (dùng kiểm tra lỗi chân khi phát cầu)
MP_FOOT = dict(zip(FOOT_JOINTS, [29, 30, 31, 32]))
ALL_MP = {**MP_IDX, **MP_FOOT}
_FIELDS = ("x", "y", "vis", "wx", "wy", "wz")


def _bbox(lms, w: int, h: int, vis_min: float = 0.3):
    """Hộp bao (px) của các khớp thân người nhìn thấy; None nếu quá ít khớp."""
    pts = [(lms[i].x * w, lms[i].y * h) for i in MP_IDX.values() if lms[i].visibility >= vis_min]
    if len(pts) < 4:
        return None
    a = np.array(pts)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PersonTracker:
    """Chọn ĐÚNG người cần chấm khi trong khung có nhiều người (không cần YOLO): chọn người đầu tiên theo `select`
    (largest = hộp lớn nhất · center = gần tâm khung · roi = trùng vùng person_roi do người dùng khai báo), rồi bám theo bằng IoU/khoảng cách tâm
    giữa các frame. Mất dấu quá `lost_frames` thì chọn lại theo cách ban đầu. `switches` đếm số lần phải chọn lại (cảnh báo chất lượng)."""

    def __init__(self, w: int, h: int, select: str = "largest", roi=None, lost_frames: int = 15):
        self.w, self.h, self.select, self.roi, self.lost_frames = w, h, select, roi, lost_frames
        self.last, self.lost, self.switches, self.first = None, 0, 0, True

    def _initial(self, cands):
        if self.roi is not None:
            r = (self.roi[0] * self.w, self.roi[1] * self.h, self.roi[2] * self.w, self.roi[3] * self.h)
            return max(cands, key=lambda c: _iou(c[1], r))
        if self.select == "center":
            cx, cy = self.w / 2, self.h / 2
            return min(cands, key=lambda c: ((c[1][0] + c[1][2]) / 2 - cx) ** 2 + ((c[1][1] + c[1][3]) / 2 - cy) ** 2)
        return max(cands, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))

    def pick(self, cands):
        """cands = [(chỉ số, hộp)] -> chỉ số người được chọn hoặc None."""
        if not cands:
            self.lost += 1
            return None
        if self.last is None:
            best = self._initial(cands)
        else:
            diag = (self.w ** 2 + self.h ** 2) ** 0.5
            cx0, cy0 = (self.last[0] + self.last[2]) / 2, (self.last[1] + self.last[3]) / 2
            score = lambda c: _iou(c[1], self.last) + 0.5 * max(0.0, 1 - (((c[1][0] + c[1][2]) / 2 - cx0) ** 2 + ((c[1][1] + c[1][3]) / 2 - cy0) ** 2) ** 0.5 / (0.25 * diag))
            best = max(cands, key=score)
            if score(best) < 0.15:                                  # không còn ai gần người đang theo dõi
                self.lost += 1
                if self.lost <= self.lost_frames:
                    return None
                best = self._initial(cands)
                self.switches += 1
        self.lost = 0
        self.last = best[1]
        return best[0]


def _cand(img_lms, wor_lms, off, w_src: int, h_src: int, src: str):
    """Ứng viên người: toạ độ px TOÀN KHUNG (33 điểm x [x, y, vis]), world (33 x 3), hộp bao thân. off = (x0, y0) của vùng đã cắt."""
    pts = np.array([[off[0] + p.x * w_src, off[1] + p.y * h_src, p.visibility] for p in img_lms])
    wor = np.array([[q.x, q.y, q.z] for q in wor_lms])
    sel = [pts[i] for i in MP_IDX.values() if pts[i, 2] >= 0.3]
    if len(sel) < 4:
        return None
    a = np.array(sel)
    return {"pts": pts, "wor": wor, "bbox": (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())), "src": src,
            "vis": float(np.mean([pts[i, 2] for i in MP_IDX.values()]))}


def _detect_crop(lm, mp, fr, box, margin: float, min_side: int, num: int = 1):
    """Top-down: cắt hình vuông quanh `box` (x (1 + 2 margin)), phóng to rồi dò pose; trả danh sách ứng viên (toạ độ đã quy về toàn khung)."""
    H, W = fr.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(min(max(max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin), min_side), min(W, H)))
    x0, y0 = int(min(max(cx - side / 2, 0), W - side)), int(min(max(cy - side / 2, 0), H - side))
    crop = fr[y0:y0 + side, x0:x0 + side]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return []
    up = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC) if side < 512 else crop
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
    out = [_cand(res.pose_landmarks[k], res.pose_world_landmarks[k], (x0, y0), side, side, "crop") for k in range(len(res.pose_landmarks))]
    return [c for c in out if c is not None]


def _tiles(w: int, h: int):
    """Các ô vuông (cạnh = cạnh ngắn của khung) phủ kín khung theo cạnh dài, chồng lấn 25%."""
    side = min(w, h)
    long_ = max(w, h)
    n = max(1, int(np.ceil((long_ - side) / (0.75 * side))) + 1) if long_ > side else 1
    pos = np.linspace(0, long_ - side, n).astype(int)
    return [(int(p), 0, side) if w >= h else (0, int(p), side) for p in pos]


def decode_yolov8(out: np.ndarray, scale: float, pad: tuple, conf: float = 0.35, iou: float = 0.5, channels_first: bool = True):
    """Giải mã đầu ra YOLOv8 ONNX (1, 4 + C, N) -> hộp người [(x0, y0, x1, y1, điểm)] theo toạ độ ảnh gốc.
    `scale` và `pad` = (px, py) là hệ số letterbox khi tiền xử lý. Lớp 0 = person."""
    o = np.squeeze(out, 0) if np.ndim(out) == 3 else np.asarray(out)
    if channels_first:
        o = o.T                                            # YOLOv8 xuất (4 + C, N) -> (N, 4 + C)
    score = o[:, 4]
    keep = score >= conf
    if not keep.any():
        return []
    b, sc = o[keep, :4], score[keep]
    xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
    xyxy = (xyxy - np.array([pad[0], pad[1], pad[0], pad[1]])) / scale
    idx = cv2.dnn.NMSBoxes([[float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])] for x in xyxy], [float(x) for x in sc], conf, iou)
    return [(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3]), float(sc[i])) for i in np.array(idx).reshape(-1)]


class YoloOnnxDetector:
    """Bộ dò người YOLOv8 (ONNX) chạy bằng onnxruntime: pose.detector = yolo_onnx + pose.yolo_model = models/yolov8n.onnx.
    Dự án KHÔNG kèm trọng số (xuất từ ultralytics: `yolo export model=yolov8n.pt format=onnx`)."""

    def __init__(self, path: str, conf: float = 0.35, imgsz: int = 640):
        import onnxruntime as ort
        self.sess, self.conf, self.imgsz = ort.InferenceSession(path, providers=["CPUExecutionProvider"]), conf, imgsz
        self.inp = self.sess.get_inputs()[0].name

    def detect(self, fr: np.ndarray):
        h, w = fr.shape[:2]
        sc = self.imgsz / max(h, w)
        nw, nh = int(round(w * sc)), int(round(h * sc))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(fr, (nw, nh))
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return decode_yolov8(self.sess.run(None, {self.inp: x})[0], sc, (px, py), self.conf)


def extract_pose(video_path: str, model_path: str, cfg_pose: Optional[dict] = None, max_frames: Optional[int] = None, roi=None) -> pd.DataFrame:
    """Chạy MediaPipe Pose Landmarker (IMAGE) trên từng frame -> bảng: 2D (px), visibility, 3D world (m, gốc = giữa hông), kèm gót/mũi chân.

    Chọn đúng người (nhiều người / người nhỏ trong khung rộng) theo thứ tự:
      1. detector = yolo_onnx: YOLO dò hộp người -> cắt vùng -> dò pose (top-down thật sự).
      2. mặc định (mediapipe): dò cả khung (tối đa num_poses người); frame nào không thấy đúng người đang theo dõi thì quét theo ô vuông (tiling)
         và cắt vùng quanh vị trí cũ (tracking crop).
    Sau đó PersonTracker chọn/bám người mục tiêu. crop_refine = luôn dò lại trên vùng cắt quanh người đã chọn, dùng nếu rõ hơn."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Không thấy model MediaPipe: {model_path}")
    c = cfg_pose or {}

    def make(n):
        return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE, num_poses=n,
            min_pose_detection_confidence=c.get("min_detection_conf", 0.5), min_pose_presence_confidence=c.get("min_presence_conf", 0.5),
            min_tracking_confidence=c.get("min_tracking_conf", 0.5)))

    lm, lm_one = make(int(c.get("num_poses", 3))), make(1)
    yolo = None
    if c.get("detector", "mediapipe") == "yolo_onnx":
        if not os.path.isfile(c.get("yolo_model", "")):
            raise FileNotFoundError(f"detector = yolo_onnx nhưng không thấy file mô hình: {c.get('yolo_model')} (xem models/README.md)")
        yolo = YoloOnnxDetector(c["yolo_model"], float(c.get("yolo_conf", 0.35)))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = PersonTracker(w, h, c.get("select", "largest"), roi, int(c.get("lost_frames", 15)))
    margin, min_side = float(c.get("crop_margin", 0.35)), int(c.get("crop_min_side", 256))
    tiling = c.get("tiling", "auto") in (True, "auto")
    tiles = _tiles(w, h)
    stem, rows, idx, n_tile = os.path.splitext(os.path.basename(video_path))[0], [], 0, 0

    def whole(fr):
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        return [x for x in (_cand(res.pose_landmarks[k], res.pose_world_landmarks[k], (0, 0), w, h, "whole") for k in range(len(res.pose_landmarks))) if x is not None]

    def tracked_match(cands):
        """Còn thấy người đang theo dõi (hộp chồng lấn hoặc tâm gần vị trí cũ, kể cả khi người di chuyển nhanh)?"""
        if tracker.last is None:
            return False
        L = tracker.last
        lc, lsz = ((L[0] + L[2]) / 2, (L[1] + L[3]) / 2), max(L[2] - L[0], L[3] - L[1], 1.0)
        return any(_iou(x["bbox"], L) > 0.1 or np.hypot((x["bbox"][0] + x["bbox"][2]) / 2 - lc[0], (x["bbox"][1] + x["bbox"][3]) / 2 - lc[1]) < 0.6 * lsz for x in cands)

    try:
        while True:
            ok, fr = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            if yolo is not None:
                cands = []
                for bx in yolo.detect(fr):
                    cands += _detect_crop(lm_one, mp, fr, bx[:4], margin, min_side)
            else:
                cands = whole(fr)
                if tiling and not tracked_match(cands):                 # mất người đang theo dõi / chưa thấy ai: tìm kỹ hơn
                    extra = []
                    if tracker.last is not None:
                        extra += _detect_crop(lm_one, mp, fr, tracker.last, 0.6, min_side)
                    if not extra or tracker.last is None:
                        for (tx, ty, side) in tiles:
                            crop = fr[ty:ty + side, tx:tx + side]
                            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                            extra += [x for x in (_cand(r.pose_landmarks[k], r.pose_world_landmarks[k], (tx, ty), side, side, "tile") for k in range(len(r.pose_landmarks))) if x is not None]
                    for x in extra:                                      # NMS: bỏ ứng viên trùng người đã có
                        if all(_iou(x["bbox"], y["bbox"]) < 0.5 for y in cands):
                            cands.append(x)
                    n_tile += 1
            k = tracker.pick([(i, x["bbox"]) for i, x in enumerate(cands)])
            chosen, refined = (cands[k] if k is not None else None), False
            if chosen is not None and c.get("crop_refine", False) and chosen["src"] == "whole":
                rc = _detect_crop(lm_one, mp, fr, chosen["bbox"], margin, min_side)
                if rc and rc[0]["vis"] >= chosen["vis"]:
                    chosen, refined = rc[0], True
            row = {"video": stem, "frame": idx, "fps_src": round(fps, 3), "width": w, "height": h, "detected": chosen is not None,
                   "n_people": len(cands), "refined": refined, "switches": tracker.switches}
            for j, i in ALL_MP.items():
                if chosen is not None:
                    pt, q = chosen["pts"][i], chosen["wor"][i]
                    row.update({f"{j}_x": pt[0], f"{j}_y": pt[1], f"{j}_vis": pt[2], f"{j}_wx": q[0], f"{j}_wy": q[1], f"{j}_wz": q[2]})
                else:
                    row.update({f"{j}_{s}": np.nan for s in _FIELDS})
            rows.append(row)
            idx += 1
    finally:
        cap.release()
        lm.close()
        lm_one.close()
    if not rows:
        raise IOError(f"Không đọc được frame nào (codec lạ? thử chuyển sang H.264): {video_path}")
    df = pd.DataFrame(rows)
    df.attrs["person_switches"], df.attrs["tile_frames"] = tracker.switches, n_tile
    return df


def save_pose(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, float_format="%.5f")


def load_pose(path: str, video: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if video is not None:
        df = df[df["video"] == video]
    return df.sort_values("frame").reset_index(drop=True)


@dataclass
class PoseData:
    video: str
    fps: float
    n: int
    frames: np.ndarray
    detected: np.ndarray
    xy: Dict[str, np.ndarray]      # (N,2) pixel
    vis: Dict[str, np.ndarray]     # (N,)
    w: Dict[str, np.ndarray]       # (N,3) mét, gốc giữa hông; x phải, y XUỐNG, z hướng camera
    width: int = 0
    height: int = 0
    foot: Optional[Dict[str, dict]] = None   # gót / mũi chân: {tên: {xy, w, vis}}; None nếu pose.csv cũ chưa có cột này
    n_people: Optional[np.ndarray] = None    # số người dò được mỗi frame (cảnh báo nhiều người trong khung)

    @classmethod
    def from_df(cls, df: pd.DataFrame, width: int = 0, height: int = 0) -> "PoseData":
        df = df.sort_values("frame").reset_index(drop=True)
        g = lambda c: df[c].to_numpy(float)
        det = df["detected"].astype(bool).to_numpy() if "detected" in df else np.ones(len(df), bool)
        foot = None
        if all(f"{j}_x" in df for j in FOOT_JOINTS):
            foot = {j: {"xy": np.c_[g(f"{j}_x"), g(f"{j}_y")], "vis": g(f"{j}_vis"), "w": np.c_[g(f"{j}_wx"), g(f"{j}_wy"), g(f"{j}_wz")]} for j in FOOT_JOINTS}
        return cls(video=str(df["video"].iloc[0]), fps=float(df["fps_src"].iloc[0]), n=len(df), frames=df["frame"].to_numpy(), detected=det, foot=foot,
                   n_people=g("n_people") if "n_people" in df else None,
                   xy={j: np.c_[g(f"{j}_x"), g(f"{j}_y")] for j in JOINTS}, vis={j: g(f"{j}_vis") for j in JOINTS},
                   w={j: np.c_[g(f"{j}_wx"), g(f"{j}_wy"), g(f"{j}_wz")] for j in JOINTS},
                   width=int(df["width"].iloc[0]) if "width" in df else width, height=int(df["height"].iloc[0]) if "height" in df else height)
