"""Tiện ích dùng chung cho bộ phân đoạn pha (phase segmenter).

Quy ước
-------
* 6 lớp mỗi frame: 0=pre, 1..4 = bốn pha, 5=post.
* 5 biên b0..b4 (chỉ số frame, end-exclusive): pha k (1..4) phủ [b[k-1], b[k]).
  Pha vắng mặt <=> độ dài 0 (b[k-1] == b[k]).
* Payload JSON (``*.phases.json``) gồm: schema, video, fps, n_frames, marked_by,
  boundaries_frames, phases[...] và các khối tùy chọn (auto_meta, auto_phases, review).

Nếu ``core/phase_schema.py`` của dự án dùng khóa khác, chỉ cần chỉnh
``make_payload`` / ``payload_boundaries`` / ``read_json`` / ``write_json_atomic`` tại đây.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

NUM_CLASSES = 6
NUM_PHASES = 4
NUM_BOUNDARIES = NUM_PHASES + 1
CLASS_NAMES = ("pre", "phase1", "phase2", "phase3", "phase4", "post")
PHASE_NAMES = ("vao_cho", "static", "sprint_start", "drive_phase")
SCHEMA_VERSION = 2

MARKED_MANUAL = "manual"
MARKED_AUTO = "auto"
MARKED_AUTO_REVIEWED = "auto_reviewed"


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Nhãn <-> biên
# --------------------------------------------------------------------------- #
def enforce_monotonic(boundaries: Sequence[float], n_frames: int) -> List[int]:
    """Ép 5 biên thành số nguyên không giảm trong [0, n_frames]."""
    b = np.asarray(boundaries, dtype=np.float64)
    if b.shape != (NUM_BOUNDARIES,):
        raise ValueError(f"Cần đúng {NUM_BOUNDARIES} biên, nhận {b.shape}")
    b = np.clip(np.rint(b), 0, n_frames).astype(np.int64)
    return [int(v) for v in np.maximum.accumulate(b)]


def boundaries_to_labels(boundaries: Sequence[int], n_frames: int) -> np.ndarray:
    b = enforce_monotonic(boundaries, n_frames)
    labels = np.zeros(n_frames, dtype=np.int64)
    for k in range(NUM_BOUNDARIES):
        labels[b[k]:] = k + 1
    return labels


def labels_to_boundaries(labels: Sequence[int]) -> List[int]:
    lab = np.maximum.accumulate(np.asarray(labels, dtype=np.int64))
    return [int(np.sum(lab <= k)) for k in range(NUM_BOUNDARIES)]


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def make_payload(video: str, fps: float, n_frames: int, boundaries: Sequence[float],
                 marked_by: str, **extra: Any) -> Dict[str, Any]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if int(n_frames) <= 0:
        raise ValueError("n_frames must be positive")
    b = enforce_monotonic(boundaries, n_frames)
    phase_map: Dict[str, Any] = {}
    for k, name in enumerate(PHASE_NAMES):
        start, end_exclusive = b[k], b[k + 1]
        phase_map[name] = ({"absent": True} if end_exclusive == start else
                           {"start_frame": int(start), "end_frame": int(end_exclusive - 1)})
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "video": video, "fps": float(fps),
        "total_frames": int(n_frames), "n_frames": int(n_frames),
        "marked_by": marked_by, "boundaries_frames": b, "phases": phase_map,
    }
    payload.update(extra)
    return payload


def payload_boundaries(payload: Dict[str, Any]) -> List[int]:
    """Chuẩn hóa payload cũ/mới thành 5 biên half-open [b0..b4]."""
    if not isinstance(payload, dict):
        raise TypeError("phase payload must be a JSON object")
    total_raw = payload.get("total_frames", payload.get("n_frames"))
    total = int(total_raw) if total_raw is not None else None
    if total is not None and total <= 0:
        raise ValueError("total_frames/n_frames must be positive")
    fps_raw = payload.get("fps")
    fps = float(fps_raw) if fps_raw is not None else None
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and positive")

    raw = payload.get("boundaries_frames", payload.get("boundaries"))
    if raw is not None:
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != NUM_BOUNDARIES:
            raise ValueError("boundary array must contain exactly five values")
        vals = np.asarray(raw, dtype=float)
        if not np.isfinite(vals).all() or not np.equal(vals, np.rint(vals)).all():
            raise ValueError("boundaries must be finite integer frame indices")
        b = [int(v) for v in vals]
    else:
        phases = payload.get("phases")
        if isinstance(phases, dict):
            normalized = {}
            for name in PHASE_NAMES:
                spec = phases.get(name)
                if spec is None or (isinstance(spec, dict) and spec.get("absent")):
                    normalized[name] = None
                    continue
                if not isinstance(spec, dict):
                    raise ValueError(f"phases.{name} must be an object")
                try:
                    start, end = int(spec["start_frame"]), int(spec["end_frame"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"phases.{name} needs integer start_frame/end_frame") from exc
                if start < 0 or end < start:
                    raise ValueError(f"phases.{name} has invalid range [{start}, {end}]")
                normalized[name] = (start, end + 1)
            present = [i for i, name in enumerate(PHASE_NAMES) if normalized[name] is not None]
            if not present:
                raise ValueError("phase payload has no present phase")
            first = present[0]
            first_start = normalized[PHASE_NAMES[first]][0]
            b = [first_start]
            cursor = first_start
            for i, name in enumerate(PHASE_NAMES):
                item = normalized[name]
                if item is None:
                    if i < first:
                        b.append(first_start)
                    else:
                        b.append(cursor)
                    continue
                start, end_exclusive = item
                if i == first and start != first_start:
                    raise ValueError("invalid first phase boundary")
                if i != first and start != cursor:
                    raise ValueError(f"gap/overlap before phases.{name}: expected {cursor}, got {start}")
                b.append(end_exclusive)
                cursor = end_exclusive
        elif isinstance(phases, (list, tuple)) and len(phases) >= NUM_PHASES:
            if any(not isinstance(item, dict) for item in phases[:NUM_PHASES]):
                raise ValueError("Legacy phase list entries must be objects")
            items = sorted(phases, key=lambda item: int(item.get("id", 0)))[:NUM_PHASES]
            rate = fps or 0.0
            def edge(item, key):
                frame_key, sec_key = f"{key}_frame", f"{key}_s"
                if item.get(frame_key) is not None: return int(item[frame_key])
                if item.get(sec_key) is not None and rate > 0: return int(round(float(item[sec_key]) * rate))
                raise ValueError(f"phase missing {frame_key}/{sec_key}")
            b = [edge(items[0], "start")]
            b.extend(edge(x, "end") + (0 if x.get("end_exclusive") else 1) for x in items)
        else:
            raise ValueError("Payload lacks boundaries or supported phases mapping/list")

    if len(b) != NUM_BOUNDARIES or b[0] < 0 or any(y < x for x, y in zip(b, b[1:])):
        raise ValueError("boundaries must be five monotonic non-negative integers")
    if total is not None and b[-1] > total:
        raise ValueError("boundary exceeds total_frames")
    return [int(v) for v in b]

def read_json(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: os.PathLike, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in o.tolist()]
    raise TypeError(f"Không serialize được {type(o)}")


def stem_from_phase_path(path: os.PathLike) -> str:
    name = Path(path).name
    return name[:-len(".phases.json")] if name.endswith(".phases.json") else Path(path).stem


# --------------------------------------------------------------------------- #
# Sai số biên
# --------------------------------------------------------------------------- #
def boundary_abs_errors(pred: Sequence[float], true: Sequence[float]) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float))


# --------------------------------------------------------------------------- #
# Nạp pose
# --------------------------------------------------------------------------- #
def _import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("Extractor phải có dạng 'package.module:function'")
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def _pose_array_from_json(data: Any) -> Tuple[np.ndarray, float]:
    """Đọc pose-cache dạng ``{fps, frames:[...]}`` hoặc danh sách frame.

    Mỗi frame chấp nhận ``xy`` Nx2 và ``conf`` N, hoặc ``poses`` Nx3.
    Placeholder NaN được giữ theo chỉ số frame, không bị loại bỏ.
    """
    if isinstance(data, dict) and "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        fps = float(data.get("fps", 0.0) or 0.0)
        return _ensure_pose_shape(poses), fps
    frames = data.get("frames") if isinstance(data, dict) else data
    if not isinstance(frames, list):
        raise ValueError("pose JSON phải là list frame hoặc object có frames/poses")
    rows = []
    times = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise ValueError(f"frame[{i}] không phải object")
        if "poses" in fr:
            row = np.asarray(fr["poses"], dtype=np.float32)
        elif "xy" in fr:
            xy = np.asarray(fr["xy"], dtype=np.float32)
            conf = np.asarray(fr.get("conf", np.ones(xy.shape[0])), dtype=np.float32).reshape(-1, 1)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(conf) != len(xy):
                raise ValueError(f"frame[{i}] xy/conf shape không hợp lệ")
            row = np.concatenate((xy, conf), axis=1)
        else:
            row = np.full((17, 3), np.nan, dtype=np.float32)
        if row.shape == (17, 2):
            row = np.concatenate((row, np.isfinite(row).all(axis=1, keepdims=True).astype(np.float32)), axis=1)
        if row.shape != (17, 3):
            raise ValueError(f"frame[{i}] phải có shape (17,2|3), nhận {row.shape}")
        rows.append(row)
        if fr.get("timestamp_s") is not None:
            times.append(float(fr["timestamp_s"]))
    if not rows:
        raise ValueError("pose cache rỗng")
    fps = float(data.get("fps", 0.0) or 0.0) if isinstance(data, dict) else 0.0
    if fps <= 0 and len(times) >= 2:
        dts = np.diff(times)
        dts = dts[dts > 0]
        if len(dts):
            fps = float(1.0 / np.median(dts))
    return np.stack(rows), fps


def _ensure_pose_shape(poses: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] not in (2, 3):
        raise ValueError(f"poses phải có shape (N,17,2|3), nhận {arr.shape}")
    if arr.shape[2] == 2:
        conf = np.isfinite(arr).all(axis=2, keepdims=True).astype(np.float32)
        arr = np.concatenate((arr, conf), axis=2)
    return arr


def load_pose_clip(stem: str, poses_dir: Optional[os.PathLike] = None,
                   video_path: Optional[os.PathLike] = None,
                   extractor: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Nạp ``<stem>.npz`` hoặc pose-cache JSON, rồi mới gọi extractor tùy chọn."""
    if poses_dir is not None:
        base = Path(poses_dir)
        npz_path = base / f"{stem}.npz"
        json_path = base / f"{stem}.json"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as z:
                return _ensure_pose_shape(z["poses"]), float(z["fps"])
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return _pose_array_from_json(json.load(f))
    if extractor and video_path:
        poses, fps = _import_callable(extractor)(str(video_path))
        return _ensure_pose_shape(np.asarray(poses)), float(fps)
    raise FileNotFoundError(
        f"Không có pose cho '{stem}' trong {poses_dir} (JSON/NPZ) và không có extractor")

def infer_is_coach(stem: str, pattern: str = r"(?i)hlv|coach") -> bool:
    return re.search(pattern, stem) is not None

