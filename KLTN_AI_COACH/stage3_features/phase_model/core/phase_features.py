"""Biểu diễn từng frame cho tầng 1 (bộ phân loại pha theo frame).

Đầu vào: ``poses`` dạng (N, 17, 3) COCO-17 (x, y, conf) hoặc (N, 17, 2).
Đầu ra: ``(X[N, D], names)``.

Quy trình
---------
1. Che keypoint có conf thấp / không hữu hạn -> NaN.
2. ``PoseNormalizer``: chia tọa độ cho độ dài thân (vai-hông) của chính clip.
   (Nếu dự án đã có lớp ``Normalizer`` riêng, thay lớp này bằng import tương ứng.)
3. Tín hiệu gốc: hip_above_shoulder, hand_ground_clearance, tốc độ/gia tốc COM,
   góc thân, knee_front/back, khuỷu, tốc độ cổ tay/mắt cá.
4. Nội suy khoảng NaN ngắn, làm mượt nhẹ; khoảng dài giữ NaN (có cột đánh dấu).
5. Chuẩn hóa theo baseline của chính clip (median/MAD), rồi cửa sổ đa thang
   0.25/0.5/1.0 s: nửa quá khứ, nửa tương lai và hiệu của chúng.
6. Cache vào ``data/phase_features_cache/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_VERSION = "pf1"
DEFAULT_CACHE_DIR = Path("data/phase_features_cache")

KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sh=5, r_sh=6, l_el=7, r_el=8,
          l_wr=9, r_wr=10, l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)

BASE_SIGNALS: Tuple[str, ...] = (
    "hip_above_shoulder", "hand_ground_clearance", "com_speed", "com_vx_dir", "com_vy",
    "com_acc", "trunk_angle", "knee_front", "knee_back", "elbow_mean", "elbow_diff",
    "wrist_speed", "ankle_speed",
)
EXTRA_COLUMNS: Tuple[str, ...] = (
    "visible_ratio", "is_coach", "view_is_front", "interp_flag", "nan_frac",
    "abs_hip_above_shoulder", "abs_hand_ground_clearance", "abs_trunk_angle",
    "abs_knee_front", "abs_log_torso_px",
)


TIME_COLUMNS: Tuple[str, ...] = ("t_sec", "t_frac", "t_left_sec", "cum_motion_frac")


@dataclass(frozen=True)
class FeatureConfig:
    scales_s: Tuple[float, ...] = (0.25, 0.5, 1.0)
    min_conf: float = 0.3
    max_gap_s: float = 0.3
    smooth_s: float = 0.1
    z_clip: float = 8.0
    ground_percentile: float = 95.0
    # --- tùy chọn mở rộng (mặc định tắt => feature giống hệt phiên bản trước) ---
    long_scales_s: Tuple[float, ...] = ()   # thang thời gian dài (giây), vd (2.0, 4.0)
    time_features: bool = False             # t_sec, t_frac, t_left_sec, cum_motion_frac
    include_torso_scale: bool = True        # False: bỏ abs_log_torso_px (hằng số theo clip)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["scales_s"] = list(self.scales_s)
        d["long_scales_s"] = list(self.long_scales_s)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "FeatureConfig":
        if not d:
            return cls()
        d = dict(d)
        if "scales_s" in d:
            d["scales_s"] = tuple(float(s) for s in d["scales_s"])
        if "long_scales_s" in d:
            d["long_scales_s"] = tuple(float(s) for s in d["long_scales_s"])
        return cls(**d)


# --------------------------------------------------------------------------- #
# Hàm tiện ích số học (NaN-aware)
# --------------------------------------------------------------------------- #
def _nanmedian(x: np.ndarray, default: float = np.nan) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else default


def _nanmean_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([a, b], axis=0), axis=0)


def _window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Trung bình x[t+lo : t+hi+1], cắt ở biên, bỏ qua NaN. NaN nếu cửa sổ rỗng."""
    n = len(x)
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    s = cs[b] - cs[a]
    c = cn[b] - cn[a]
    out = np.full(n, np.nan)
    m = c > 0
    out[m] = s[m] / c[m]
    return out


def _interp_short_gaps(x: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nội suy tuyến tính các đoạn NaN dài <= max_gap. Trả (x_out, mask_đã_nội_suy)."""
    x = np.asarray(x, dtype=float)
    nan = ~np.isfinite(x)
    out = x.copy()
    filled = np.zeros(len(x), dtype=bool)
    if not nan.any() or nan.all():
        return out, filled
    idx = np.arange(len(x))
    interp = np.interp(idx, idx[~nan], x[~nan])
    d = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s <= max_gap:
            out[s:e] = interp[s:e]
            filled[s:e] = True
    return out, filled


def _clean_series(x: np.ndarray, max_gap: int, smooth_w: int) -> Tuple[np.ndarray, np.ndarray]:
    y, filled = _interp_short_gaps(x, max_gap)
    if smooth_w > 1:
        half = smooth_w // 2
        sm = _window_mean(y, -half, half)
        y = np.where(np.isfinite(y), sm, np.nan)
    return y, filled


def _deriv(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x) * fps


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    cos = np.einsum("...i,...i->...", v1, v2) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Pose + chuẩn hóa
# --------------------------------------------------------------------------- #
def _as_pose_array(poses: np.ndarray, min_conf: float) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] not in (2, 3):
        raise ValueError(f"poses phải có shape (N,17,2|3) COCO-17, nhận {arr.shape}")
    xy = arr[..., :2].copy()
    finite = np.isfinite(xy).all(axis=-1)
    if arr.shape[2] == 3:
        conf = np.where(np.isfinite(arr[..., 2]), arr[..., 2], 0.0)
    else:
        conf = finite.astype(float)
    vis = finite & (conf >= min_conf)
    xy[~vis] = np.nan
    return xy, vis


class PoseNormalizer:
    """Chia tọa độ cho độ dài thân trung vị của clip; ước lượng mặt đất từ mắt cá."""

    def __init__(self, ground_percentile: float = 95.0):
        self.ground_percentile = ground_percentile
        self.scale_: float = np.nan
        self.ground_y_: float = np.nan

    def fit(self, xy: np.ndarray) -> "PoseNormalizer":
        sh = _nanmean_pair(xy[:, KP["l_sh"]], xy[:, KP["r_sh"]])
        hip = _nanmean_pair(xy[:, KP["l_hip"]], xy[:, KP["r_hip"]])
        torso = np.linalg.norm(sh - hip, axis=-1)
        scale = _nanmedian(torso)
        if not np.isfinite(scale) or scale <= 1e-6:
            raise ValueError("Clip không đủ keypoint vai/hông để ước lượng độ dài thân")
        self.scale_ = scale
        ank_y = xy[:, [KP["l_an"], KP["r_an"]], 1].ravel()
        ank_y = ank_y[np.isfinite(ank_y)]
        if ank_y.size == 0:
            raise ValueError("Clip không có keypoint mắt cá để ước lượng mặt đất")
        self.ground_y_ = float(np.percentile(ank_y, self.ground_percentile)) / scale
        return self

    def transform(self, xy: np.ndarray) -> np.ndarray:
        return xy / self.scale_


def _estimate_direction(P: np.ndarray, hint: Optional[str]) -> float:
    if hint == "front":
        return 1.0
    hip_x = _nanmean_pair(P[:, KP["l_hip"], 0], P[:, KP["r_hip"], 0])
    face = _nanmedian(P[:, KP["nose"], 0] - hip_x)
    if np.isfinite(face) and abs(face) > 0.05:
        return 1.0 if face > 0 else -1.0
    n = len(P)
    k = max(1, n // 5)
    if n >= 2:
        d = _nanmedian(hip_x[-k:]) - _nanmedian(hip_x[:k])
        if np.isfinite(d) and abs(d) > 1e-6:
            return 1.0 if d > 0 else -1.0
    return 1.0


# --------------------------------------------------------------------------- #
# Tín hiệu gốc
# --------------------------------------------------------------------------- #
def _compute_signals(P: np.ndarray, fps: float, hint: Optional[str], cfg: FeatureConfig,
                     ground_y: float) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    n = P.shape[0]
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    sw = max(1, int(round(cfg.smooth_s * fps)))
    filled_any = np.zeros(n, dtype=bool)

    def clean(x: np.ndarray) -> np.ndarray:
        nonlocal filled_any
        y, f = _clean_series(x, max_gap, sw)
        filled_any |= f
        return y

    kp = lambda name: P[:, KP[name], :]  # noqa: E731
    sh = _nanmean_pair(kp("l_sh"), kp("r_sh"))
    hip = _nanmean_pair(kp("l_hip"), kp("r_hip"))
    direction = _estimate_direction(P, hint)
    front = hint == "front"

    sig: Dict[str, np.ndarray] = {}
    sig["hip_above_shoulder"] = clean(sh[:, 1] - hip[:, 1])
    wr = _nanmean_pair(kp("l_wr"), kp("r_wr"))
    sig["hand_ground_clearance"] = clean(ground_y - wr[:, 1])

    com = _nanmean_pair(sh, hip)
    cx, cy = clean(com[:, 0]), clean(com[:, 1])
    vx, vy = _deriv(cx, fps), _deriv(cy, fps)
    sig["com_vx_dir"] = np.zeros(n) if front else vx * direction
    sig["com_vy"] = vy
    sig["com_speed"] = np.hypot(vx, vy)
    sig["com_acc"] = np.hypot(_deriv(vx, fps), _deriv(vy, fps))

    dx = sh[:, 0] - hip[:, 0]
    dx = np.abs(dx) if front else dx * direction
    sig["trunk_angle"] = clean(np.degrees(np.arctan2(hip[:, 1] - sh[:, 1], dx)))

    kl = _joint_angle(kp("l_hip"), kp("l_kn"), kp("l_an"))
    kr = _joint_angle(kp("r_hip"), kp("r_kn"), kp("r_an"))
    if front:  # không có chiều tiến -> chân gập hơn coi là chân trước
        kf, kb = np.fmin(kl, kr), np.fmax(kl, kr)
    else:
        lead_left = (kp("l_an")[:, 0] - kp("r_an")[:, 0]) * direction > 0
        kf, kb = np.where(lead_left, kl, kr), np.where(lead_left, kr, kl)
    sig["knee_front"], sig["knee_back"] = clean(kf), clean(kb)

    el = _joint_angle(kp("l_sh"), kp("l_el"), kp("l_wr"))
    er = _joint_angle(kp("r_sh"), kp("r_el"), kp("r_wr"))
    sig["elbow_mean"] = clean(_nanmean_pair(el, er))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sig["elbow_diff"] = clean(np.abs(el - er))

    def kp_speed(name: str) -> np.ndarray:
        x, y = clean(kp(name)[:, 0]), clean(kp(name)[:, 1])
        return np.hypot(_deriv(x, fps), _deriv(y, fps))

    sig["wrist_speed"] = _nanmean_pair(kp_speed("l_wr"), kp_speed("r_wr"))
    sig["ankle_speed"] = _nanmean_pair(kp_speed("l_an"), kp_speed("r_an"))

    stack = np.stack([sig[k] for k in BASE_SIGNALS], axis=0)
    nan_frac = np.mean(~np.isfinite(stack), axis=0)
    return sig, filled_any.astype(float), nan_frac


def _robust_z(x: np.ndarray, clip: float) -> np.ndarray:
    med = _nanmedian(x, 0.0)
    fin = x[np.isfinite(x)]
    mad = float(np.median(np.abs(fin - med))) * 1.4826 if fin.size else 0.0
    if mad < 1e-6:
        mad = float(np.std(fin)) if fin.size > 1 else 1.0
    mad = max(mad, 1e-3)
    return np.clip((x - med) / mad, -clip, clip)


# --------------------------------------------------------------------------- #
# Tên cột & cache
# --------------------------------------------------------------------------- #
def _tag(s: float) -> str:
    return f"{s:g}".replace(".", "p")


def _check_cfg(cfg: FeatureConfig) -> None:
    dup = set(cfg.long_scales_s) & set(cfg.scales_s)
    if dup or len(set(cfg.long_scales_s)) != len(cfg.long_scales_s):
        raise ValueError(f"long_scales_s không được trùng nhau hoặc trùng scales_s: {sorted(dup)}")
    if any(s <= 0 for s in cfg.long_scales_s):
        raise ValueError("long_scales_s phải > 0")


def feature_names(cfg: Optional[FeatureConfig] = None) -> List[str]:
    cfg = cfg or FeatureConfig()
    _check_cfg(cfg)
    extra = [c for c in EXTRA_COLUMNS if cfg.include_torso_scale or c != "abs_log_torso_px"]
    names = extra + [f"{s}__z" for s in BASE_SIGNALS]
    for s in BASE_SIGNALS:
        for sc in cfg.scales_s:
            t = _tag(sc)
            names += [f"{s}__past{t}", f"{s}__fut{t}", f"{s}__d{t}"]
    if cfg.time_features:
        names += list(TIME_COLUMNS)
    for s in BASE_SIGNALS:
        for sc in cfg.long_scales_s:
            t = _tag(sc)
            names += [f"{s}__past{t}", f"{s}__fut{t}", f"{s}__d{t}"]
    return names


def _cache_path(cache_dir: Path, poses: np.ndarray, fps: float, hint: Optional[str],
                is_coach: bool, cfg: FeatureConfig) -> Path:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(poses, dtype=np.float32)).tobytes())
    h.update(json.dumps([FEATURE_VERSION, round(float(fps), 4), hint, bool(is_coach),
                         cfg.to_dict()], sort_keys=True).encode())
    return cache_dir / f"{h.hexdigest()[:24]}.npz"


# --------------------------------------------------------------------------- #
# API chính
# --------------------------------------------------------------------------- #
def build_feature_matrix(poses: np.ndarray, fps: float, front_side_hint: Optional[str] = None,
                         *, is_coach: bool = False, cfg: Optional[FeatureConfig] = None,
                         cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
                         use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Dựng ma trận đặc trưng (N, D) và danh sách tên cột.

    ``front_side_hint``: ``"front"`` | ``"side"`` | ``None`` (coi như side).
    """
    cfg = cfg or FeatureConfig()
    _check_cfg(cfg)
    if front_side_hint not in (None, "front", "side"):
        raise ValueError("front_side_hint phải là 'front', 'side' hoặc None")
    names = feature_names(cfg)

    cpath = None
    if use_cache and cache_dir is not None:
        cpath = _cache_path(Path(cache_dir), poses, fps, front_side_hint, is_coach, cfg)
        if cpath.exists():
            try:
                with np.load(cpath, allow_pickle=False) as z:
                    if list(z["names"]) == names:
                        return z["X"], names
            except Exception as exc:  # cache hỏng -> tính lại
                logger.warning("Cache lỗi (%s): %s", cpath.name, exc)

    xy, vis = _as_pose_array(poses, cfg.min_conf)
    norm = PoseNormalizer(cfg.ground_percentile).fit(xy)
    P = norm.transform(xy)
    sig, interp_flag, nan_frac = _compute_signals(P, fps, front_side_hint, cfg, norm.ground_y_)
    n = P.shape[0]

    cols: List[np.ndarray] = [
        vis.mean(axis=1),
        np.full(n, float(is_coach)),
        np.full(n, 1.0 if front_side_hint == "front" else 0.0),
        interp_flag,
        nan_frac,
        sig["hip_above_shoulder"],
        sig["hand_ground_clearance"],
        sig["trunk_angle"] / 90.0,
        sig["knee_front"] / 180.0,
    ]
    if cfg.include_torso_scale:
        cols.append(np.full(n, np.log(max(norm.scale_, 1e-6))))
    z = {s: _robust_z(sig[s], cfg.z_clip) for s in BASE_SIGNALS}
    cols += [z[s] for s in BASE_SIGNALS]
    for s in BASE_SIGNALS:
        for sc in cfg.scales_s:
            w = max(1, int(round(sc * fps)))
            past = _window_mean(z[s], -w, -1)
            fut = _window_mean(z[s], 1, w)
            past = np.where(np.isfinite(past), past, z[s])
            fut = np.where(np.isfinite(fut), fut, z[s])
            cols += [past, fut, fut - past]

    if cfg.time_features:
        # Vị trí trong clip + tỉ lệ năng lượng chuyển động tích lũy (không cần ngưỡng).
        tt = np.arange(n, dtype=np.float64)
        motion = np.zeros(n)
        for k in ("com_speed", "ankle_speed"):
            motion += np.where(np.isfinite(z[k]), np.clip(z[k], 0.0, None), 0.0)
        cum = np.cumsum(motion)
        cols += [tt / fps, tt / max(n - 1, 1), (n - 1 - tt) / fps, cum / max(float(cum[-1]), 1e-6)]
    for s in BASE_SIGNALS:
        for sc in cfg.long_scales_s:
            w = max(1, int(round(sc * fps)))
            past = _window_mean(z[s], -w, -1)
            fut = _window_mean(z[s], 1, w)
            past = np.where(np.isfinite(past), past, z[s])
            fut = np.where(np.isfinite(fut), fut, z[s])
            cols += [past, fut, fut - past]

    X = np.column_stack(cols).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    assert X.shape[1] == len(names), (X.shape, len(names))

    if cpath is not None:
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, X=X, names=np.array(names))
        except OSError as exc:
            logger.warning("Không ghi được cache: %s", exc)
    return X, names
