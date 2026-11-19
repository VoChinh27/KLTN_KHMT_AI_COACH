"""
stage3_features/phase_metrics.py
==================================
Mục đích : Bộ chỉ số THỐNG NHẤT để so mọi phương pháp phân đoạn pha
           (heuristic cũ, mô hình HSMM, các phiên bản sau).

Năm biên của mỗi clip (frame_idx, end INCLUSIVE như file nhãn):
    start:vao_cho, start:static, start:sprint_start, start:drive_phase,
    end:last  (kết thúc của pha có mặt cuối cùng)
Biên của pha vắng mặt = None.

Chỉ số:
    boundary_summary : sai số có dấu theo từng biên (bias, MAE, trung vị, p90),
                       tỉ lệ trong dung sai, số biên bị bỏ sót / bịa thêm.
    frame_macro_f1   : F1 trung bình 6 lớp trên từng frame (bỏ IGNORE).
    segment_f1       : F1 mức đoạn theo IoU (mặc định 0.5) cho từng pha.
    absent_accuracy  : độ chính xác nhận biết pha vắng mặt.
    order_violations : số cặp pha bị đảo/chồng thứ tự (mục tiêu = 0).
    ece              : sai số hiệu chỉnh kỳ vọng của độ tin cậy.
    error_detection_recall : trong các biên sai quá dung sai, bao nhiêu % bị
                       gắn độ tin cậy thấp (tức là người duyệt sẽ để ý).
    review_effort    : số biên phải sửa / clip và số frame phải dời.
    bootstrap_ci, paired_permutation_test : khoảng tin cậy theo CLIP và kiểm
                       định ghép cặp giữa 2 phương pháp (không cần scipy).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.phase_schema import PHASE_ORDER, Rep
from stage3_features.phase_model.core.phase_labels import IGNORE_LABEL, N_CLASSES

BOUNDARIES: Tuple[str, ...] = tuple(f"start:{p}" for p in PHASE_ORDER) + ("end:last",)
Tol = Union[int, Dict[str, int]]


def rep_boundaries(rep: Optional[Rep]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {b: None for b in BOUNDARIES}
    if rep is None:
        return out
    present = rep.present()
    for name, w in present:
        out[f"start:{name}"] = int(w.start_frame)
    if present:
        out["end:last"] = int(present[-1][1].end_frame)
    return out


def _tol_for(tol: Tol, boundary: str) -> int:
    return int(tol[boundary]) if isinstance(tol, dict) else int(tol)


# ---------------------------------------------------------------------
# Biên
# ---------------------------------------------------------------------
def boundary_errors(truths: Sequence[Rep], preds: Sequence[Optional[Rep]]
                    ) -> Dict[str, Dict[str, list]]:
    """Với mỗi biên: danh sách sai số có dấu (pred - truth) khi cả hai có mặt,
    cùng chỉ số clip; thêm danh sách bỏ sót / bịa thêm."""
    res = {b: {"err": [], "clip": [], "missed": [], "spurious": []} for b in BOUNDARIES}
    for i, (t, p) in enumerate(zip(truths, preds)):
        tb, pb = rep_boundaries(t), rep_boundaries(p)
        for b in BOUNDARIES:
            if tb[b] is not None and pb[b] is not None:
                res[b]["err"].append(pb[b] - tb[b]); res[b]["clip"].append(i)
            elif tb[b] is not None and pb[b] is None:
                res[b]["missed"].append(i)
            elif tb[b] is None and pb[b] is not None:
                res[b]["spurious"].append(i)
    return res


def boundary_summary(truths: Sequence[Rep], preds: Sequence[Optional[Rep]], tol: Tol = 3
                     ) -> Dict[str, Dict[str, float]]:
    res = boundary_errors(truths, preds)
    out: Dict[str, Dict[str, float]] = {}
    for b in BOUNDARIES:
        e = np.asarray(res[b]["err"], dtype=float)
        n_truth = len(e) + len(res[b]["missed"])
        t = _tol_for(tol, b)
        if len(e) == 0:
            out[b] = {"n": 0, "n_truth": n_truth, "tol": t, "within_tol": float("nan"),
                      "within_tol_all": 0.0 if n_truth else float("nan"),
                      "bias": float("nan"), "mae": float("nan"), "median_abs": float("nan"),
                      "p90_abs": float("nan"), "missed": len(res[b]["missed"]),
                      "spurious": len(res[b]["spurious"])}
            continue
        a = np.abs(e)
        within = float(np.mean(a <= t))
        out[b] = {
            "n": int(len(e)), "n_truth": n_truth, "tol": t,
            "within_tol": within,                                   # trên các biên cả hai cùng có
            "within_tol_all": float(np.sum(a <= t) / max(1, n_truth)),  # bỏ sót tính là sai
            "bias": float(e.mean()), "mae": float(a.mean()),
            "median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)),
            "missed": len(res[b]["missed"]), "spurious": len(res[b]["spurious"]),
        }
    return out


def per_clip_within_tol(truths: Sequence[Rep], preds: Sequence[Optional[Rep]], tol: Tol = 3
                        ) -> np.ndarray:
    """Tỉ lệ biên đúng (trong dung sai) của TỪNG clip; bỏ sót/bịa = sai. Dùng
    cho bootstrap và kiểm định ghép cặp."""
    out = []
    for t, p in zip(truths, preds):
        tb, pb = rep_boundaries(t), rep_boundaries(p)
        hit, tot = 0, 0
        for b in BOUNDARIES:
            if tb[b] is None and pb[b] is None:
                continue
            tot += 1
            if tb[b] is not None and pb[b] is not None and abs(pb[b] - tb[b]) <= _tol_for(tol, b):
                hit += 1
        out.append(hit / tot if tot else float("nan"))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------
# Frame / đoạn
# ---------------------------------------------------------------------
def frame_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ok = (y_true != IGNORE_LABEL) & (y_pred != IGNORE_LABEL)
    yt, yp = y_true[ok], y_pred[ok]
    f1s: Dict[str, float] = {}
    for c in range(N_CLASSES):
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        if tp + fp + fn == 0:
            continue
        f1s[str(c)] = 2 * tp / (2 * tp + fp + fn)
    acc = float(np.mean(yt == yp)) if len(yt) else float("nan")
    return {"macro_f1": float(np.mean(list(f1s.values()))) if f1s else float("nan"),
            "accuracy": acc, "per_class": f1s}


def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def segment_f1(truths: Sequence[Rep], preds: Sequence[Optional[Rep]], iou_thr: float = 0.5
               ) -> Dict[str, float]:
    out: Dict[str, float] = {}
    f1s = []
    for ph in PHASE_ORDER:
        tp = fp = fn = 0
        for t, p in zip(truths, preds):
            tw = t.phases[ph]; pw = p.phases[ph] if p is not None else None
            t_has = not tw.absent
            p_has = pw is not None and not pw.absent
            hit = t_has and p_has and _iou((tw.start_frame, tw.end_frame),
                                           (pw.start_frame, pw.end_frame)) >= iou_thr
            tp += int(hit); fp += int(p_has and not hit); fn += int(t_has and not hit)
        out[ph] = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else float("nan")
        if np.isfinite(out[ph]):
            f1s.append(out[ph])
    out["macro"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def absent_accuracy(truths: Sequence[Rep], preds: Sequence[Optional[Rep]]) -> Dict[str, float]:
    correct = total = 0
    t_abs = hit_abs = 0
    for t, p in zip(truths, preds):
        if p is None:
            continue
        for ph in PHASE_ORDER:
            ta, pa = t.phases[ph].absent, p.phases[ph].absent
            total += 1; correct += int(ta == pa)
            if ta:
                t_abs += 1; hit_abs += int(pa)
    return {"accuracy": correct / total if total else float("nan"),
            "n_truth_absent": t_abs,
            "absent_recall": hit_abs / t_abs if t_abs else float("nan")}


def order_violations(rep: Optional[Rep]) -> int:
    if rep is None:
        return 0
    pr = [w for _, w in rep.present()]
    return sum(1 for a, b in zip(pr, pr[1:]) if b.start_frame <= a.end_frame)


# ---------------------------------------------------------------------
# Độ tin cậy
# ---------------------------------------------------------------------
def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    ok = np.isfinite(conf)
    conf, correct = conf[ok], correct[ok]
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def error_detection_recall(conf: Sequence[float], abs_err: Sequence[float], tol: int,
                           low_conf_threshold: float = 0.6) -> Dict[str, float]:
    """Trong các biên SAI (|err| > tol), bao nhiêu % có độ tin cậy < ngưỡng
    (tức là sẽ bị đánh dấu để duyệt). Và ngược lại: trong các biên 'tin cao',
    bao nhiêu % thực sự đúng (precision của 'tin cao')."""
    conf, e = np.asarray(conf, float), np.asarray(abs_err, float)
    wrong = e > tol
    high = conf >= low_conf_threshold
    return {
        "recall_of_errors": float(np.mean(~high[wrong])) if wrong.any() else float("nan"),
        "precision_of_high": float(np.mean(~wrong[high])) if high.any() else float("nan"),
        "n_wrong": int(wrong.sum()), "n_high": int(high.sum()),
    }


def review_effort(truths: Sequence[Rep], preds: Sequence[Optional[Rep]], tol: Tol = 3
                  ) -> Dict[str, float]:
    """Công người duyệt phải bỏ ra nếu sửa mọi biên sai quá dung sai."""
    n_fix, frames = [], []
    for t, p in zip(truths, preds):
        tb, pb = rep_boundaries(t), rep_boundaries(p)
        c, f = 0, 0
        for b in BOUNDARIES:
            if tb[b] is None and pb[b] is None:
                continue
            if tb[b] is None or pb[b] is None:
                c += 1; f += 0
            elif abs(pb[b] - tb[b]) > _tol_for(tol, b):
                c += 1; f += abs(pb[b] - tb[b])
        n_fix.append(c); frames.append(f)
    return {"boundaries_to_fix_per_clip": float(np.mean(n_fix)) if n_fix else float("nan"),
            "frames_to_move_per_clip": float(np.mean(frames)) if frames else float("nan")}


# ---------------------------------------------------------------------
# Thống kê suy luận (theo CLIP, không theo frame)
# ---------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], stat=np.nanmean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(stat(v)), float(np.percentile(stats, 100 * alpha / 2)), \
        float(np.percentile(stats, 100 * (1 - alpha / 2)))


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                            seed: int = 0) -> Dict[str, float]:
    """Kiểm định ghép cặp 2 phía (đảo dấu ngẫu nhiên) cho hiệu a-b theo clip."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "mean_diff": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "p_value": float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))}
