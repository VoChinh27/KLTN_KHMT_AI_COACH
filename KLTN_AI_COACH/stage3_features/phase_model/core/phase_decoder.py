"""
stage3_features/phase_decoder.py
==================================
Mục đích : Tầng 2 của bộ phân đoạn pha — HSMM (semi-Markov) TRÁI-PHẢI trên
           6 trạng thái [outside_pre, vao_cho, static, sprint_start,
           drive_phase, outside_post].

Đầu vào  : log_emissions (N x 6) — log điểm từng frame, thường là
           log P(lớp|x) của bộ phân lớp tầng 1 huấn luyện class_weight=
           'balanced' (xem probs_to_log_emissions() và tham số prior_power).
Đầu ra   : DecodeResult — đoạn tối ưu (Viterbi) + hậu nghiệm biên
           (forward-backward) để tính độ tin cậy.

Thiết kế (bịt các lỗ hổng đã nêu trong đánh giá kế hoạch):
    1. PHA VẮNG MẶT : chỉ đi TIẾN, được NHẢY QUA pha giữa với phạt
       `skip_penalty` (log) cho mỗi pha bị bỏ qua. Nhảy qua outside_* miễn phí
       (chúng là tùy chọn, độ dài có thể bằng 0).
    2. outside_pre/post KHÔNG có phạt thời lượng (độ dài 1..N tùy ý).
    3. Pha giữa dùng prior thời lượng THEO GIÂY (log-normal rời rạc, có
       min/max), có shrinkage về prior rộng khi ít mẫu (fit_duration_priors).
    4. `emission_scale` (<1) làm dịu độ tự tin của các frame (frame liền kề
       tương quan mạnh nên cộng log-xác suất thô sẽ quá tự tin).

Độ phức tạp: O(N * S * Dmax) — vài trăm đến vài nghìn frame chạy trong giây.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.phase_schema import PHASE_ORDER, PhaseWindow, Rep
from stage3_features.phase_model.core.phase_labels import CLASSES, MIDDLE_IDS, N_CLASSES

NEG = -np.inf


# =====================================================================
# Prior thời lượng
# =====================================================================
@dataclass
class StatePrior:
    min_s: float
    max_s: float
    mu_log: float      # trung bình của ln(thời lượng giây)
    sigma_log: float   # độ lệch chuẩn của ln(thời lượng giây)

    def frame_range(self, fps: float, max_len: int) -> Tuple[int, int]:
        dmin = max(1, int(math.ceil(self.min_s * fps)))
        dmax = min(max_len, int(math.floor(self.max_s * fps)))
        return dmin, max(dmin, dmax)

    def log_pmf(self, fps: float, max_len: int) -> Tuple[np.ndarray, int, int]:
        """Mảng log-pmf chỉ số theo số frame d (0..max_len), -inf ngoài [dmin,dmax]."""
        dmin, dmax = self.frame_range(fps, max_len)
        out = np.full(max_len + 1, NEG)
        d = np.arange(dmin, dmax + 1, dtype=np.float64)
        lp = -np.log(d) - 0.5 * ((np.log(d / fps) - self.mu_log) / self.sigma_log) ** 2
        lp -= _lse(lp)  # chuẩn hóa để so sánh được với phạt skip
        out[dmin:dmax + 1] = lp
        return out, dmin, dmax


# Prior rộng mặc định (giây) — chỉ dùng khi chưa có dữ liệu; số liệu mẫu:
# vao_cho ~2.0s, static ~2.8s, sprint_start ~0.8s, drive_phase ~0.9s.
_WIDE_DEFAULTS: Dict[str, StatePrior] = {
    "vao_cho":      StatePrior(0.2, 15.0, math.log(2.0), 0.9),
    "static":       StatePrior(0.2, 15.0, math.log(2.0), 0.9),
    "sprint_start": StatePrior(0.1, 4.0,  math.log(0.8), 0.7),
    "drive_phase":  StatePrior(0.1, 6.0,  math.log(1.0), 0.8),
}


@dataclass
class DurationPriors:
    priors: Dict[str, StatePrior] = field(default_factory=lambda: dict(_WIDE_DEFAULTS))

    @classmethod
    def default(cls) -> "DurationPriors":
        return cls()

    @classmethod
    def fit(cls, durations_s: Dict[str, Sequence[float]], k_shrink: float = 5.0) -> "DurationPriors":
        """Ước lượng từ dữ liệu có SHRINKAGE về prior rộng: w = n/(n+k).
        durations_s: {phase: [thời lượng giây của từng clip có pha đó]}."""
        out: Dict[str, StatePrior] = {}
        for p in PHASE_ORDER:
            wide = _WIDE_DEFAULTS[p]
            x = np.asarray([v for v in durations_s.get(p, []) if v and v > 0], dtype=np.float64)
            n = len(x)
            if n < 2:
                out[p] = wide
                continue
            w = n / (n + k_shrink)
            lx = np.log(x)
            mu = w * float(lx.mean()) + (1 - w) * wide.mu_log
            sd = w * float(max(lx.std(ddof=1), 0.15)) + (1 - w) * wide.sigma_log
            min_s = w * 0.5 * float(x.min()) + (1 - w) * wide.min_s
            max_s = w * 2.0 * float(x.max()) + (1 - w) * wide.max_s
            out[p] = StatePrior(max(0.05, min_s), max(max_s, min_s * 2), mu, max(sd, 0.2))
        return cls(out)

    def to_dict(self) -> Dict:
        return {p: vars(v) for p, v in self.priors.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "DurationPriors":
        pri = dict(_WIDE_DEFAULTS)
        for p, v in d.items():
            if p in PHASE_ORDER:
                pri[p] = StatePrior(**v)
        return cls(pri)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DurationPriors":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# =====================================================================
# Tiện ích
# =====================================================================
def _lse(a: np.ndarray) -> float:
    if a.size == 0:
        return NEG
    m = float(np.max(a))
    if not np.isfinite(m):
        return NEG
    return m + float(np.log(np.sum(np.exp(a - m))))


def probs_to_log_emissions(proba: np.ndarray, class_freq: Optional[np.ndarray] = None,
                          prior_power: float = 0.0, eps: float = 1e-6) -> np.ndarray:
    """P(lớp|x) -> log emission = log P(lớp|x) - prior_power * log P(lớp).

    prior_power = 0.0 (MẶC ĐỊNH, khuyến nghị): dùng khi bộ phân lớp được huấn
        luyện với class_weight='balanced' — xác suất của nó đã tương ứng prior
        đều nên KHÔNG chia thêm.
    prior_power = 1.0: Bayes đầy đủ cho bộ phân lớp huấn luyện không trọng số.
        CẨN THẬN: lớp hiếm (outside_post ~1% frame) bị khuếch đại ở MỌI frame,
        có thể nuốt cả clip. Nếu dùng, chọn giá trị 0.3-0.5 và kiểm bằng
        leave-one-clip-out.
    """
    p = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0)
    E = np.log(p)
    if prior_power and class_freq is not None:
        E = E - prior_power * np.log(np.clip(class_freq, eps, 1.0))[None, :]
    return E


# =====================================================================
# Kết quả giải mã
# =====================================================================
@dataclass
class DecodeResult:
    success: bool
    n_frames: int
    segments: List[Tuple[int, int, int]] = field(default_factory=list)  # (state_id, start, end_exclusive)
    log_score: float = NEG
    log_Z: float = NEG
    start_post: Optional[np.ndarray] = None   # (N,   S): P(đoạn trạng thái s BẮT ĐẦU tại frame t)
    end_post: Optional[np.ndarray] = None     # (N+1, S): P(đoạn s KẾT THÚC tại ranh giới t, end exclusive)
    occupancy: Optional[np.ndarray] = None    # (N,   S): P(frame k thuộc trạng thái s)
    _posterior_state: Optional[Tuple[object, tuple]] = field(default=None, repr=False, compare=False)

    def ensure_posteriors(self) -> "DecodeResult":
        """Tính forward-backward đúng một lần, chỉ khi confidence được yêu cầu."""
        if self.start_post is None and self._posterior_state is not None:
            method, args = self._posterior_state
            method(self, *args)
            self._posterior_state = None
        return self

    def state_path(self) -> np.ndarray:
        path = np.full(self.n_frames, -1, dtype=np.int8)
        for s, a, b in self.segments:
            path[a:b] = s
        return path

    def to_rep(self, frame_idx: Optional[Sequence[int]] = None) -> Rep:
        """Đổi đoạn tối ưu thành Rep. frame_idx: ánh xạ vị trí -> frame_idx gốc
        (khi chuỗi pose đã rớt frame); None = vị trí chính là frame_idx."""
        rep = Rep()
        for s, a, b in self.segments:
            if s not in MIDDLE_IDS:
                continue
            fa = int(frame_idx[a]) if frame_idx is not None else a
            fb = int(frame_idx[b - 1]) if frame_idx is not None else b - 1
            rep.phases[CLASSES[s]] = PhaseWindow(fa, fb)
        return rep

    def boundary_confidence(self, tol: int = 3) -> Dict[str, Dict[str, float]]:
        """Xác suất hậu nghiệm để mỗi biên nằm trong ±tol frame quanh vị trí
        đã chọn. Đây là xác suất THÔ của mô hình — phải hiệu chỉnh
        (phase_confidence.py) trước khi hiển thị cho người dùng."""
        out: Dict[str, Dict[str, float]] = {}
        if not self.success:
            return out
        self.ensure_posteriors()
        if self.start_post is None:
            return out
        N = self.n_frames
        chosen = {s: (a, b) for s, a, b in self.segments}
        for s in MIDDLE_IDS:
            name = CLASSES[s]
            absent_prob = float(np.clip(1.0 - self.start_post[:, s].sum(), 0.0, 1.0))
            if s not in chosen:
                out[name] = {"start": 0.0, "end": 0.0, "absent_prob": absent_prob}
                continue
            a, b = chosen[s]
            p_start = float(self.start_post[max(0, a - tol):min(N, a + tol + 1), s].sum())
            p_end = float(self.end_post[max(1, b - tol):min(N, b + tol) + 1, s].sum())
            out[name] = {"start": min(p_start, 1.0), "end": min(p_end, 1.0), "absent_prob": absent_prob}
        return out


# =====================================================================
# Bộ giải mã
# =====================================================================
class HSMMPhaseDecoder:
    def __init__(self, priors: Optional[DurationPriors] = None,
                 skip_penalty: float = 4.0, emission_scale: float = 0.5):
        self.priors = priors or DurationPriors.default()
        self.skip_penalty = float(skip_penalty)
        self.emission_scale = float(emission_scale)

    # ---- chi phí chuyển trạng thái ----
    def _costs(self):
        S, lam = N_CLASSES, self.skip_penalty
        mid = set(MIDDLE_IDS)

        def n_mid_between(p: int, s: int) -> int:
            return sum(1 for m in mid if p < m < s)

        trans = np.full((S, S), NEG)
        for p in range(S):
            for s in range(p + 1, S):
                trans[p, s] = -lam * n_mid_between(p, s)
        start_cost = np.array([-lam * sum(1 for m in mid if m < s) for s in range(S)])
        end_cost = np.array([-lam * sum(1 for m in mid if m > s) for s in range(S)])
        return trans, start_cost, end_cost

    def _durations(self, N: int, fps: float):
        S = N_CLASSES
        dur = np.full((S, N + 1), NEG)
        dmin = np.ones(S, dtype=int)
        dmax = np.full(S, N, dtype=int)
        for s in range(S):
            if s in MIDDLE_IDS:
                lp, lo, hi = self.priors.priors[CLASSES[s]].log_pmf(fps, N)
                dur[s] = lp
                dmin[s], dmax[s] = lo, hi
            else:  # outside: không phạt thời lượng
                dur[s, 1:] = 0.0
        return dur, dmin, dmax

    # ---- giải mã ----
    def decode(self, log_emissions: np.ndarray, fps: float, posteriors: bool = True) -> DecodeResult:
        E = np.asarray(log_emissions, dtype=np.float64)
        if E.ndim != 2 or E.shape[1] != N_CLASSES:
            raise ValueError(f"log_emissions phải có shape (N, {N_CLASSES}), nhận {E.shape}")
        if fps <= 0:
            raise ValueError("fps phải > 0")
        N = E.shape[0]
        if N < 2:
            return DecodeResult(False, N)
        E = np.clip(E, -30.0, 30.0) * self.emission_scale
        S = N_CLASSES

        cs = np.zeros((S, N + 1))
        cs[:, 1:] = np.cumsum(E.T, axis=1)
        trans, start_cost, end_cost = self._costs()
        dur, dmin, dmax = self._durations(N, fps)

        # ---------------- Viterbi ----------------
        alpha = np.full((N + 1, S), NEG)
        inn = np.full((N + 1, S), NEG)
        inn_arg = np.full((N + 1, S), -2, dtype=int)   # -1: bắt đầu clip; >=0: trạng thái trước
        back_d = np.zeros((N + 1, S), dtype=int)
        inn[0, :] = start_cost
        inn_arg[0, :] = -1

        for t in range(1, N + 1):
            for s in range(S):
                hi = min(dmax[s], t)
                if hi < dmin[s]:
                    continue
                ds = np.arange(dmin[s], hi + 1)
                u = t - ds
                vals = inn[u, s] + dur[s, ds] + cs[s, t] - cs[s, u]
                j = int(np.argmax(vals))
                if np.isfinite(vals[j]):
                    alpha[t, s] = vals[j]
                    back_d[t, s] = ds[j]
            if t < N:
                for s in range(1, S):
                    cand = alpha[t, :s] + trans[:s, s]
                    j = int(np.argmax(cand))
                    if np.isfinite(cand[j]):
                        inn[t, s] = cand[j]
                        inn_arg[t, s] = j

        finals = alpha[N, :] + end_cost
        s_end = int(np.argmax(finals))
        if not np.isfinite(finals[s_end]):
            return DecodeResult(False, N)

        segments: List[Tuple[int, int, int]] = []
        t, s = N, s_end
        while t > 0:
            d = int(back_d[t, s])
            a = t - d
            segments.append((s, a, t))
            prev = int(inn_arg[a, s])
            t = a
            if prev < 0:
                break
            s = prev
        segments.reverse()
        result = DecodeResult(True, N, segments, log_score=float(finals[s_end]))
        posterior_args = (cs, trans, start_cost, end_cost, dur, dmin, dmax)
        if posteriors:
            self._posteriors(result, *posterior_args)
        else:
            result._posterior_state = (self._posteriors, posterior_args)
        return result

    # ---- forward-backward (cùng mô hình với Viterbi) ----
    def _posteriors(self, res: DecodeResult, cs, trans, start_cost, end_cost, dur, dmin, dmax) -> None:
        N, S = res.n_frames, N_CLASSES
        alpha = np.full((N + 1, S), NEG)
        inn = np.full((N + 1, S), NEG)
        inn[0, :] = start_cost
        for t in range(1, N + 1):
            for s in range(S):
                hi = min(dmax[s], t)
                if hi < dmin[s]:
                    continue
                ds = np.arange(dmin[s], hi + 1)
                u = t - ds
                alpha[t, s] = _lse(inn[u, s] + dur[s, ds] + cs[s, t] - cs[s, u])
            if t < N:
                for s in range(1, S):
                    inn[t, s] = _lse(alpha[t, :s] + trans[:s, s])
        logZ = _lse(alpha[N, :] + end_cost)
        if not np.isfinite(logZ):
            return

        beta = np.full((N + 1, S), NEG)
        out = np.full((N + 1, S), NEG)
        out[N, :] = end_cost
        for t in range(N - 1, -1, -1):
            for s in range(S):
                hi = min(dmax[s], N - t)
                if hi < dmin[s]:
                    continue
                ds = np.arange(dmin[s], hi + 1)
                v = t + ds
                beta[t, s] = _lse(dur[s, ds] + cs[s, v] - cs[s, t] + out[v, s])
            for s in range(S - 1):
                out[t, s] = _lse(trans[s, s + 1:] + beta[t, s + 1:])

        start_post = np.exp(inn[:N, :] + beta[:N, :] - logZ)
        end_post = np.exp(alpha + out - logZ)
        end_post[0, :] = 0.0
        occ = np.cumsum(start_post, axis=0) - np.cumsum(end_post[:N, :], axis=0)
        res.log_Z = float(logZ)
        res.start_post = np.nan_to_num(start_post)
        res.end_post = np.nan_to_num(end_post)
        res.occupancy = np.clip(occ, 0.0, 1.0)


class PhaseDecoder:
    """Giao diện ổn định cho bộ segmenter: nhận posterior (N,6).

    Viterbi chạy ngay; forward-backward bị trì hoãn đến lúc gọi
    ``boundary_confidence()`` hoặc ``ensure_posteriors()``.
    """

    def __init__(self, priors: Optional[DurationPriors] = None,
                 skip_penalty: float = 4.0, emission_scale: float = 0.5,
                 prior_power: float = 0.0, class_freq: Optional[np.ndarray] = None):
        self.decoder = HSMMPhaseDecoder(priors, skip_penalty, emission_scale)
        self.prior_power = float(prior_power)
        self.class_freq = None if class_freq is None else np.asarray(class_freq, dtype=float)

    def decode(self, posteriors: np.ndarray, fps: float,
               compute_posteriors: bool = False, **kwargs) -> DecodeResult:
        # Tương thích caller cũ dùng posteriors=True/False.
        if "posteriors" in kwargs:
            compute_posteriors = bool(kwargs.pop("posteriors"))
        if kwargs:
            raise TypeError(f"Tham số không hỗ trợ: {sorted(kwargs)}")
        p = np.asarray(posteriors, dtype=float)
        if p.ndim != 2 or p.shape[1] != N_CLASSES:
            raise ValueError(f"posteriors phải có shape (N,{N_CLASSES}), nhận {p.shape}")
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("posteriors phải hữu hạn và không âm")
        row_sum = p.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("mỗi frame cần posterior có tổng dương")
        p = p / row_sum
        emissions = probs_to_log_emissions(
            p, class_freq=self.class_freq, prior_power=self.prior_power
        )
        return self.decoder.decode(emissions, fps=float(fps), posteriors=compute_posteriors)
