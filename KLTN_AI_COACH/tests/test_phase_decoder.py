"""Kiểm thử stage3_features/phase_decoder.py bằng dữ liệu tổng hợp."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core.phase_schema import PhaseSet, Rep
from stage3_features.phase_model.core.phase_labels import rep_to_frame_labels, class_frequencies, CLASS_TO_ID, N_CLASSES
from stage3_features.phase_model.core.phase_decoder import HSMMPhaseDecoder, probs_to_log_emissions
from tests.test_phase_schema import V1_SAMPLE

FPS = 30.0

def synth(labels, strength, rng):
    """Xác suất giả lập của bộ phân lớp: nhiễu độc lập từng frame."""
    logits = rng.normal(0, 1.0, (len(labels), N_CLASSES))
    logits[np.arange(len(labels)), labels] += strength
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)

# Bộ phân lớp giả lập có prior đều (như huấn luyện class_weight=balanced) -> prior_power=0.

def run(rep, total, strength, seed=0, **kw):
    rng = np.random.default_rng(seed)
    lab = rep_to_frame_labels(rep, total).astype(int)
    E = probs_to_log_emissions(synth(lab, strength, rng))
    dec = HSMMPhaseDecoder(**kw)
    t0 = time.time(); res = dec.decode(E, FPS); dt = time.time() - t0
    return res, dt

def bounds(rep):
    return {p: (w.start_frame, w.end_frame) for p, w in rep.phases.items() if not w.absent}

def test_clean_full_four_phases_with_long_outside():
    truth = PhaseSet.from_dict(V1_SAMPLE).reps[0]
    res, dt = run(truth, 743, strength=6.0)
    assert res.success
    got = res.to_rep()
    for p, (a, b) in bounds(truth).items():
        ga, gb = bounds(got)[p]
        assert abs(ga - a) <= 3 and abs(gb - b) <= 3, (p, (a, b), (ga, gb))
    print(f"   giải mã 743 frame trong {dt:.2f}s")

def test_absent_sprint_start_is_detected():
    truth = Rep.from_windows({"vao_cho": (200, 290), "static": (291, 380), "sprint_start": None,
                              "drive_phase": (381, 420)})
    res, _ = run(truth, 500, strength=5.0)
    got = res.to_rep()
    assert got.phases["sprint_start"].absent, "phải nhận ra pha vắng mặt, không bịa"
    assert not got.phases["drive_phase"].absent
    conf = res.boundary_confidence(tol=3)
    assert conf["sprint_start"]["absent_prob"] > 0.5

def test_noisy_emissions_order_and_accuracy():
    truth = PhaseSet.from_dict(V1_SAMPLE).reps[0]
    errs = []
    for seed in range(5):
        res, _ = run(truth, 743, strength=2.5, seed=seed)
        got = res.to_rep()
        order = [w for _, w in got.present()]
        assert all(w1.end_frame < w2.start_frame for w1, w2 in zip(order, order[1:])), "vi phạm thứ tự"
        for p, (a, b) in bounds(truth).items():
            if not got.phases[p].absent:
                errs.append(abs(got.phases[p].start_frame - a))
    assert np.median(errs) <= 4, f"sai số trung vị quá lớn: {np.median(errs)}"

def test_posterior_consistency():
    truth = PhaseSet.from_dict(V1_SAMPLE).reps[0]
    res, _ = run(truth, 743, strength=3.0)
    assert res.log_Z >= res.log_score - 1e-6                         # tổng >= đường tối ưu
    assert np.allclose(res.occupancy.sum(1), 1.0, atol=1e-3)        # mỗi frame đúng 1 trạng thái
    assert np.allclose(res.start_post.sum(0).clip(max=1.0), res.start_post.sum(0), atol=1e-3)
    conf = res.boundary_confidence(tol=3)
    assert all(0.0 <= v <= 1.0 for c in conf.values() for v in c.values())

def test_confidence_lower_when_noisier():
    truth = PhaseSet.from_dict(V1_SAMPLE).reps[0]
    hi, _ = run(truth, 743, strength=6.0); lo, _ = run(truth, 743, strength=1.5)
    mean = lambda r: np.mean([c["start"] for c in r.boundary_confidence(3).values()])
    assert mean(hi) > mean(lo)

def test_too_short_clip_graceful():
    res = HSMMPhaseDecoder().decode(np.zeros((1, N_CLASSES)), FPS)
    assert not res.success

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK", n)
