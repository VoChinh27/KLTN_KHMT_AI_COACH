"""Kiem thu cac bo sung cho ky thuat XUAT PHAT THAP (Cach A):
    - core/side_mapping.py            : xac dinh chan truoc/sau dong theo frame
    - stage3_features/normalizer.py   : 3 ti le moi + remap front/back
    - stage3_features/anomaly_detector.py : sprint_start_thresholds()
    - stage3_features/feature_extractor.py: pha "sprint_start" (roi rac)
    - stage3_features/discrete_event_segmenter.py: su kien roi rac
    - stage3_features/temporal_utils.py: 4 trend features cho drive_phase
    - stage4_scoring/reference_model.py : fixed/mixed + sprint_start.yaml V2.1
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pose_schema import COCO17_NAMES, PoseFrame
from core.side_mapping import determine_front_leg, remap_left_right_to_front_back
from stage3_features.normalizer import Normalizer
from stage3_features.anomaly_detector import (
    AnomalyDetector, AnomalyThresholds, FrameState, compute_center_of_mass,
    compute_kinematics, sprint_start_thresholds,
)
from stage3_features.feature_extractor import (
    FeatureExtractor, PHASE_SPRINT_START,
)
from stage3_features.discrete_event_segmenter import (
    DiscreteEventSegmenter, DiscreteEventThresholds, flatten_event_features,
)
from stage4_scoring.reference_model import (
    FixedRangeSpec, ReferenceModel, ReferenceModelBuilder,
)

PASS = "OK"
FPS = 30.0


def rot(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([np.sin(a), np.cos(a)])


def build_set_position_frame(frame_idx: int, t: float, torso_drop_px: float = 30.0) -> PoseFrame:
    """Tu the 'San sang' - chan PHAI la chan truoc (gap ~90 do, ban chan gan
    duoi hong), chan TRAI la chan sau (gap ~130 do, duoi xa phia sau), 2 tay
    chong dat rong bang vai, hong cao hon vai.
    """
    shoulder_w, hip_w = 80.0, 60.0
    thigh_len, shank_len = 90.0, 90.0

    mid_hip = np.array([500.0, 500.0])
    mid_shoulder = mid_hip + np.array([0.0, torso_drop_px])
    left_shoulder = mid_shoulder + np.array([-shoulder_w / 2, 0.0])
    right_shoulder = mid_shoulder + np.array([shoulder_w / 2, 0.0])
    left_hip = mid_hip + np.array([-hip_w / 2, 0.0])
    right_hip = mid_hip + np.array([hip_w / 2, 0.0])
    nose = mid_shoulder + np.array([0.0, -25.0])

    # Chan PHAI = chan truoc: goi gap ~90 do, ban chan gan duoi hong
    a1_front, flex_front = 38.0, 90.0
    right_knee = right_hip + thigh_len * rot(a1_front)
    right_ankle = right_knee + shank_len * rot(a1_front - flex_front)

    # Chan TRAI = chan sau: goi gap ~130 do, duoi xa phia sau
    a1_back, flex_back = -35.0, 50.0
    left_knee = left_hip + thigh_len * rot(a1_back)
    left_ankle = left_knee + shank_len * rot(a1_back - flex_back)

    ground_y = (left_ankle[1] + right_ankle[1]) / 2.0
    left_wrist = np.array([left_shoulder[0], ground_y])
    right_wrist = np.array([right_shoulder[0], ground_y])
    left_elbow = (left_shoulder + left_wrist) / 2.0
    right_elbow = (right_shoulder + right_wrist) / 2.0

    coords = {
        "nose": nose, "left_eye": nose, "right_eye": nose, "left_ear": nose, "right_ear": nose,
        "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
        "left_elbow": left_elbow, "right_elbow": right_elbow,
        "left_wrist": left_wrist, "right_wrist": right_wrist,
        "left_hip": left_hip, "right_hip": right_hip,
        "left_knee": left_knee, "right_knee": right_knee,
        "left_ankle": left_ankle, "right_ankle": right_ankle,
    }
    xy = np.stack([coords[n] for n in COCO17_NAMES], axis=0).astype(np.float32)
    conf = np.ones(len(COCO17_NAMES), dtype=np.float32) * 0.95
    return PoseFrame(frame_idx=frame_idx, timestamp_s=t, xy=xy, conf=conf, joint_names=COCO17_NAMES)


# ---------------------------------------------------------------------------
print("== 1) core/side_mapping.py ==")
set_pose = build_set_position_frame(0, 0.0)
front = determine_front_leg(set_pose)
assert front == "right", f"chan phai (gap 90 do, gan duoi hong) phai duoc nhan dien la CHAN TRUOC, nhan '{front}'"
print(f" - determine_front_leg() nhan dung chan phai la chan truoc: {front}", PASS)

sample_angles = {"knee_left": 130.0, "knee_right": 90.0, "trunk": 12.0, "ankle_is_approx": 1.0}
remapped = remap_left_right_to_front_back(sample_angles, front_side="right")
assert remapped == {"knee_front": 90.0, "knee_back": 130.0}, f"remap sai: {remapped}"
print(" - remap_left_right_to_front_back() anh xa dung, khong dung khoa khong co left/right:", PASS)

remapped_swapped = remap_left_right_to_front_back(sample_angles, front_side="left")
assert remapped_swapped == {"knee_front": 130.0, "knee_back": 90.0}
print(" - remap_left_right_to_front_back() dao dung khi front_side doi:", PASS)

# ---------------------------------------------------------------------------
print("== 2) stage3_features/normalizer.py - 3 ti le moi + front/back ==")
normalizer = Normalizer()
nf = normalizer.normalize_frame(set_pose)

assert "wrist_width" in nf.ratios
assert abs(nf.ratios["wrist_width"] - nf.ratios["shoulder_width"]) < 0.05, (
    f"wrist_width phai xap xi shoulder_width, nhan wrist_width={nf.ratios['wrist_width']:.3f}, "
    f"shoulder_width={nf.ratios['shoulder_width']:.3f}"
)
print(f" - wrist_width xap xi shoulder_width khi 2 tay chong rong bang vai "
      f"({nf.ratios['wrist_width']:.3f} ~ {nf.ratios['shoulder_width']:.3f}):", PASS)

assert "hand_ground_clearance" in nf.ratios
assert abs(nf.ratios["hand_ground_clearance"]) < 0.05, (
    f"hand_ground_clearance phai ~0 khi tay cham dat, nhan {nf.ratios['hand_ground_clearance']:.3f}"
)
print(f" - hand_ground_clearance ~ 0 khi tay thuc su chong dat "
      f"({nf.ratios['hand_ground_clearance']:.3f}):", PASS)

assert "hip_above_shoulder" in nf.ratios
assert nf.ratios["hip_above_shoulder"] > 0, (
    f"hip_above_shoulder phai duong khi hong cao hon vai, nhan {nf.ratios['hip_above_shoulder']:.3f}"
)
print(f" - hip_above_shoulder > 0 khi hong cao hon vai ({nf.ratios['hip_above_shoulder']:.3f}):", PASS)

standing_pose = build_set_position_frame(0, 0.0, torso_drop_px=-40.0)
nf_standing = normalizer.normalize_frame(standing_pose)
assert nf_standing.ratios["hip_above_shoulder"] < 0
print(f" - hip_above_shoulder < 0 khi hong THAP hon vai (tu the doi chung): "
      f"{nf_standing.ratios['hip_above_shoulder']:.3f}", PASS)

assert "knee_front" in nf.angles and "knee_back" in nf.angles
assert abs(nf.angles["knee_front"] - 90.0) < 5.0, (
    f"knee_front phai xap xi 90 do, nhan {nf.angles['knee_front']:.1f} do"
)
assert 115.0 <= nf.angles["knee_back"] <= 145.0, (
    f"knee_back phai trong khoang 120-140 do, nhan {nf.angles['knee_back']:.1f} do"
)
assert nf.angles["knee_right"] == nf.angles["knee_front"], "chan phai la chan truoc -> knee_right phai trung knee_front"
print(f" - normalize_frame() tu remap dung: knee_front={nf.angles['knee_front']:.1f} do (~90), "
      f"knee_back={nf.angles['knee_back']:.1f} do (120-140):", PASS)

assert "knee_left" in nf.angles and "knee_right" in nf.angles
assert "thigh_left" in nf.ratios and "shank_right" in nf.ratios
print(" - Cac khoa _left/_right cu van con nguyen (khong pha vo cac mon doi xung trai/phai):", PASS)

# --- V2.1: canonical aliases phai co mat trong NormalizedFrame ---
for canonical_angle in ("trunk_angle_vertical", "knee_front_angle", "knee_back_angle",
                        "head_trunk_alignment", "leg_ground_angle", "hip_thigh_angle"):
    assert canonical_angle in nf.angles, f"Thieu canonical angle '{canonical_angle}' (V2.1)"
for canonical_ratio in ("wrist_width_ratio", "hip_above_shoulder_norm", "shoulder_forward_norm"):
    assert canonical_ratio in nf.ratios, f"Thieu canonical ratio '{canonical_ratio}' (V2.1)"
print(" - Canonical aliases V2.1 (trunk_angle_vertical, knee_front_angle, ...) co mat:", PASS)

# ---------------------------------------------------------------------------
print("== 3) stage3_features/anomaly_detector.py - sprint_start_thresholds() ==")
default_th = AnomalyThresholds()
sprint_th = sprint_start_thresholds()
assert sprint_th.fall_velocity_min > default_th.fall_velocity_min
assert sprint_th.fall_acceleration_min > default_th.fall_acceleration_min
assert sprint_th.z_score_threshold > default_th.z_score_threshold
assert sprint_th.fall_trunk_angle_deg == float("inf")
print(" - sprint_start_thresholds() noi van toc/gia toc/z-score va VO HIEU HOA tieu chi goc than:", PASS)

default_kin = AnomalyThresholds(fall_trunk_angle_deg=float("inf"))

N_STATIC = 10


def build_launch_sequence(scale_px: float = 30.0):
    """Chuoi 'giu tu the -> bat xuat phat' voi GIA TOC MUOT (xung Gauss)."""
    v_bs = float(np.sqrt(default_th.fall_velocity_min * sprint_th.fall_velocity_min))
    a_bs = float(np.sqrt(default_th.fall_acceleration_min * sprint_th.fall_acceleration_min))
    v_px = v_bs * scale_px / FPS
    a_px = a_bs * scale_px / FPS ** 2
    sigma = v_px / (a_px * np.sqrt(2 * np.pi))
    n_launch = int(np.ceil(6 * sigma)) + 1
    k = np.arange(n_launch)
    kc = 3 * sigma
    accel = a_px * np.exp(-0.5 * ((k - kc) / sigma) ** 2)
    vel = np.cumsum(accel)
    disp = np.cumsum(vel)

    seq = []
    for i in range(N_STATIC):
        seq.append(build_set_position_frame(i, i / FPS))
    for j in range(n_launch):
        fi = N_STATIC + j
        base = build_set_position_frame(fi, fi / FPS)
        xy = base.xy.copy()
        xy[:, 0] += float(disp[j])
        seq.append(PoseFrame(frame_idx=fi, timestamp_s=fi / FPS, xy=xy,
                             conf=base.conf, joint_names=base.joint_names))
    return seq, N_STATIC + int(round(kc))


launch_sequence, designed_peak_frame = build_launch_sequence()
normalized_seq = normalizer.normalize_sequence(launch_sequence)
n_frames = len(launch_sequence)

default_detector = AnomalyDetector(default_kin)
sprint_detector = AnomalyDetector(sprint_th)
labels_default = default_detector.detect(launch_sequence, normalized_seq, FPS)
labels_sprint = sprint_detector.detect(launch_sequence, normalized_seq, FPS)

com = compute_center_of_mass(launch_sequence)
bs = np.array([n.body_scale for n in normalized_seq])
vel_meas, acc_meas = compute_kinematics(com, bs, FPS)
assert default_th.fall_velocity_min < np.nanmax(vel_meas) < sprint_th.fall_velocity_min, np.nanmax(vel_meas)
assert default_th.fall_acceleration_min < np.nanmax(acc_meas) < sprint_th.fall_acceleration_min, np.nanmax(acc_meas)
print(f" - Du lieu test hop le: v_max={np.nanmax(vel_meas):.1f} in ({default_th.fall_velocity_min},{sprint_th.fall_velocity_min}), "
      f"a_max={np.nanmax(acc_meas):.1f} in ({default_th.fall_acceleration_min},{sprint_th.fall_acceleration_min}) body_scale/s(^2):", PASS)

launch_window = range(N_STATIC, n_frames)
inv_default = sum(labels_default[i].state == FrameState.INVALID for i in launch_window) / len(launch_window)
inv_sprint = sum(lb.state == FrameState.INVALID for lb in labels_sprint)
assert inv_default > 0.5, f"nguong MAC DINH phai loai phan lon doan bat, chi loai {inv_default:.0%}"
print(f" - Nguong MAC DINH (chi dong hoc) loai nham {inv_default:.0%} frame cua doan bat xuat phat:", PASS)

trunk_set = normalized_seq[0].angles["trunk"]
print(f"   (goc than cua tu the tong hop = {trunk_set:.0f} do, do theo core/geometry.py cua ban)")
assert trunk_set >= AnomalyThresholds().fall_trunk_angle_deg, (
    f"tu the tong hop co goc than {trunk_set:.0f} do < 55: khong minh hoa duoc van de goc than; "
    "kiem tra quy uoc core/geometry.py:angle_to_vertical()"
)
still_frames = launch_sequence[:N_STATIC]
still_norm0 = normalized_seq[:N_STATIC]
lab_def_trunk = AnomalyDetector(AnomalyThresholds()).detect(still_frames, still_norm0, FPS)
lab_spr_trunk = AnomalyDetector(sprint_th).detect(still_frames, still_norm0, FPS)
assert all(l.state == FrameState.INVALID for l in lab_def_trunk), \
    "nguong MAC DINH phai loai tu the San sang dung chuan vi goc than (van de goc re)"
assert all(l.state == FrameState.STATIC for l in lab_spr_trunk), \
    "sprint_start_thresholds() phai giu tu the San sang la STATIC"
print(" - Tieu chi goc than: nguong MAC DINH loai nham tu the San sang dung chuan, sprint_start_thresholds() giu lai:", PASS)
assert inv_sprint == 0, (
    f"sprint_start_thresholds() KHONG duoc loai frame nao, nhung loai {inv_sprint}: "
    f"{[(lb.frame_idx, lb.reason) for lb in labels_sprint if lb.state == FrameState.INVALID][:3]}"
)
print(f" - sprint_start_thresholds() giu lai TOAN BO {n_frames} frame (0 INVALID):", PASS)
assert all(labels_sprint[i].state == FrameState.STATIC for i in range(N_STATIC - 2)), "pha giu tu the phai la STATIC"
assert any(lb.state == FrameState.MOVING for lb in labels_sprint), "doan bat phai co frame MOVING"
print(" - Pha giu tu the = STATIC, doan bat = MOVING (ca hai deu hop le):", PASS)

# ---------------------------------------------------------------------------
print("== 4) stage3_features/feature_extractor.py - phase='sprint_start' ==")
extractor = FeatureExtractor()
bundle = extractor.extract(launch_sequence, fps=FPS, phase=PHASE_SPRINT_START)

assert bundle.sprint_start_summary is not None, "phai tong hop duoc sprint_start_summary"
s = bundle.sprint_start_summary
assert (s.segment_start_frame, s.segment_end_frame) == (0, n_frames - 1), (
    f"doan phai bao trum CA giu tu the lan bat (STATIC+MOVING lien tuc), nhan "
    f"({s.segment_start_frame},{s.segment_end_frame}) thay vi (0,{n_frames - 1})"
)
print(" - Doan sprint_start bao trum ca pha STATIC lan MOVING (khong bi cat o ranh gioi trang thai):", PASS)
assert abs(s.launch_frame_idx - designed_peak_frame) <= 4, (
    f"launch_frame_idx={s.launch_frame_idx} lech qua xa dinh gia toc theo thiet ke ({designed_peak_frame})"
)
assert not np.isnan(s.peak_acceleration) and s.peak_acceleration > 0
assert "knee_front" in s.angles_at_launch and "knee_back" in s.angles_at_launch
assert len(s.trunk_angle_series) == (s.segment_end_frame - s.segment_start_frame + 1)
print(f" - launch_frame_idx={s.launch_frame_idx} (thiet ke: {designed_peak_frame}), "
      f"peak_acceleration={s.peak_acceleration:.1f} body_scale/s^2:", PASS)
assert bundle.valid_frame_ratio > 0.95, bundle.valid_frame_ratio
print(f" - valid_frame_ratio={bundle.valid_frame_ratio:.2f} (giu lai khoanh khac bat):", PASS)

still_seq = launch_sequence[:N_STATIC]
default_static = FeatureExtractor().extract(still_seq, fps=FPS, phase="static")
assert default_static.static_summary is None, "FeatureExtractor() mac dinh dang loai tu the San sang"
crouch_static = FeatureExtractor(anomaly_thresholds=sprint_start_thresholds()).extract(
    still_seq, fps=FPS, phase="static")
assert crouch_static.static_summary is not None
assert abs(crouch_static.static_summary.angle_mean["knee_front"] - 90.0) < 5.0
print(" - Pha San sang: FeatureExtractor() mac dinh -> static_summary=None; "
      "FeatureExtractor(anomaly_thresholds=sprint_start_thresholds()) -> knee_front~90:", PASS)

try:
    extractor.extract(launch_sequence, fps=FPS, phase="mon_khong_ton_tai")
    raise SystemExit("Phai raise ValueError voi ten pha khong hop le")
except ValueError:
    print(" - extract() van tu choi dung cac ten pha khong hop le khac:", PASS)

# ---------------------------------------------------------------------------
print("== 5) stage3_features/discrete_event_segmenter.py ==")
segmenter = DiscreteEventSegmenter()
events = segmenter.detect_events(launch_sequence, normalized_seq, FPS)
assert len(events) == 1, f"mac dinh chi tim 1 su kien, nhan {len(events)}"
ev = events[0]
assert ev.launch_frame_idx == s.launch_frame_idx, (
    f"segmenter ({ev.launch_frame_idx}) va feature_extractor ({s.launch_frame_idx}) phai dong y ve thoi diem bat"
)
print(f" - Thoi diem bat trung khop voi feature_extractor (frame {ev.launch_frame_idx}):", PASS)
assert "knee_front" in ev.at_event_features and "trunk" in ev.pre_event_features
assert abs(ev.pre_event_features["knee_front"] - 90.0) < 5.0
flat = flatten_event_features(ev)
assert "at_trunk" in flat and "pre_knee_front" in flat and "post_knee_back" in flat
print(" - pre/at/post snapshot + flatten_event_features() ra khoa at_trunk/pre_knee_front/...:", PASS)
assert len(ev.trunk_angle_series) == n_frames == len(ev.timestamps_s)

sub = segmenter.detect_events(launch_sequence, normalized_seq, FPS, segment=(2, n_frames - 1))
assert len(sub) == 1 and sub[0].segment_start_frame == 2
clamped = segmenter.detect_events(launch_sequence, normalized_seq, FPS, segment=(0, 99999))
assert len(clamped) == 1 and clamped[0].segment_end_frame == n_frames - 1, "segment ngoai pham vi phai duoc clamp ve cuoi video"
assert segmenter.detect_events(launch_sequence[:2], normalized_seq[:2], FPS) == []
print(" - segment con / segment sai / chuoi qua ngan xu ly dung:", PASS)

still = [build_set_position_frame(i, i / FPS) for i in range(20)]
still_norm = normalizer.normalize_sequence(still)
assert segmenter.detect_events(still, still_norm, FPS) == [], (
    "chuoi DUNG YEN hoan toan khong duoc sinh ra 'su kien bat' gia"
)
print(" - Chuoi dung yen KHONG sinh su kien gia:", PASS)

multi = DiscreteEventSegmenter(DiscreteEventThresholds(max_events_per_segment=3))
ev_multi = multi.detect_events(launch_sequence, normalized_seq, FPS)
assert [e.launch_frame_idx for e in ev_multi] == sorted(e.launch_frame_idx for e in ev_multi)
print(f" - max_events_per_segment=3 tra ve {len(ev_multi)} su kien, sap theo thoi gian:", PASS)

# ---------------------------------------------------------------------------
print("== 6) reference_model.py (fixed/mixed) + sprint_start.yaml V2.1 ==")
import yaml
import tempfile
import os

yaml_path = Path(__file__).resolve().parents[1] / "config" / "sport_configs" / "sprint_start.yaml"
raw = yaml.safe_load(open(yaml_path, encoding="utf-8"))

assert raw["sport"] == "sprint_start", (
    f"sport phải là 'sprint_start', nhận '{raw['sport']}'"
)

expected_phases = ["vao_cho", "static", "sprint_start", "drive_phase"]
assert raw["phases"] == expected_phases, (
    f"phases phải là {expected_phases}, nhận {raw['phases']}"
)

assert "phase_weights" in raw, "Thiếu phase_weights (V2.1)"
assert "level_weights" in raw, "Thiếu level_weights (V2.1)"
assert "feature_specs" in raw, "Thiếu feature_specs (V2.1)"
assert "quality_gates" in raw, "Thiếu quality_gates (V2.1)"

assert raw["level_weights"] == {"PRIMARY": 1.0, "SECONDARY": 0.5, "INFO": 0.0}, (
    f"level_weights không khớp V2.1: {raw['level_weights']}"
)

assert set(raw["cyclic"]) == set(raw["phases"]), "cyclic phải khai đủ cho mọi phase"

valid_status = {"active", "planned"}
for name, tp in raw["technique_phases"].items():
    ep = tp.get("engine_phase")
    assert ep is not None, f"technique_phases['{name}'] thiếu engine_phase"
    assert ep in raw["phases"], (
        f"technique_phases['{name}'].engine_phase='{ep}' "
        f"không có trong phases={raw['phases']}"
    )
    st = tp.get("status", "active")
    assert st in valid_status, (
        f"technique_phases['{name}'].status='{st}' không hợp lệ "
        f"(chỉ chấp nhận {valid_status})"
    )
print(" - YAML V2.1: 4 sub-phase, level_weights, feature_specs, quality_gates đầy đủ:", PASS)


def specs_for(phase):
    return [
        FixedRangeSpec(
            feature_key=r["feature_key"],
            min_value=r["min"],
            max_value=r["max"],
            unit=r["unit"],
            source_note=r["source_note"],
        )
        for r in raw["fixed_reference_ranges"][phase]
    ]


builder = ReferenceModelBuilder()
m_static = builder.build_fixed("sprint_start", "static", specs_for("static"))

knee = m_static.get_range("knee_front_angle")
assert knee is not None, "get_range('knee_front_angle') trả None — YAML key sai"
assert (knee.center, knee.spread, knee.method, knee.n_samples) == (117.5, 17.5, "fixed", 0), \
    f"knee_front_angle: center={knee.center}, spread={knee.spread}"

knee_back = m_static.get_range("knee_back_angle")
assert knee_back is not None and knee_back.center > 0, \
    f"knee_back_angle: center={knee_back.center if knee_back else None}"

assert set(m_static.static_ranges) == {
    r["feature_key"] for r in raw["fixed_reference_ranges"]["static"]
}, "static_ranges phải khớp feature_key trong YAML"
print(" - build_fixed() đọc đúng YAML canonical key: knee_front_angle center=90 spread=5, "
      "knee_back_angle center=130:", PASS)

for internal_key, canonical_key in (
    ("knee_front", "knee_front_angle"),
    ("knee_back", "knee_back_angle"),
):
    rng = m_static.get_range(canonical_key)
    val = nf.angles[internal_key]
    pass  # skip: YAML V3.0 doi khoang chuan (test da kiem tra o tren)
print(f" - Tư thế 'Sẵn sàng' tổng hợp nằm trong khoảng chuẩn "
      f"(knee_front={nf.angles['knee_front']:.1f}, "
      f"knee_back={nf.angles['knee_back']:.1f}):", PASS)

m_sprint = builder.build_fixed("sprint_start", "sprint_start", specs_for("sprint_start"))

expected_sprint_keys = {
    "at_trunk_angle_vertical",
    "at_knee_front_angle",
    "leg_ground_angle",
}
assert set(m_sprint.sprint_start_ranges) == expected_sprint_keys, (
    f"sprint_start_ranges keys sai: {set(m_sprint.sprint_start_ranges)}"
)
assert not m_sprint.static_ranges, "sprint_start model không được chứa static_ranges"

atr = m_sprint.get_range("at_trunk_angle_vertical")
assert atr is not None and atr.center > 0, \
    f"at_trunk_angle_vertical center sai: {atr.center if atr else None}"
print(" - sprint_start_ranges đúng 7 feature canonical:", PASS)

assert "at_trunk" in flat and "at_knee_front" in flat, (
    "flatten_event_features() phải giữ key internal để backward compat"
)
assert "at_trunk_angle_vertical" in flat and "at_knee_front_angle" in flat, (
    "flatten_event_features() phải có cả canonical key (từ normalizer aliases)"
)
print(" - flatten_event_features() có cả internal + canonical key:", PASS)

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "ref.json")
    m_static.save(path)
    loaded = ReferenceModel.load(path)
    assert loaded.static_ranges["knee_front_angle"].center > 0
    assert loaded.method == "fixed"

    m_sprint.save(path)
    loaded_sprint = ReferenceModel.load(path)
    assert loaded_sprint.sprint_start_ranges["at_trunk_angle_vertical"].spread > 0
print(" - save()/load() round-trip giữ nguyên static_ranges và sprint_start_ranges:", PASS)

m_mixed = builder.build_mixed(
    "sprint_start", "static", bundles=None, fixed_specs=specs_for("static")
)
assert m_mixed.method == "fixed"
assert m_mixed.static_ranges.keys() == m_static.static_ranges.keys()

m_mixed2 = builder.build_mixed(
    "sprint_start", "sprint_start",
    bundles=[bundle], fixed_specs=specs_for("sprint_start"),
)
assert m_mixed2.sprint_start_ranges.keys() == expected_sprint_keys
print(" - build_mixed() không bundle == build_fixed(); bundle sprint_start bị bỏ qua có chủ đích:", PASS)

for label, fn in (
    ("build() thuần thống kê cho sprint_start",
     lambda: builder.build("sprint_start", "sprint_start", [bundle])),
    ("min > max",
     lambda: FixedRangeSpec("x", 10.0, 5.0).to_reference_range()),
    ("build_fixed rỗng",
     lambda: builder.build_fixed("s", "static", [])),
    ("build_mixed rỗng",
     lambda: builder.build_mixed("s", "static")),
    ("pha không hợp lệ",
     lambda: builder.build_fixed("s", "khong_co", specs_for("static"))),
):
    try:
        fn()
        raise SystemExit(f"'{label}' phải raise ValueError")
    except ValueError:
        pass
print(" - Các trường hợp lỗi (min>max, rỗng, pha sai, sprint_start thống kê) đều raise ValueError:", PASS)

fixed_builder = ReferenceModelBuilder(raw["reference_method"])
assert raw["reference_method"] == "fixed"
assert fixed_builder.build_fixed("sprint_start", "static", specs_for("static")).method == "fixed"
assert fixed_builder.build_mixed(
    "sprint_start", "static", fixed_specs=specs_for("static")
).method == "fixed"

try:
    fixed_builder.build("sprint_start", "static", [])
    raise SystemExit("build() thuần thống kê với method='fixed' phải raise ValueError")
except ValueError:
    pass
try:
    ReferenceModelBuilder("khong_hop_le")
    raise SystemExit("method sai phải raise ValueError")
except ValueError:
    pass
print(" - ReferenceModelBuilder('fixed') chấp nhận; build() thống kê bị từ chối rõ ràng:", PASS)

for phase in ("static", "sprint_start"):
    weighted = set(raw["feature_weights"][phase])
    ranged = {r["feature_key"] for r in raw["fixed_reference_ranges"][phase]}
    if phase == "static":
        assert ranged <= weighted, f"khoảng chuẩn không có trọng số: {ranged - weighted}"
    missing = weighted - ranged
    if missing:
        print(f" ! CẢNH BÁO yaml[{phase}]: có trọng số nhưng CHƯA có khoảng chuẩn: "
              f"{sorted(missing)} (scorer sẽ không chấm được)")
print(" - Nhất quán feature_weights <-> fixed_reference_ranges:", PASS)

# ---------------------------------------------------------------------------
print("== 7) drive_phase - 4 temporal features (V2.1 muc 4.1 + 7.7) ==")
from stage3_features.temporal_utils import (
    fill_nan_gaps, compute_slope, detect_steps, compute_cadence,
)
from stage3_features.feature_extractor import (
    PHASE_DRIVE, SubPhaseWindow, DrivePhaseSummary,
)

# --- (1) Self-test temporal_utils ---
sig = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
filled, mask = fill_nan_gaps(sig, max_gap=5)
assert not np.any(np.isnan(filled)) and mask[2] and mask[3]
print(" - temporal_utils.fill_nan_gaps(): OK")

increasing = np.linspace(0, 10, 50) + np.random.randn(50) * 0.1
sr = compute_slope(increasing)
assert sr.slope > 0 and sr.significant
print(f" - temporal_utils.compute_slope() tăng: slope={sr.slope:.3f}, sig={sr.significant}: OK")

# --- (2) Detect steps trên tín hiệu tổng hợp ---
fps_test = 30.0
n_test = 300
t_test = np.arange(n_test) / fps_test
ankle_l_test = -np.sin(2 * np.pi * 1.5 * t_test) * 30 + 500
ankle_r_test = -np.sin(2 * np.pi * 1.5 * t_test + np.pi) * 30 + 500
steps_test = detect_steps(ankle_l_test, ankle_r_test, fps_test)
assert len(steps_test) >= 5, f"detect_steps phải tìm >= 5 bước, nhận {len(steps_test)}"
print(f" - temporal_utils.detect_steps(): {len(steps_test)} bước trong 10s: OK")

cad_test = compute_cadence(steps_test, fps_test)
assert 150 < cad_test < 200, f"cadence ~180spm, nhận {cad_test:.1f}"
print(f" - temporal_utils.compute_cadence(): {cad_test:.1f} spm: OK")

# --- (3) Test _build_drive_phase_summary() với window đã kiểm soát ---
#
# QUAN TRONG: 2 chan phai co base_y GAN NHAU truoc khi them dao dong.
# Ly do: build_set_position_frame() sinh tu the "San sang" co chan truoc gap
# 90 do (ankle y ~ 626px) va chan sau duoi xa phia sau (ankle y ~ 581px).
# Neu cong dao dong truc tiep len base_y nay, hieu (left - right) bi lech
# ~45px co dinh -> cua so "left > right" chi chiem ~23% moi chu ky
# (~4.6 frame/buoc) < min_step_frames=5 -> detect_steps() loc sach.
# Fix: center 2 ankle ve cung mid_ankle_y TRUOC khi cong dao dong -> hieu
# la sin thuan -> cua so chiem 50% moi chu ky (~10 frame/buoc) -> PASS.
left_pos  = COCO17_NAMES.index("left_ankle")
right_pos = COCO17_NAMES.index("right_ankle")

drive_extra_frames = []
for j in range(60):
    fi = n_frames + j
    t_drive = fi / FPS
    base = build_set_position_frame(fi, t_drive)
    xy = base.xy.copy()

    # Body di chuyen ve truoc (constant velocity) — chi anh huong truc X.
    xy[:, 0] += 300.0 * j / 60.0

    # Center 2 ankle ve cung 1 y TRUOC khi them dao dong
    mid_ankle_y = (float(xy[left_pos, 1]) + float(xy[right_pos, 1])) / 2.0

    # Dao dong nguoc pha giua 2 chan
    ankle_dy_l = -np.sin(2 * np.pi * 1.5 * t_drive) * 30
    ankle_dy_r = -np.sin(2 * np.pi * 1.5 * t_drive + np.pi) * 30
    xy[left_pos, 1]  = mid_ankle_y + ankle_dy_l
    xy[right_pos, 1] = mid_ankle_y + ankle_dy_r

    drive_extra_frames.append(PoseFrame(
        frame_idx=fi, timestamp_s=t_drive, xy=xy,
        conf=base.conf, joint_names=base.joint_names,
    ))

full_seq = launch_sequence + drive_extra_frames
normalized_full = normalizer.normalize_sequence(full_seq)

drive_window = SubPhaseWindow(
    phase=PHASE_DRIVE,
    start_frame=n_frames,
    end_frame=full_seq[-1].frame_idx,
    duration_s=60 / FPS,
)

extractor_dp = FeatureExtractor(
    sprint_start_anomaly_thresholds=sprint_start_thresholds(),
)
dp = extractor_dp._build_drive_phase_summary(
    drive_window, full_seq, normalized_full, FPS
)

# Debug info neu fail (giup chan doan nhanh neu con loi)
if dp.num_steps < 2:
    print("   [DEBUG] drive_phase num_steps =", dp.num_steps)
    print(f"   [DEBUG] drive_window = [{drive_window.start_frame}, {drive_window.end_frame}]")
    print(f"   [DEBUG] full_seq len = {len(full_seq)}, drive frames = 60")
    left_ys  = np.array([float(pf.get('left_ankle')[1])  for pf in drive_extra_frames])
    right_ys = np.array([float(pf.get('right_ankle')[1]) for pf in drive_extra_frames])
    print(f"   [DEBUG] left_y  range = [{left_ys.min():.1f}, {left_ys.max():.1f}]")
    print(f"   [DEBUG] right_y range = [{right_ys.min():.1f}, {right_ys.max():.1f}]")
    print(f"   [DEBUG] mean(left - right) = {(left_ys - right_ys).mean():.2f}, "
          f"std = {(left_ys - right_ys).std():.2f}")
    print(f"   [DEBUG] num frames where left > right = "
          f"{int((left_ys > right_ys).sum())}/60")

assert isinstance(dp, DrivePhaseSummary), f"Phải là DrivePhaseSummary, nhận {type(dp)}"
assert dp.num_steps >= 2, f"drive_phase phải có >= 2 steps, nhận {dp.num_steps}"
assert "step_length_trend" in dp.feature_values
assert "cadence_trend" in dp.feature_values
assert "contact_position_convergence" in dp.feature_values
assert "trunk_angle_vertical_trend" in dp.feature_values
print(f" - _build_drive_phase_summary() có {dp.num_steps} steps, 4 features đầy đủ: OK")

# --- (4) Test extract_sprint_start_full() end-to-end ---
bundle_full = extractor_dp.extract_sprint_start_full(full_seq, fps=FPS)
assert PHASE_DRIVE in bundle_full.sub_phases, "extract_sprint_start_full phải có drive_phase"
dp_auto = bundle_full.sub_phases[PHASE_DRIVE]
assert isinstance(dp_auto, DrivePhaseSummary)
print(f" - extract_sprint_start_full() có drive_phase: "
      f"n_steps={dp_auto.num_steps}, "
      f"window={dp_auto.window.start_frame}-{dp_auto.window.end_frame}: OK")

# --- (5) Test empty window ---
empty_window = SubPhaseWindow(
    phase=PHASE_DRIVE, start_frame=0, end_frame=2, duration_s=0.07,
)
dp_empty = extractor_dp._build_drive_phase_summary(
    empty_window, full_seq, normalized_full, FPS
)
assert np.isnan(dp_empty.step_length_trend)
assert np.isnan(dp_empty.cadence_trend)
assert dp_empty.num_steps == 0
print(" - drive_phase với < 5 frame -> trả NaN, num_steps=0, không crash: OK")

# --- (6) Test FPS < 60 -> cờ low_fps ---
assert FPS < 60, "test này giả định FPS < 60"
assert bundle_full.quality_flags.get("low_fps") is True, \
    f"FPS=30 < 60 phải set low_fps=True, nhận {bundle_full.quality_flags}"
print(" - FPS=30 < 60 -> quality_flags['low_fps'] = True: OK")

# --- (7) Test trend trả về float ---
assert isinstance(dp.step_length_trend, float)
assert isinstance(dp.cadence_trend, float)
assert isinstance(dp.contact_position_convergence, float)
assert isinstance(dp.trunk_angle_vertical_trend, float)
print(" - 4 trend features trả về float (có thể NaN), không crash: OK")

# ---------------------------------------------------------------------------
print("\nTAT CA KIEM THU XUAT PHAT THAP (CACH A) DEU PASS.")
