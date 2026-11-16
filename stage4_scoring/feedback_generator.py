"""
stage4_scoring/feedback_generator.py
=======================================
Mục đích : Sinh phản hồi huấn luyện có CĂN CỨ SỐ LIỆU từ ScoreReport.

V2.1 - Bổ sung:
    - Bỏ hoàn toàn message cũ "~45 độ" (không còn dùng).
    - Thêm display name cho tất cả feature spec v2.1.
    - Phân loại 3 loại lỗi: TECHNIQUE_ERROR / MEASUREMENT_ERROR /
      INSUFFICIENT_DATA.
    - Hiển thị quality flags từ FeatureBundle.

V2.1.2 - Cải thiện phân loại lỗi:
    - Rule mới: nếu video có quality flag tổng (low_resolution,
      insufficient_data, ...) → mọi lỗi đều xếp MEASUREMENT_ERROR.
    - Thêm display name + advice cho drive_phase (4 trend features).

V2.2 - Sửa bug thiết kế feedback (FB1–FB5):
    - low_fps KHÔNG còn là video-level unreliable flag (FB1).
      Chỉ ảnh hưởng feature slope (drive_phase) ở feature-level.
      → Clip 30fps vẫn có thể nhận "Lỗi kỹ thuật" khi deviation lớn.
    - wrist_low_confidence mở rộng cover elbow_*/hand_* (FB2).
    - DIRECTIONAL_ADVICE cover đầy đủ feature chính (FB3).
    - Boolean cũng dùng advice override (FB4).
    - render_text group theo level, chỉ chi tiết feature fail (FB5).
    - Thêm error_confidence (high/medium/low) vào FeedbackItem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from stage3_features.feature_extractor import (
    FeatureBundle,
    PHASE_DRIVE,
    PHASE_DYNAMIC,
    PHASE_SPRINT_START,
    PHASE_STATIC,
)
from stage4_scoring.scorer import FeatureScore, ScoreReport

GOOD_THRESHOLD = 85.0
WARN_THRESHOLD = 60.0

SYMBOL_GOOD, SYMBOL_WARN, SYMBOL_BAD = "✓", "⚠", "✗"


# =====================================================================
# Phân loại lỗi (v2.2)
# =====================================================================

class ErrorType(str, Enum):
    TECHNIQUE_ERROR    = "technique_error"
    MEASUREMENT_ERROR  = "measurement_error"
    INSUFFICIENT_DATA  = "insufficient_data"


ERROR_TYPE_LABEL = {
    ErrorType.TECHNIQUE_ERROR:   "Lỗi kỹ thuật",
    ErrorType.MEASUREMENT_ERROR: "Lỗi đo lường (pose/camera)",
    ErrorType.INSUFFICIENT_DATA: "Thiếu dữ liệu để chấm",
}


# =====================================================================
# Quality flag nào được coi là "video không đủ tin cậy"?
# V2.2 (FB1): low_fps ĐÃ BỊ LOẠI khỏi danh sách này.
# low_fps chỉ nên cảnh báo ở summary và chỉ override feature slope.
# low_resolution / insufficient_data vẫn giữ vì thực sự làm dữ liệu
# không dùng được ở mức tổng thể.
# =====================================================================

_VIDEO_LEVEL_UNRELIABLE_FLAGS = {
    "low_resolution",
    "insufficient_data",
    "perspective_warning",
    "pose_unstable",
    "out_of_frame",
    "multi_person",
    # "low_fps"  ← ĐÃ XÓA (FB1). Chỉ xử lý ở feature-level cho slope.
}


# Mapping flag → nhóm feature bị ảnh hưởng (FB2 mở rộng)
_FLAG_FEATURE_PREFIXES: Dict[str, tuple] = {
    "wrist":      ("wrist", "elbow", "hand", "arm"),
    "knee_back":  ("knee_back", "hip_back", "thigh_back"),
    "ankle":      ("ankle", "rear_", "leg_ground", "at_knee"),
    "rear_ankle": ("ankle", "rear_", "leg_ground", "at_knee"),
}


# =====================================================================
# Display names - cập nhật đầy đủ theo spec v2.1
# =====================================================================

FEATURE_DISPLAY: Dict[str, Dict[str, str]] = {
    # --- Vào chỗ ---
    "knee_front_angle":      {"name": "Góc gối chân trước", "unit": "°"},
    "knee_back_angle":       {"name": "Góc gối chân sau", "unit": "°"},
    "wrist_width_ratio":     {"name": "Độ rộng hai tay / vai", "unit": "ratio"},
    "trunk_thigh_angle":     {"name": "Góc thân - đùi trước", "unit": "°"},
    "head_trunk_alignment":  {"name": "Độ thẳng hàng đầu - thân", "unit": "°"},
    "com_support_region":    {"name": "Vùng tựa trọng tâm", "unit": "bool"},
    # --- Sẵn sàng ---
    "hip_above_shoulder_norm": {"name": "Độ cao hông so với vai", "unit": "ratio"},
    "shoulder_forward_norm":   {"name": "Vai nhô về trước", "unit": "ratio"},
    # --- Bộc phát ---
    "at_trunk_angle_vertical":   {"name": "Góc thân tại thời điểm bật", "unit": "°"},
    "front_leg_full_extension":  {"name": "Chân trước duỗi gần hoàn toàn", "unit": "bool"},
    "rear_leg_leaves_first":     {"name": "Chân sau rời bàn đạp trước", "unit": "bool"},
    "arm_leg_opposition":        {"name": "Tay đánh đối diện chân sau", "unit": "bool"},
    "leg_ground_angle":          {"name": "Góc chân - mặt đất khi kết thúc đạp", "unit": "°"},
    "at_knee_front_angle":       {"name": "Góc gối trước tại thời điểm bật", "unit": "°"},
    "first_step_timing":         {"name": "Thời điểm bước chân đầu", "unit": "frame"},
    # --- Chạy lao ---
    "step_length_trend":            {"name": "Xu hướng độ dài bước", "unit": "slope"},
    "cadence_trend":                {"name": "Xu hướng tần số bước", "unit": "slope"},
    "contact_position_convergence": {"name": "Độ hội tụ vị trí tiếp đất", "unit": "slope"},
    "trunk_angle_vertical_trend":   {"name": "Xu hướng dựng thân", "unit": "deg/step"},
    # --- Chung ---
    "elbow_left":  {"name": "Góc khuỷu tay trái", "unit": "°"},
    "elbow_right": {"name": "Góc khuỷu tay phải", "unit": "°"},
    "cadence_spm": {"name": "Nhịp bước (cadence)", "unit": "bước/phút"},
}


def _display_name(feature_key: str) -> str:
    if feature_key in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature_key]["name"]
    if feature_key.startswith("dtw_"):
        return f"Độ khớp dáng chuyển động ({feature_key[len('dtw_'):]})"
    if feature_key.startswith("rom_"):
        return f"Biên độ {feature_key[len('rom_'):]}"
    if feature_key.startswith("symmetry_"):
        return f"Độ đối xứng {feature_key[len('symmetry_'):]}"
    return feature_key


# =====================================================================
# Directional advice mặc định (FB3 — mở rộng đầy đủ)
# Ưu tiên: YAML override > DIRECTIONAL_ADVICE > _GENERIC_ADVICE
# =====================================================================

DIRECTIONAL_ADVICE: Dict[tuple, str] = {
    # --- Gối ---
    ("knee_left", "low"):          "Hãy nâng đầu gối trái cao hơn.",
    ("knee_right", "low"):         "Hãy nâng đầu gối phải cao hơn.",
    ("knee_front_angle", "low"):   "Góc gối chân trước đang quá nhỏ — hãy mở gối ra gần 90° hơn.",
    ("knee_front_angle", "high"):  "Góc gối chân trước đang quá lớn — hãy gập gối hơn để hạ thấp hơn.",
    ("knee_back_angle", "low"):    "Góc gối chân sau đang quá nhỏ — hãy duỗi chân sau ra phía sau hơn.",
    ("knee_back_angle", "high"):   "Góc gối chân sau đang quá lớn — hãy gập gối sau lại một chút.",
    ("at_knee_front_angle", "low"):  "Tại thời điểm bật, gối trước chưa mở đủ — hãy đẩy mạnh hơn để duỗi gối.",
    ("at_knee_front_angle", "high"): "Tại thời điểm bật, gối trước mở quá mức — kiểm tra lại lực đạp.",
    # --- Thân / hông ---
    ("trunk", "high"):                   "Thân người đang nghiêng quá nhiều — hãy giữ thân thẳng hơn.",
    ("trunk_thigh_angle", "low"):        "Góc thân–đùi trước quá nhỏ — hãy nâng hông cao hơn hoặc duỗi thân.",
    ("trunk_thigh_angle", "high"):       "Góc thân–đùi trước quá lớn — hãy hạ thấp thân hoặc gập đùi trước hơn.",
    ("at_trunk_angle_vertical", "low"):  "Thân quá nằm ngang lúc bật — hãy dựng thân sớm hơn.",
    ("at_trunk_angle_vertical", "high"): "Thân quá thẳng đứng lúc bật — hãy giữ thân nghiêng về trước hơn.",
    ("hip_above_shoulder_norm", "low"):  "Hông còn thấp so với vai — hãy nâng hông cao hơn ở tư thế sẵn sàng.",
    ("hip_above_shoulder_norm", "high"): "Hông quá cao so với vai — hãy hạ hông xuống một chút.",
    ("shoulder_forward_norm", "low"):    "Vai chưa nhô về trước đủ — hãy đưa trọng tâm về phía trước hơn.",
    ("shoulder_forward_norm", "high"):   "Vai nhô quá xa về trước — dễ mất thăng bằng, hãy kéo vai lại một chút.",
    ("head_trunk_alignment", "high"):    "Đầu và thân lệch nhau nhiều — hãy giữ đầu thẳng hàng với thân.",
    ("leg_ground_angle", "low"):         "Góc chân–mặt đất quá nhỏ lúc kết thúc đạp — hãy duỗi chân đầy đủ hơn.",
    ("leg_ground_angle", "high"):        "Góc chân–mặt đất quá lớn — kiểm tra lại hướng lực đạp.",
    # --- Tay ---
    ("wrist_width_ratio", "low"):  "Hai tay đặt quá hẹp so với vai — hãy mở rộng khoảng cách hai tay.",
    ("wrist_width_ratio", "high"): "Hai tay đặt quá rộng so với vai — hãy thu hẹp khoảng cách hai tay.",
    ("elbow_left", "low"):         "Khuỷu tay trái gập quá nhiều — hãy mở khuỷu ra tự nhiên hơn.",
    ("elbow_left", "high"):        "Khuỷu tay trái mở quá thẳng — hãy giữ khuỷu hơi gập.",
    ("elbow_right", "low"):        "Khuỷu tay phải gập quá nhiều — hãy mở khuỷu ra tự nhiên hơn.",
    ("elbow_right", "high"):       "Khuỷu tay phải mở quá thẳng — hãy giữ khuỷu hơi gập.",
    # --- Boolean sprint_start ---
    ("front_leg_full_extension", "low"): "Chân trước chưa duỗi gần hoàn toàn lúc bật — hãy đẩy hết lực qua hông và gối.",
    ("rear_leg_leaves_first", "low"):    "Chân sau chưa rời bàn đạp trước chân trước — hãy chủ động nhấc chân sau sớm hơn.",
    ("arm_leg_opposition", "low"):       "Tay và chân chưa đối xứng (cùng bên) — hãy đánh tay đối diện với chân sau.",
    ("com_support_region", "low"):       "Trọng tâm nằm ngoài vùng tựa — hãy điều chỉnh vị trí bàn tay/bàn chân.",
    # --- Drive / cadence ---
    ("cadence_spm", "low"):  "Nhịp bước hơi chậm — hãy tăng tần suất bước chân.",
    ("cadence_spm", "high"): "Nhịp bước hơi nhanh so với chuẩn — hãy chạy tự nhiên hơn.",
    ("step_length_trend", "low"):  "Độ dài bước không tăng đủ qua các bước — hãy đẩy mạnh hơn mỗi bước.",
    ("step_length_trend", "high"): "Độ dài bước tăng quá nhanh — kiểm soát biên độ để tránh mất thăng bằng.",
    ("cadence_trend", "low"):      "Tần số bước không tăng đủ — hãy tăng nhịp chân sớm hơn.",
    ("cadence_trend", "high"):     "Tần số bước tăng quá mức — giữ nhịp ổn định hơn.",
    ("contact_position_convergence", "low"):  "Vị trí tiếp đất chưa hội tụ về dưới thân — hãy hạ chân gần thân hơn.",
    ("trunk_angle_vertical_trend", "low"):    "Thân dựng dậy quá chậm — hãy nâng thân dần qua từng bước.",
    ("trunk_angle_vertical_trend", "high"):   "Thân dựng dậy quá nhanh — giữ thân nghiêng lâu hơn trong giai đoạn đầu.",
    ("first_step_timing", "low"):  "Bước chân đầu quá sớm — hãy hoàn tất lực đạp trước khi nhấc chân.",
    ("first_step_timing", "high"): "Bước chân đầu quá muộn — hãy chuyển trọng tâm và nhấc chân sớm hơn.",
}

_GENERIC_ADVICE = "Hãy điều chỉnh để tiệm cận mức tham chiếu của HLV hơn."
_SYMMETRY_ADVICE = "Có sự mất cân đối rõ giữa 2 bên — nên tập trung bài tập cân bằng 2 bên."


# =====================================================================
# Feedback data classes
# =====================================================================

@dataclass
class FeedbackItem:
    feature_key: str
    display_name: str
    status_symbol: str
    student_value: float
    reference_center: float
    unit: str
    score_0_100: float
    message: str
    level: str = "PRIMARY"
    error_type: Optional[ErrorType] = None
    error_confidence: str = "medium"   # high / medium / low (V2.2)
    evidence: Optional[str] = None


@dataclass
class FeedbackReport:
    phase: str
    overall_score: float
    summary_line: str
    items: List[FeedbackItem] = field(default_factory=list)
    unscored_features: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    error_summary: Dict[str, int] = field(default_factory=dict)

    def render_text(self) -> str:
        """
        V2.2 (FB5): Group theo level, chỉ chi tiết feature fail,
        feature đạt gộp thành 1 dòng.
        """
        lines = [self.summary_line, ""]

        if self.error_summary:
            lines.append("Phân loại lỗi:")
            for etype, count in self.error_summary.items():
                if count > 0:
                    lines.append(f"  - {ERROR_TYPE_LABEL.get(etype, etype)}: {count}")
            lines.append("")

        # Tách PRIMARY / SECONDARY, pass / fail
        primary_fail: List[FeedbackItem] = []
        secondary_fail: List[FeedbackItem] = []
        n_pass = 0

        for item in self.items:
            is_fail = (
                np.isnan(item.score_0_100)
                or item.score_0_100 < WARN_THRESHOLD
            )
            if not is_fail:
                n_pass += 1
                continue
            if (item.level or "PRIMARY").upper() == "PRIMARY":
                primary_fail.append(item)
            else:
                secondary_fail.append(item)

        if primary_fail:
            lines.append("── PRIMARY (ưu tiên sửa) ──")
            for item in primary_fail:
                prefix = ""
                if item.error_type:
                    conf = f"/{item.error_confidence}" if item.error_confidence else ""
                    prefix = f"[{ERROR_TYPE_LABEL.get(item.error_type, '')}{conf}] "
                lines.append(
                    f"{item.status_symbol} {prefix}{item.display_name}: {item.message}"
                )
            lines.append("")

        if secondary_fail:
            lines.append("── SECONDARY ──")
            for item in secondary_fail:
                prefix = ""
                if item.error_type:
                    conf = f"/{item.error_confidence}" if item.error_confidence else ""
                    prefix = f"[{ERROR_TYPE_LABEL.get(item.error_type, '')}{conf}] "
                lines.append(
                    f"{item.status_symbol} {prefix}{item.display_name}: {item.message}"
                )
            lines.append("")

        if n_pass > 0:
            lines.append(f"✓ {n_pass} chỉ số đạt yêu cầu.")
            lines.append("")

        if self.quality_flags:
            lines.append("Cảnh báo chất lượng dữ liệu:")
            for flag in self.quality_flags:
                lines.append(f"  - {flag}")
            lines.append("")

        if self.unscored_features:
            lines.append(
                "(Chưa có khoảng chuẩn cho: "
                + ", ".join(self.unscored_features)
                + ")"
            )
        return "\n".join(lines)


# =====================================================================
# Helpers
# =====================================================================

def _status_symbol_with_thresholds(
    score: float, good_threshold: float, warn_threshold: float
) -> str:
    if np.isnan(score):
        return SYMBOL_WARN
    if score >= good_threshold:
        return SYMBOL_GOOD
    if score >= warn_threshold:
        return SYMBOL_WARN
    return SYMBOL_BAD


def _has_video_level_unreliable_flag(quality_flags: List[str]) -> bool:
    """
    True nếu video có bất kỳ quality flag tổng nào khiến kết quả đo không
    đáng tin ở mức tổng thể (không chỉ 1 nhóm joint cụ thể).
    V2.2: low_fps KHÔNG còn nằm trong tập này.
    """
    if not quality_flags:
        return False
    flags_lower = {f.lower() for f in quality_flags}
    return bool(flags_lower & _VIDEO_LEVEL_UNRELIABLE_FLAGS)


def _has_low_fps(quality_flags: List[str]) -> bool:
    if not quality_flags:
        return False
    return any("low_fps" in f.lower() for f in quality_flags)


def _deviation_ratio(fs: FeatureScore) -> float:
    """
    |student - ref_center| / tolerance (nếu có).
    Dùng để quyết định TECHNIQUE vs MEASUREMENT khi có low_fps (FB1 bước 3).
    """
    if fs.reference_center is None or np.isnan(fs.reference_center):
        return 0.0
    if np.isnan(fs.student_value):
        return 0.0
    diff = abs(fs.student_value - fs.reference_center)
    # tolerance có thể nằm trong fs (nếu Scorer gắn), fallback ước lượng
    tol = getattr(fs, "tolerance", None)
    if tol is None or tol <= 0 or (isinstance(tol, float) and np.isnan(tol)):
        # Ước lượng thô: 10% giá trị tham chiếu hoặc 5 đơn vị
        tol = max(abs(fs.reference_center) * 0.1, 5.0)
    return diff / tol


def _classify_error(
    fs: FeatureScore, quality_flags: List[str]
) -> tuple:
    """
    Phân loại lỗi dựa trên FeatureScore + quality flags.

    Returns
    -------
    (Optional[ErrorType], confidence_str)
        confidence ∈ {"high", "medium", "low"}

    Thứ tự ưu tiên (V2.2):
        1. Feature INFO → None.
        2. Score NaN → INSUFFICIENT_DATA.
        3. Score >= WARN → None (đạt).
        4. Video-level unreliable (KHÔNG gồm low_fps) → MEASUREMENT.
        5. Feature-level flag cụ thể (wrist→elbow, knee_back, ankle…) → MEASUREMENT.
        6. low_fps + slope → MEASUREMENT.
        7. low_fps + deviation lớn (>3× tol) → TECHNIQUE (confidence medium).
           low_fps + deviation nhỏ → MEASUREMENT (confidence low).
        8. Còn lại → TECHNIQUE_ERROR.
    """
    if fs.is_info_only:
        return None, "high"

    if np.isnan(fs.score_0_100):
        return ErrorType.INSUFFICIENT_DATA, "high"

    if fs.score_0_100 >= WARN_THRESHOLD:
        return None, "high"

    # Rule 4: video-level unreliable (không gồm low_fps)
    if _has_video_level_unreliable_flag(quality_flags):
        return ErrorType.MEASUREMENT_ERROR, "high"

    # Rule 5: feature-level quality flag (FB2 mở rộng)
    feat_key_lower = fs.feature_key.lower()
    for flag in quality_flags:
        flag_lower = flag.lower()
        for flag_token, prefixes in _FLAG_FEATURE_PREFIXES.items():
            if flag_token in flag_lower:
                if any(p in feat_key_lower for p in prefixes):
                    return ErrorType.MEASUREMENT_ERROR, "high"

    # Rule 6: low_fps + slope
    if _has_low_fps(quality_flags) and fs.scoring_type == "slope":
        return ErrorType.MEASUREMENT_ERROR, "medium"

    # Rule 7: low_fps + numeric → dùng độ lớn deviation
    if _has_low_fps(quality_flags) and fs.scoring_type in (
        "range", "numeric", "frame_window"
    ):
        ratio = _deviation_ratio(fs)
        if ratio > 3.0:
            # Lệch quá lớn so với noise band → có thể là kỹ thuật thật
            return ErrorType.TECHNIQUE_ERROR, "medium"
        return ErrorType.MEASUREMENT_ERROR, "low"

    # Rule 8: không có flag chất lượng → lỗi kỹ thuật
    return ErrorType.TECHNIQUE_ERROR, "high"


def _build_message(
    fs: FeatureScore, display_name: str, evidence: Optional[str]
) -> str:
    unit = fs.unit or ""
    if np.isnan(fs.score_0_100):
        return "Không đủ dữ liệu để chấm chỉ số này."

    if fs.scoring_type == "boolean":
        if fs.is_in_range:
            return "Đạt yêu cầu."
        return "Chưa đạt yêu cầu kỹ thuật."

    if fs.scoring_type == "dtw":
        return f"Điểm tương đồng với mẫu tham chiếu: {fs.score_0_100:.0f}/100."

    if fs.is_in_range:
        return (
            f"Đo được {fs.student_value:.1f}{unit}, nằm trong khoảng tham chiếu. "
            f"Đạt yêu cầu, tiếp tục duy trì."
        )

    if fs.reference_center is not None and not np.isnan(fs.reference_center):
        ref_center = fs.reference_center
    else:
        ref_center = 0.0

    direction = "thấp hơn" if fs.student_value < ref_center else "cao hơn"
    diff = abs(fs.student_value - ref_center)

    base = (
        f"Đo được {fs.student_value:.1f}{unit}, {direction} mức tham chiếu "
        f"({ref_center:.1f}{unit}) khoảng {diff:.1f}{unit}."
    )

    if fs.score_0_100 >= GOOD_THRESHOLD:
        return f"{base} Đạt yêu cầu, tiếp tục duy trì."

    if evidence:
        return f"{base} (Thể hiện {evidence}.)"
    return base


def _build_message_with_overrides(
    fs: FeatureScore,
    display_name: str,
    evidence: Optional[str],
    advice_overrides: Dict[tuple, str],
) -> str:
    if np.isnan(fs.score_0_100):
        return "Không đủ dữ liệu để chấm chỉ số này."

    # V2.2 (FB4): boolean cũng dùng override khi fail
    if fs.scoring_type == "boolean":
        if fs.is_in_range or fs.score_0_100 >= GOOD_THRESHOLD:
            return _build_message(fs, display_name, evidence)
        # Boolean fail → direction luôn "low" (expected = 1.0 / True)
        override = advice_overrides.get((fs.feature_key, "low"))
        if not override:
            override = DIRECTIONAL_ADVICE.get((fs.feature_key, "low"))
        if override:
            return f"Chưa đạt yêu cầu kỹ thuật. {override}"
        return "Chưa đạt yêu cầu kỹ thuật."

    if fs.is_in_range or fs.score_0_100 >= GOOD_THRESHOLD:
        return _build_message(fs, display_name, evidence)

    ref_center = fs.reference_center if fs.reference_center is not None else 0.0
    direction = "low" if fs.student_value < ref_center else "high"

    # Ưu tiên: YAML override > DIRECTIONAL_ADVICE > generic
    override = advice_overrides.get((fs.feature_key, direction))
    if not override:
        override = DIRECTIONAL_ADVICE.get((fs.feature_key, direction))

    unit = fs.unit or ""
    diff = abs(fs.student_value - ref_center)
    direction_text = "thấp hơn" if direction == "low" else "cao hơn"
    base = (
        f"Đo được {fs.student_value:.1f}{unit}, {direction_text} mức tham chiếu "
        f"({ref_center:.1f}{unit}) khoảng {diff:.1f}{unit}."
    )

    if override:
        if evidence:
            return f"{base} {override} (Thể hiện {evidence}.)"
        return f"{base} {override}"

    if evidence:
        return f"{base} (Thể hiện {evidence}.)"
    return f"{base} {_GENERIC_ADVICE}"


def _extract_quality_flags(bundle: Optional[FeatureBundle]) -> List[str]:
    """
    Trích xuất quality flags từ FeatureBundle nếu có.

    Chấp nhận cả 2 dạng:
        - Dict[str, bool]: lấy key nào có value=True.
        - List[str]: dùng trực tiếp.
    """
    if bundle is None:
        return []
    flags_raw = getattr(bundle, "quality_flags", None) or {}
    if isinstance(flags_raw, dict):
        return [str(k) for k, v in flags_raw.items() if v]
    if isinstance(flags_raw, (list, tuple)):
        return [str(f) for f in flags_raw]
    return []


# =====================================================================
# FeedbackGenerator
# =====================================================================

class FeedbackGenerator:
    def __init__(
        self,
        good_threshold: float = GOOD_THRESHOLD,
        warn_threshold: float = WARN_THRESHOLD,
        advice_overrides: Optional[Dict[tuple, str]] = None,
    ):
        self.good_threshold = good_threshold
        self.warn_threshold = warn_threshold
        self.advice_overrides = advice_overrides or {}

    def generate(
        self,
        score_report: ScoreReport,
        bundle: Optional[FeatureBundle] = None,
    ) -> FeedbackReport:
        quality_flags = _extract_quality_flags(bundle)
        items: List[FeedbackItem] = []

        # Sort: điểm thấp trước
        ordered = sorted(
            score_report.feature_scores,
            key=lambda fs: (np.inf if np.isnan(fs.score_0_100) else fs.score_0_100),
        )

        error_summary: Dict[str, int] = {
            ErrorType.TECHNIQUE_ERROR.value: 0,
            ErrorType.MEASUREMENT_ERROR.value: 0,
            ErrorType.INSUFFICIENT_DATA.value: 0,
        }

        for fs in ordered:
            display_name = _display_name(fs.feature_key)
            message = _build_message_with_overrides(
                fs, display_name, None, self.advice_overrides
            )
            symbol = _status_symbol_with_thresholds(
                fs.score_0_100, self.good_threshold, self.warn_threshold
            )
            error_type, err_conf = _classify_error(fs, quality_flags)
            if error_type is not None:
                error_summary[error_type.value] += 1

            items.append(
                FeedbackItem(
                    feature_key=fs.feature_key,
                    display_name=display_name,
                    status_symbol=symbol,
                    student_value=fs.student_value,
                    reference_center=fs.reference_center,
                    unit=fs.unit,
                    score_0_100=fs.score_0_100,
                    message=message,
                    level=fs.level,
                    error_type=error_type,
                    error_confidence=err_conf,
                )
            )

        # Summary line
        if score_report.phase == "vao_cho":
            phase_label = "vào chỗ"
        elif score_report.phase == PHASE_STATIC:
            phase_label = "tư thế tĩnh"
        elif score_report.phase == PHASE_SPRINT_START:
            phase_label = "bộc phát xuất phát"
        elif score_report.phase == PHASE_DRIVE:
            phase_label = "chạy lao"
        elif score_report.phase == PHASE_DYNAMIC:
            phase_label = "chuyển động"
        else:
            phase_label = score_report.phase

        overall = score_report.overall_score
        if not np.isnan(overall):
            summary_line = (
                f"Điểm {phase_label}: {overall:.0f}/100 "
                f"(PRIMARY fail: {score_report.n_primary_failed}, "
                f"SECONDARY fail: {score_report.n_secondary_failed})"
            )
        else:
            summary_line = f"Điểm {phase_label}: chưa đủ dữ liệu để chấm"

        return FeedbackReport(
            phase=score_report.phase,
            overall_score=overall,
            summary_line=summary_line,
            items=items,
            unscored_features=[
                _display_name(k) for k in score_report.unscored_features
            ],
            quality_flags=quality_flags,
            error_summary=error_summary,
        )