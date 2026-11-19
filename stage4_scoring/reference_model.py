"""
stage4_scoring/reference_model.py
====================================
Mục đích : Xây khoảng chuẩn (reference range) cho từng đặc trưng, theo từng
           pha (vao_cho/static/dynamic/sprint_start/drive_phase), TỪ HAI
           NGUỒN có thể kết hợp:

    (1) THỐNG KÊ  : suy ra từ nhiều video HLV (build_static/build_dynamic,
                     như trước) — phù hợp khi có đủ video HLV để tin cậy.
    (2) CỐ ĐỊNH   : khai báo trực tiếp theo giáo trình/tài liệu kỹ thuật đã
                     được xác lập sẵn (ví dụ: xuất phát thấp — góc gối chân
                     trước ~90°, chân sau 120-140°, hông cao hơn vai 10-20cm)
                     — dùng khi số lượng video HLV quá ít để tin cậy thống
                     kê thuần túy, hoặc khi tiêu chuẩn kỹ thuật đã được xác
                     lập rõ ràng và không nên chỉ dựa vào vài mẫu tự thu
                     thập (xem FixedRangeSpec bên dưới).

Cả 2 nguồn đều tạo ra CÙNG MỘT kiểu dữ liệu (ReferenceRange: center + spread
+ method), nên scorer.py dùng được NGUYÊN VẸN, không cần sửa gì — chỉ khác
ở CÁCH ra đời con số center/spread đó.

QUAN TRỌNG (nhắc lại từ bản gốc): mỗi VIDEO HLV đóng góp ĐÚNG 1 điểm dữ liệu
cho khoảng chuẩn ở cấp "dynamic" (cadence, ROM, symmetry) — không lấy trung
bình cả ngàn frame của 1 video làm nhiều mẫu độc lập. Ở cấp "static", các
frame trong đoạn giữ tư thế ổn định của NHIỀU video được GỘP lại (pooled).

V2.1.1 - Hỗ trợ PHASE_DRIVE:
    - Thêm field drive_phase_ranges vào ReferenceModel.
    - Thêm "drive_phase" vào _ASSIGNABLE_PHASES.
    - _assign_ranges() route drive_phase -> model.drive_phase_ranges.
    - get_range() tra cứu thêm drive_phase_ranges.
    - to_dict()/load() serialize field mới.
    - build_fixed()/build_mixed() chấp nhận phase=drive_phase (giống
      sprint_start: chỉ fixed, không thống kê vì DrivePhaseSummary có cấu
      trúc riêng khác static_summary).

V2.1.2 - Sửa bug ReferenceModelBuilder.build() với phase 'vao_cho':
    - Trước đây build() luôn gán ranges vào `static_ranges` kể cả khi
      phase == PHASE_INTO_POSITION ('vao_cho'), trong khi Scorer._score_static()
      chấm 'vao_cho' bằng `reference.phase_ranges` -> phase_ranges RỖNG, không
      chấm được.
    - Nay build() tạo ReferenceModel rỗng rồi gọi _assign_ranges() (nơi duy
      nhất phân biệt vao_cho -> phase_ranges, static -> static_ranges).
    - n_coach_videos đếm theo đúng `phase` được yêu cầu (trước đó luôn đếm
      theo PHASE_STATIC nên 'vao_cho' luôn ra 0).
    - Pipeline chính KHÔNG bị ảnh hưởng (dùng build_mixed/build_fixed vốn đã
      gọi _assign_ranges).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stage3_features.feature_extractor import (
    FeatureBundle,
    PHASE_DRIVE,
    PHASE_DYNAMIC,
    PHASE_INTO_POSITION,
    PHASE_SPRINT_START,
    PHASE_STATIC,
)

# Khóa "ankle_is_approx" là cờ trạng thái model, không phải chỉ số kỹ thuật
# -> loại khỏi mọi khoảng chuẩn/chấm điểm.
_STATIC_EXCLUDED_KEYS = {"ankle_is_approx"}

MIN_SPREAD_FLOOR = 1e-3  # tránh spread=0 khiến z-score chia cho 0

# Danh sách phase co the gan ranges vao ReferenceModel.
# drive_phase duoc them o V2.1.1 de dong bo voi Pipeline.from_config() va
# Scorer._score_drive_phase().
_ASSIGNABLE_PHASES = (
    PHASE_INTO_POSITION,
    PHASE_STATIC,
    PHASE_DYNAMIC,
    PHASE_SPRINT_START,
    PHASE_DRIVE,
)


@dataclass
class ReferenceRange:
    """Khoảng chuẩn cho 1 đặc trưng, dù đến từ thống kê hay khai báo cố định."""

    feature_key: str
    center: float          # mean/median (thống kê) hoặc điểm giữa (min+max)/2 (cố định)
    spread: float          # std/IQR (thống kê) hoặc (max-min)/2 (cố định) — đã có floor tối thiểu
    method: str            # "mean_std" | "median_iqr" | "fixed"
    n_samples: int         # số video/frame dùng ước lượng (thống kê); 0 nếu "fixed"
    unit: str = ""

    # Convenience properties: scorer.py và các module khác dùng trực tiếp
    # min_value / max_value (thay vì tự tính từ center ± spread).
    @property
    def min_value(self) -> float:
        return self.center - self.spread

    @property
    def max_value(self) -> float:
        return self.center + self.spread

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FixedRangeSpec:
    """Một khoảng chuẩn CỐ ĐỊNH do người dùng khai báo trực tiếp — ví dụ lấy
    từ giáo trình/tài liệu kỹ thuật (góc gối ~90°, 120-140°...), KHÔNG suy
    ra từ thống kê vài video HLV.

    Dùng khi:
        - Số lượng video HLV quá ít để thống kê đáng tin cậy (ví dụ pha
          "Sẵn sàng" của xuất phát thấp, nơi góc độ đã được xác lập sẵn
          trong giáo trình điền kinh).
        - Muốn CHỐT một tiêu chuẩn kỹ thuật đã biết trước, không để vài
          video tự quay (có thể lệch chuẩn) âm thầm định nghĩa lại "đúng"
          là gì.
    """

    feature_key: str
    min_value: float
    max_value: float
    unit: str = ""
    source_note: str = ""  # ví dụ: "Giáo trình Điền kinh — Xuất phát thấp, Pha 2 (Set)"

    def to_reference_range(self) -> ReferenceRange:
        if self.max_value < self.min_value:
            raise ValueError(
                f"FixedRangeSpec cho '{self.feature_key}': max_value phải >= min_value "
                f"(nhận min={self.min_value}, max={self.max_value})"
            )
        center = (self.min_value + self.max_value) / 2.0
        spread = max((self.max_value - self.min_value) / 2.0, MIN_SPREAD_FLOOR)
        return ReferenceRange(
            feature_key=self.feature_key, center=center, spread=spread,
            method="fixed", n_samples=0, unit=self.unit,
        )


@dataclass
class ReferenceModel:
    sport: str
    phase: str
    method: str
    static_ranges: Dict[str, ReferenceRange] = field(default_factory=dict)
    dynamic_ranges: Dict[str, ReferenceRange] = field(default_factory=dict)
    # Khoảng chuẩn cho pha rời rạc (ví dụ "sprint_start") — cùng kiểu dữ
    # liệu với 2 dict trên, để scorer.py có thể dùng lại nguyên vẹn công
    # thức chấm điểm khi được nối dây (việc nối dây chưa thực hiện ở đây).
    sprint_start_ranges: Dict[str, ReferenceRange] = field(default_factory=dict)
    phase_ranges: Dict[str, ReferenceRange] = field(default_factory=dict)
    # V2.1.1 - Khoảng chuẩn cho pha chạy lao 25-30m đầu (drive_phase).
    # Chứa 4 trend features dạng slope (step_length_trend, cadence_trend,
    # contact_position_convergence, trunk_angle_vertical_trend).
    drive_phase_ranges: Dict[str, ReferenceRange] = field(default_factory=dict)
    # đường cong trung bình theo % chu kỳ của HLV, dùng để vẽ overlay hoặc DTW
    reference_curves: Dict[str, List[float]] = field(default_factory=dict)
    n_coach_videos: int = 0

    def get_range(self, feature_key: str) -> Optional[ReferenceRange]:
        return (
            self.static_ranges.get(feature_key)
            or self.dynamic_ranges.get(feature_key)
            or self.sprint_start_ranges.get(feature_key)
            or self.drive_phase_ranges.get(feature_key)
            or self.phase_ranges.get(feature_key)
        )

    def to_dict(self) -> Dict:
        return {
            "sport": self.sport,
            "phase": self.phase,
            "method": self.method,
            "n_coach_videos": self.n_coach_videos,
            "static_ranges": {k: v.to_dict() for k, v in self.static_ranges.items()},
            "dynamic_ranges": {k: v.to_dict() for k, v in self.dynamic_ranges.items()},
            "sprint_start_ranges": {k: v.to_dict() for k, v in self.sprint_start_ranges.items()},
            "drive_phase_ranges": {k: v.to_dict() for k, v in self.drive_phase_ranges.items()},
            "phase_ranges": {k: v.to_dict() for k, v in self.phase_ranges.items()},
            "reference_curves": self.reference_curves,
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReferenceModel":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        model = cls(
            sport=d["sport"], phase=d["phase"], method=d["method"],
            n_coach_videos=d.get("n_coach_videos", 0),
            reference_curves=d.get("reference_curves", {}),
        )
        model.static_ranges = {
            k: ReferenceRange(**v) for k, v in d.get("static_ranges", {}).items()
        }
        model.dynamic_ranges = {
            k: ReferenceRange(**v) for k, v in d.get("dynamic_ranges", {}).items()
        }
        model.sprint_start_ranges = {
            k: ReferenceRange(**v) for k, v in d.get("sprint_start_ranges", {}).items()
        }
        # V2.1.1: doc drive_phase_ranges neu co trong file JSON.
        # Backward compat: file JSON cu (truoc V2.1.1) khong co key nay ->
        # dict rong, khong crash.
        model.drive_phase_ranges = {
            k: ReferenceRange(**v) for k, v in d.get("drive_phase_ranges", {}).items()
        }
        model.phase_ranges = {
            k: ReferenceRange(**v) for k, v in d.get("phase_ranges", {}).items()
        }
        return model


def _center_spread(values: np.ndarray, method: str) -> tuple:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if method == "median_iqr":
        center = float(np.median(values))
        q75, q25 = np.percentile(values, [75, 25])
        spread = float(q75 - q25)
    elif method == "mean_std":
        center = float(np.mean(values))
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    else:
        raise ValueError(f"method không hỗ trợ: {method}")
    spread = max(spread, MIN_SPREAD_FLOOR)
    return center, spread


class ReferenceModelBuilder:
    def __init__(self, method: str = "median_iqr"):
        if method not in ("median_iqr", "mean_std", "fixed"):
            raise ValueError("method phải là 'median_iqr', 'mean_std' hoặc 'fixed'")
        self.method = method
        # 'fixed' (khai trong sport_configs/<mon>.yaml) nghĩa là khoảng chuẩn đến
        # từ giáo trình qua build_fixed()/build_mixed(); phần THỐNG KÊ (nếu có
        # bundle HLV trong build_mixed) dùng median_iqr làm mặc định an toàn.
        self._stat_method = "median_iqr" if method == "fixed" else method

    # ------------------------------------------------------------------
    def build_static(self, bundles: List[FeatureBundle]) -> Dict[str, ReferenceRange]:
        """Gộp (pool) toàn bộ frame hợp lệ trong đoạn giữ tư thế của MỌI
        video HLV thành 1 phân bố cho mỗi góc, rồi lấy center/spread."""
        pooled: Dict[str, List[float]] = {}
        for bundle in bundles:
            if bundle.phase not in (PHASE_INTO_POSITION, PHASE_STATIC) or bundle.static_summary is None:
                continue
            df = bundle.normalized_df
            window = df[
                (df.frame_idx >= bundle.static_summary.segment_start_frame)
                & (df.frame_idx <= bundle.static_summary.segment_end_frame)
                & (df.is_valid)
            ]
            for col in window.columns:
                if col.startswith("angle_"):
                    key = col[len("angle_") :]
                    unit = "°"
                elif col.startswith("ratio_"):
                    key = col[len("ratio_") :]
                    unit = "ratio"
                else:
                    continue
                if key in _STATIC_EXCLUDED_KEYS:
                    continue
                pooled.setdefault(key, []).extend(window[col].dropna().tolist())

        ranges: Dict[str, ReferenceRange] = {}
        for key, values in pooled.items():
            arr = np.array(values, dtype=np.float64)
            center, spread = _center_spread(arr, self._stat_method)
            unit = (
                "ratio"
                if key in {
                    "shoulder_width", "hip_width", "wrist_width",
                    "hand_ground_clearance", "hip_above_shoulder",
                    "shoulder_forward",
                }
                or key.startswith(("thigh_", "shank_"))
                else "°"
            )
            ranges[key] = ReferenceRange(
                feature_key=key, center=center, spread=spread,
                method=self._stat_method, n_samples=len(arr), unit=unit,
            )
        if not ranges:
            raise ValueError(
                "Không xây được khoảng chuẩn static: không có bundle HLV hợp lệ nào "
                "(kiểm tra lại bundle.phase=='static' và static_summary có giá trị). "
                "Nếu chưa có đủ video HLV, cân nhắc dùng build_fixed()/build_mixed() "
                "với khoảng chuẩn theo giáo trình thay vì suy thống kê."
            )
        return ranges

    # ------------------------------------------------------------------
    def build_dynamic(self, bundles: List[FeatureBundle]) -> tuple:
        """Mỗi video HLV đóng góp 1 điểm dữ liệu / đặc trưng (KHÔNG pool theo
        frame — xem giải thích trong docstring module). Trả về (dynamic_ranges,
        reference_curves, n_videos_used)."""
        per_feature_samples: Dict[str, List[float]] = {}
        curve_samples: Dict[str, List[np.ndarray]] = {}
        n_videos_used = 0

        for bundle in bundles:
            if bundle.phase != PHASE_DYNAMIC or bundle.dynamic_summary is None:
                continue
            n_videos_used += 1
            d = bundle.dynamic_summary
            per_feature_samples.setdefault("cadence_spm", []).append(d.cadence_spm_mean)
            for angle_key, rom in d.rom_by_angle.items():
                per_feature_samples.setdefault(f"rom_{angle_key}", []).append(rom)
                curve_samples.setdefault(angle_key, []).append(
                    d.mean_curve_by_angle[angle_key]
                )
            for sym_name, sym_val in d.symmetry_index.items():
                if not np.isnan(sym_val):
                    per_feature_samples.setdefault(f"symmetry_{sym_name}", []).append(sym_val)

        if n_videos_used == 0:
            raise ValueError(
                "Không xây được khoảng chuẩn dynamic: không có bundle HLV hợp lệ nào "
                "(kiểm tra lại bundle.phase=='dynamic' và dynamic_summary có giá trị)."
            )

        ranges: Dict[str, ReferenceRange] = {}
        for key, values in per_feature_samples.items():
            arr = np.array(values, dtype=np.float64)
            center, spread = _center_spread(arr, self._stat_method)
            unit = (
                "bước/phút"
                if key == "cadence_spm"
                else ("%" if key.startswith("symmetry_") else "°")
            )
            ranges[key] = ReferenceRange(
                feature_key=key, center=center, spread=spread,
                method=self._stat_method, n_samples=len(arr), unit=unit,
            )

        reference_curves: Dict[str, List[float]] = {}
        for angle_key, curves in curve_samples.items():
            stacked = np.stack(curves, axis=0)
            reference_curves[angle_key] = np.nanmean(stacked, axis=0).tolist()

        return ranges, reference_curves, n_videos_used

    # ------------------------------------------------------------------
    def build(self, sport: str, phase: str, bundles: List[FeatureBundle]) -> ReferenceModel:
        """Xây khoảng chuẩn THUẦN THỐNG KÊ từ dữ liệu HLV. Với pha
        'sprint_start' / 'drive_phase', dùng build_fixed()/build_mixed()
        thay thế (xem lý do trong thông báo lỗi bên dưới)."""
        if self.method == "fixed":
            raise ValueError(
                "ReferenceModelBuilder(method='fixed') không dùng được với build() thuần thống kê — "
                "dùng build_fixed() (chỉ giáo trình) hoặc build_mixed() (giáo trình + video HLV)."
            )
        if phase in (PHASE_INTO_POSITION, PHASE_STATIC):
            static_ranges = self.build_static(bundles)
            model = ReferenceModel(
                sport=sport, phase=phase, method=self.method,
                n_coach_videos=sum(
                    1 for b in bundles if b.phase == phase and b.static_summary
                ),
            )
            # _assign_ranges phân biệt đúng: vao_cho -> phase_ranges,
            # static -> static_ranges.
            self._assign_ranges(model, phase, static_ranges)
            return model
        elif phase == PHASE_DYNAMIC:
            dynamic_ranges, reference_curves, n_videos = self.build_dynamic(bundles)
            return ReferenceModel(
                sport=sport, phase=phase, method=self.method,
                dynamic_ranges=dynamic_ranges, reference_curves=reference_curves,
                n_coach_videos=n_videos,
            )
        elif phase == PHASE_SPRINT_START:
            raise ValueError(
                "Pha 'sprint_start' hiện CHƯA hỗ trợ xây khoảng chuẩn thuần thống kê "
                "qua build() — FeatureBundle của pha này mang 'sprint_start_summary' "
                "(1 sự kiện rời rạc), không phải 'static_summary'/'dynamic_summary' mà "
                "build_static()/build_dynamic() đang gộp theo. Dùng build_fixed() với "
                "khoảng chuẩn theo giáo trình, hoặc build_mixed() nếu muốn kết hợp thêm "
                "dữ liệu HLV khi cơ chế gộp thống kê cho pha này được bổ sung sau này."
            )
        elif phase == PHASE_DRIVE:
            raise ValueError(
                "Pha 'drive_phase' hiện CHƯA hỗ trợ xây khoảng chuẩn thuần thống kê "
                "qua build() — FeatureBundle của pha này mang 'static_summary.feature_mean' "
                "chứa 4 trend features (slope), không phải phân bố frame như static. "
                "Dùng build_fixed() với khoảng chuẩn theo giáo trình (YAML "
                "fixed_reference_ranges.drive_phase), hoặc build_mixed() nếu muốn "
                "kết hợp thêm dữ liệu HLV khi cơ chế gộp thống kê cho pha này được "
                "bổ sung sau này."
            )
        else:
            raise ValueError(
                f"phase phải là một trong {_ASSIGNABLE_PHASES}, nhận '{phase}'"
            )

    # ------------------------------------------------------------------
    def build_fixed(
        self, sport: str, phase: str, specs: List[FixedRangeSpec]
    ) -> ReferenceModel:
        """Xây ReferenceModel HOÀN TOÀN từ khoảng chuẩn cố định (giáo trình),
        KHÔNG cần bất kỳ video HLV nào — dùng cho các pha đã có tiêu chuẩn kỹ
        thuật xác lập sẵn (ví dụ Pha 2 "Sẵn sàng" của xuất phát thấp, hoặc
        Pha 4 "Chạy lao" với 4 trend features)."""
        if not specs:
            raise ValueError(
                "specs rỗng — cần ít nhất 1 FixedRangeSpec để xây khoảng chuẩn cố định"
            )
        ranges = {spec.feature_key: spec.to_reference_range() for spec in specs}
        model = ReferenceModel(sport=sport, phase=phase, method="fixed", n_coach_videos=0)
        self._assign_ranges(model, phase, ranges)
        return model

    def build_mixed(
        self,
        sport: str,
        phase: str,
        bundles: Optional[List[FeatureBundle]] = None,
        fixed_specs: Optional[List[FixedRangeSpec]] = None,
    ) -> ReferenceModel:
        """Kết hợp khoảng chuẩn THỐNG KÊ (từ `bundles`, nếu có đủ video HLV
        và bundle đúng loại) với khoảng chuẩn CỐ ĐỊNH (từ `fixed_specs`, theo
        giáo trình) cho CÙNG một pha.

        Với mỗi feature_key: nếu có mặt trong `fixed_specs`, khoảng CỐ ĐỊNH
        LUÔN THẮNG (ưu tiên vì đã được xác lập theo tài liệu kỹ thuật đáng
        tin cậy hơn 1 mẫu nhỏ tự thu thập); các feature_key còn lại (không
        có khai báo cố định) vẫn dùng khoảng suy ra từ thống kê, nếu có.

        Nếu `bundles` rỗng/None (hoặc không có bundle đúng loại cho `phase`),
        kết quả tương đương build_fixed() thuần túy — không raise lỗi, vì
        đây chính là trường hợp dùng chính (pha chưa có/chưa đủ video HLV).
        """
        statistical_ranges: Dict[str, ReferenceRange] = {}
        reference_curves: Dict[str, List[float]] = {}
        n_coach_videos = 0

        if bundles:
            if phase in (PHASE_INTO_POSITION, PHASE_STATIC):
                usable = [b for b in bundles if b.phase == phase and b.static_summary]
                if usable:
                    statistical_ranges = self.build_static(usable)
                    n_coach_videos = len(usable)
            elif phase == PHASE_DYNAMIC:
                usable = [b for b in bundles if b.phase == PHASE_DYNAMIC and b.dynamic_summary]
                if usable:
                    statistical_ranges, reference_curves, n_coach_videos = self.build_dynamic(usable)
            # Với PHASE_SPRINT_START và PHASE_DRIVE: chưa có cơ chế gộp thống
            # kê (xem build() ở trên) — `bundles` bị bỏ qua một cách CÓ CHỦ
            # ĐÍCH, không raise lỗi, để build_mixed() vẫn dùng được thuần túy
            # như build_fixed().

        fixed_ranges = {
            spec.feature_key: spec.to_reference_range() for spec in (fixed_specs or [])
        }
        merged = dict(statistical_ranges)
        merged.update(fixed_ranges)  # cố định ưu tiên ghi đè thống kê khi trùng khóa

        if not merged:
            raise ValueError(
                "build_mixed() không có gì để xây: cả bundles hợp lệ lẫn fixed_specs đều rỗng."
            )

        model = ReferenceModel(
            sport=sport, phase=phase,
            method=(
                "mixed"
                if statistical_ranges and fixed_ranges
                else ("fixed" if fixed_ranges else self._stat_method)
            ),
            reference_curves=reference_curves,
            n_coach_videos=n_coach_videos,
        )
        self._assign_ranges(model, phase, merged)
        return model

    # ------------------------------------------------------------------
    @staticmethod
    def _assign_ranges(
        model: ReferenceModel, phase: str, ranges: Dict[str, ReferenceRange]
    ) -> None:
        if phase == PHASE_STATIC:
            model.static_ranges = ranges
        elif phase == PHASE_INTO_POSITION:
            model.phase_ranges = ranges
        elif phase == PHASE_DYNAMIC:
            model.dynamic_ranges = ranges
        elif phase == PHASE_SPRINT_START:
            model.sprint_start_ranges = ranges
        elif phase == PHASE_DRIVE:
            model.drive_phase_ranges = ranges
        else:
            raise ValueError(
                f"phase phải là một trong {_ASSIGNABLE_PHASES}, nhận '{phase}'"
            )