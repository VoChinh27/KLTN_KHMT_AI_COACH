"""Đa môn thể thao: đăng ký môn, bài tập, thư mục dữ liệu và «engine» chấm điểm của từng môn.

Giao diện (app_ui.py) KHÔNG biết tên bài tập cụ thể nào; mọi thứ lấy từ `config/sports.yaml`:

    default: badminton
    sports:
      badminton:
        name: Cầu lông (Badminton)
        icon: "🏸"
        engine: badminton3d.engine_badminton:BadmintonEngine     # null = chưa có bộ chấm điểm (chỉ quản lý dữ liệu)
        config: config/badminton3d.yaml                         # môn mới: config/<môn>.yaml
        input: Input                                            # môn mới: Input/<môn>
        reference: Reference                                    # môn mới: Reference/<môn>
        output: Output                                          # môn mới: Output/<môn>
        actions: {ready: {label: ..., short: ..., hand: false}, serve: {...}}

Cầu lông giữ bố cục cũ (Input/, Reference/, Output/, config/badminton3d.yaml) để dòng lệnh `python -m badminton3d` không bị ảnh hưởng.
Môn tạo mới theo quy ước Input/<môn>/, Reference/<môn>/, Output/<môn>/, config/<môn>.yaml.

Hợp đồng của một engine (xem docs/MULTISPORT.md): lớp có `supported_actions`, `quick_settings` và các phương thức
analyze / build_reference / record / run_all / reference_rows. Báo cáo engine trả về dùng cùng cấu trúc report.json
({"status", "score": {"overall", "grade", "errors": [...]}, "warnings", ...}) nên giao diện hiển thị được mọi môn.
"""
from __future__ import annotations

import importlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_NAME = "sports.yaml"
SLUG = re.compile(r"^[a-z][a-z0-9_]{1,23}$")
RESERVED = {"_tong_hop", "_luu_tru", "config", "input", "output", "reference", "models", "docs", "tests"}
DEFAULT_KEY = "badminton"

DEFAULT_REGISTRY = {
    "default": DEFAULT_KEY,
    "sports": {
        DEFAULT_KEY: {
            "name": "Cầu lông (Badminton)", "icon": "🏸", "engine": "badminton3d.engine_badminton:BadmintonEngine",
            "config": "config/badminton3d.yaml", "input": "Input", "reference": "Reference", "output": "Output",
            "hand_label": "Tay cầm vợt",
            "tips": ["Máy cố định (chân đế), thấy toàn thân", "Người chiếm ≥ 50% chiều cao khung hình", "Cùng góc quay với video HLV",
                     "Bài phát cầu: quay từ phía tay vợt", "Từ 30 FPS, đủ sáng"],
            "actions": {"ready": {"label": "Tư thế sẵn sàng (ready)", "short": "Sẵn sàng", "hand": False},
                        "serve": {"label": "Phát cầu trái tay (serve)", "short": "Phát cầu", "hand": True}},
        }
    },
}

GENERIC_TIPS = ["Máy cố định, thấy toàn thân người tập", "Người chiếm ≥ 50% chiều cao khung hình", "Cùng góc quay với video HLV", "Từ 30 FPS, đủ sáng"]


class SportError(Exception):
    """Lỗi có thông điệp tiếng Việt để hiện thẳng cho người dùng."""


@dataclass
class Action:
    key: str
    label: str
    short: str = ""
    hand: bool = False                 # bài này có cần chọn tay thuận / tay cầm dụng cụ không


@dataclass
class Sport:
    key: str
    name: str
    icon: str
    engine_path: Optional[str]
    config: Path
    input: Path
    reference: Path
    output: Path
    actions: Dict[str, Action]
    root: Path = ROOT
    hand_label: str = "Tay thuận"
    tips: List[str] = field(default_factory=list)
    _engine: object = field(default=None, repr=False, compare=False)

    @property
    def tong_hop(self) -> Path:
        return self.output / "_tong_hop"

    @property
    def engine(self):
        if self._engine is None and self.engine_path:
            self._engine = load_engine(self.engine_path)
        return self._engine

    def supports(self, action: str) -> bool:
        e = self.engine
        return bool(e) and action in getattr(e, "supported_actions", ())

    def action_of(self, name: str) -> Optional[str]:
        """Bài tập suy ra từ tên file (<vai trò>_<bài>_<số>.mp4): khóa dài nhất khớp được ưu tiên."""
        base = Path(name).name.lower()
        for k in sorted(self.actions, key=len, reverse=True):
            if k.lower() in base:
                return k
        return None

    def reference_file(self, action: str) -> Path:
        return self.reference / f"{action}.json"

    def label(self, action: Optional[str]) -> str:
        a = self.actions.get(action or "")
        return a.label if a else "không rõ"

    def short(self, action: Optional[str]) -> str:
        a = self.actions.get(action or "")
        return (a.short or a.label) if a else "?"


# ---------------------------------------------------------------- đăng ký
def registry_path(root: Path = ROOT) -> Path:
    return Path(root) / "config" / REGISTRY_NAME


def load_registry(root: Path = ROOT) -> dict:
    p = registry_path(root)
    if not p.is_file():
        save_registry(DEFAULT_REGISTRY, root)
    try:
        reg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SportError(f"File {p.name} bị lỗi cú pháp: {exc}")
    if not isinstance(reg.get("sports"), dict) or not reg["sports"]:
        raise SportError(f"{p.name} chưa khai báo môn thể thao nào (mục 'sports').")
    return reg


def save_registry(reg: dict, root: Path = ROOT) -> None:
    p = registry_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "# config/sports.yaml — danh sách môn thể thao và bài tập (giao diện đọc file này; xem docs/MULTISPORT.md)\n"
    p.write_text(header + yaml.safe_dump(reg, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")


def _build(key: str, d: dict, root: Path) -> Sport:
    acts = {}
    for ak, ad in (d.get("actions") or {}).items():
        ad = ad or {}
        acts[str(ak)] = Action(str(ak), str(ad.get("label", ak)), str(ad.get("short", "") or ad.get("label", ak)), bool(ad.get("hand", False)))
    return Sport(key=key, name=str(d.get("name", key)), icon=str(d.get("icon", "🏅")), engine_path=d.get("engine") or None,
                 config=Path(root) / d.get("config", f"config/{key}.yaml"), input=Path(root) / d.get("input", f"Input/{key}"),
                 reference=Path(root) / d.get("reference", f"Reference/{key}"), output=Path(root) / d.get("output", f"Output/{key}"),
                 actions=acts, root=Path(root), hand_label=str(d.get("hand_label", "Tay thuận")), tips=list(d.get("tips") or GENERIC_TIPS))


def list_sports(root: Path = ROOT) -> Dict[str, Sport]:
    reg = load_registry(root)
    return {k: _build(k, v or {}, root) for k, v in reg["sports"].items()}


def default_sport_key(root: Path = ROOT) -> str:
    reg = load_registry(root)
    return reg.get("default") if reg.get("default") in reg["sports"] else next(iter(reg["sports"]))


def get_sport(key: Optional[str] = None, root: Path = ROOT) -> Sport:
    sports = list_sports(root)
    key = key or default_sport_key(root)
    if key not in sports:
        raise SportError(f"Không có môn thể thao «{key}» trong config/{REGISTRY_NAME}.")
    return sports[key]


# ---------------------------------------------------------------- engine
_ENGINE_CACHE: Dict[str, object] = {}


def load_engine(path: str):
    """'package.module:LớpEngine' -> đối tượng engine (khởi tạo không tham số). Lỗi import thì ném SportError rõ ràng."""
    if path in _ENGINE_CACHE:
        return _ENGINE_CACHE[path]
    try:
        mod, _, cls = path.partition(":")
        obj = getattr(importlib.import_module(mod), cls)()
    except Exception as exc:  # noqa: BLE001
        raise SportError(f"Không nạp được engine «{path}»: {exc}")
    _ENGINE_CACHE[path] = obj
    return obj


# ---------------------------------------------------------------- tạo môn / bài tập mới
def _slug_ok(value: str, what: str) -> str:
    v = (value or "").strip().lower()
    if not SLUG.match(v):
        raise SportError(f"{what} «{value}» không hợp lệ: dùng chữ thường không dấu, số, dấu gạch dưới; bắt đầu bằng chữ; dài 2–24 ký tự (ví dụ: tennis, chay_bo).")
    if v in RESERVED:
        raise SportError(f"{what} «{v}» trùng tên thư mục hệ thống, hãy chọn tên khác.")
    return v


def parse_actions(text: str) -> List[Tuple[str, str]]:
    """Mỗi dòng: «khoa | Nhãn hiển thị» (hoặc chỉ «khoa»). Trả danh sách (khóa, nhãn)."""
    out, seen = [], set()
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        k, _, lab = ln.partition("|")
        key = _slug_ok(k, "Mã bài tập")
        if key in seen:
            raise SportError(f"Mã bài tập «{key}» bị trùng.")
        seen.add(key)
        out.append((key, lab.strip() or key.replace("_", " ").capitalize()))
    return out


def config_template(key: str, name: str, actions: Sequence[Tuple[str, str]]) -> str:
    acts = "".join(f"  {k}:\n    label: \"{lab}\"\n    tolerance_floor_deg: 10.0      # dung sai tối thiểu cho mỗi góc (độ)\n    min_segment_s: 0.5             # đoạn chuyển động tối thiểu để chấm (giây)\n"
                   for k, lab in actions) or "  {}\n"
    return f"""# config/{key}.yaml — cấu hình mẫu cho môn «{name}»
# File này do giao diện tạo sẵn. Hãy điều chỉnh các ngưỡng cho phù hợp, rồi nối với engine chấm điểm (xem docs/MULTISPORT.md).
sport:
  key: {key}
  name: "{name}"

scoring:
  pass_score: 70                 # điểm tối thiểu để coi là «đạt»
  critical_cap: 69               # lỗi nghiêm trọng thì điểm tối đa = giá trị này (null = tắt)
  grades: [[85, "Tốt"], [70, "Khá"], [50, "Cần cải thiện"], [0, "Chưa đạt"]]

video_qc:
  fps_min: 24                    # dưới ngưỡng: cảnh báo
  shake_warn_pct: 0.25           # rung máy (% bề rộng khung)
  subject_min_frac: 0.25         # người cao dưới ngần này chiều cao khung: cảnh báo

actions:
{acts}"""


def create_sport(key: str, name: str, icon: str = "🏅", actions: Sequence[Tuple[str, str]] = (), root: Path = ROOT) -> Sport:
    """Tạo môn mới: Input/<môn>/, Reference/<môn>/, Output/<môn>/, config/<môn>.yaml mẫu và đăng ký vào config/sports.yaml (có hoàn tác nếu lỗi)."""
    key = _slug_ok(key, "Mã môn")
    name = (name or "").strip()
    if not name:
        raise SportError("Hãy nhập tên môn thể thao.")
    if not actions:
        raise SportError("Hãy khai báo ít nhất 1 bài tập (mỗi dòng: ma_bai | Nhãn hiển thị).")
    reg = load_registry(root)
    if key in reg["sports"]:
        raise SportError(f"Môn «{key}» đã tồn tại.")
    root = Path(root)
    created: List[Path] = []
    try:
        for d, readme in ((root / "Input" / key, f"# Input/{key}/\n\nĐặt video vào đây, tên dạng <vai trò>_<bài tập>_<số>.mp4 (ví dụ coach_{(actions[0][0] if actions else 'bai')}_01.mp4, trainee_{(actions[0][0] if actions else 'bai')}_01.mp4).\n"),
                          (root / "Reference" / key, f"# Reference/{key}/\n\nChuẩn HLV của môn {name}: <bài tập>.json (dựng bằng giao diện hoặc engine của môn).\n"),
                          (root / "Output" / key, "")):
            if not d.exists():
                d.mkdir(parents=True)
                created.append(d)
            if readme:
                (d / "README.md").write_text(readme, encoding="utf-8")
        cfg = root / "config" / f"{key}.yaml"
        if cfg.exists():
            raise SportError(f"File cấu hình config/{key}.yaml đã tồn tại, hãy chọn mã môn khác.")
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(config_template(key, name, actions), encoding="utf-8")
        created.append(cfg)
        reg["sports"][key] = {"name": name, "icon": icon or "🏅", "engine": None, "config": f"config/{key}.yaml", "input": f"Input/{key}", "reference": f"Reference/{key}",
                              "output": f"Output/{key}", "hand_label": "Tay thuận", "tips": GENERIC_TIPS,
                              "actions": {k: {"label": lab, "short": lab, "hand": False} for k, lab in actions}}
        save_registry(reg, root)
    except Exception:
        for p in reversed(created):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        raise
    return get_sport(key, root)


def add_action(sport_key: str, action_key: str, label: str, hand: bool = False, root: Path = ROOT) -> Sport:
    """Thêm bài tập mới vào môn đã có (đăng ký + thêm khối mẫu vào config/<môn>.yaml nếu file có mục `actions:`)."""
    action_key = _slug_ok(action_key, "Mã bài tập")
    reg = load_registry(root)
    if sport_key not in reg["sports"]:
        raise SportError(f"Không có môn «{sport_key}».")
    acts = reg["sports"][sport_key].setdefault("actions", {})
    if action_key in acts:
        raise SportError(f"Bài tập «{action_key}» đã có trong môn này.")
    label = (label or "").strip() or action_key.replace("_", " ").capitalize()
    acts[action_key] = {"label": label, "short": label, "hand": bool(hand)}
    cfg = Path(root) / reg["sports"][sport_key].get("config", f"config/{sport_key}.yaml")
    if cfg.is_file():
        lines = cfg.read_text(encoding="utf-8").split("\n")
        for i, ln in enumerate(lines):
            if re.match(r"^actions:\s*(\{\s*\})?\s*(#.*)?$", ln) and "{}" not in ln:
                lines[i + 1:i + 1] = [f"  {action_key}:", f"    label: \"{label}\"", "    tolerance_floor_deg: 10.0      # dung sai tối thiểu cho mỗi góc (độ)", "    min_segment_s: 0.5             # đoạn chuyển động tối thiểu để chấm (giây)"]
                cfg.write_text("\n".join(lines), encoding="utf-8")
                break
    save_registry(reg, root)
    return get_sport(sport_key, root)
