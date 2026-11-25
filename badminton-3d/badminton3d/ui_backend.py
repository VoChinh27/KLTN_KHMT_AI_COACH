"""Lớp nối giữa giao diện (app_ui.py, Streamlit) và các engine chấm điểm, theo từng MÔN THỂ THAO.

Không import streamlit ở đây để kiểm thử được bằng unittest. Mọi hàm nhận một đối tượng `Sport` (badminton3d/sports.py) nên giao diện không biết tên
bài tập, thư mục hay file cấu hình cụ thể nào: tất cả lấy từ config/sports.yaml. Việc nặng (quay camera, dựng chuẩn, chạy toàn bộ) do engine của môn thực hiện.
"""
from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from badminton3d.sports import Sport, SportError  # noqa: F401  (SportError được giao diện dùng lại)

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
HAND_VN = {"auto": "Tự động (auto)", "right": "Tay phải (right)", "left": "Tay trái (left)"}
ROLE_VN = {"trainee": "Người tập (trainee)", "coach": "Huấn luyện viên (coach)"}
GRADE_COLOR = {"Tốt": "#0E9F6E", "Khá": "#2F80ED", "Cần cải thiện": "#F2994A", "Chưa đạt": "#EB5757"}
_NOISE = ("W0000", "I0000", "INFO:", "WARNING:", "absl", "inference_feedback_manager", "landmark_projection", "TensorFlow Lite", "XNNPACK")


class UIError(Exception):
    """Lỗi có thông điệp tiếng Việt thân thiện để hiện thẳng cho người dùng."""


class ReferenceMissing(UIError):
    pass


def _engine(sport: Sport, action: Optional[str] = None):
    """Engine của môn; báo lỗi thân thiện nếu môn chưa có bộ chấm điểm hoặc engine chưa hỗ trợ bài này."""
    try:
        e = sport.engine
    except SportError as exc:
        raise UIError(str(exc))
    if e is None:
        raise UIError(f"Môn «{sport.name}» chưa có bộ chấm điểm (engine). Bạn vẫn quản lý được video, chuẩn và cấu hình; để chấm điểm hãy gắn engine "
                      f"theo hướng dẫn trong docs/MULTISPORT.md (khai báo `engine:` của môn trong config/sports.yaml).")
    if action and action not in getattr(e, "supported_actions", ()):
        raise UIError(f"Engine «{getattr(e, 'name', sport.engine_path)}» chưa hỗ trợ bài «{sport.label(action)}».")
    return e


# ---------------------------------------------------------------- video đầu vào
def list_input_videos(sport: Sport) -> List[dict]:
    """Video trong thư mục Input của môn (không đệ quy: bỏ qua _luu_tru và thư mục môn khác)."""
    out = []
    if not sport.input.is_dir():
        return out
    for p in sorted(sport.input.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."):
            m = re.match(r"(coach|trainee)_", p.name, re.I)
            out.append({"stem": p.stem, "name": p.name, "path": p, "action": sport.action_of(p.stem), "role": m.group(1).lower() if m else "unknown",
                        "analyzed": (sport.output / p.stem / "report.json").is_file()})
    return out


def video_label(sport: Sport, v: dict) -> str:
    return f"{v['name']}  ·  {sport.short(v['action'])}  ·  {'đã chấm' if v['analyzed'] else 'chưa chấm'}"


def unique_stem(role: str, action: str, tag: str = "up", input_dir: Path = Path("Input")) -> str:
    """Tên mới không trùng, đúng quy ước <role>_<bài>_<id>: trainee_<bài>_up01, up02..."""
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for f in Path(input_dir).iterdir() if (m := re.match(rf"{role}_{re.escape(action)}_{tag}(\d+)$", f.stem))]
    return f"{role}_{action}_{tag}{(max(used) + 1 if used else 1):02d}"


def save_upload(sport: Sport, data: bytes, original_name: str, role: str, action: str) -> Path:
    """Lưu video người dùng tải lên vào Input của môn với tên chuẩn (giữ đuôi file gốc)."""
    ext = Path(original_name).suffix.lower()
    if ext not in VIDEO_EXT:
        raise UIError(f"Định dạng '{ext or '(không có)'}' chưa được hỗ trợ. Hãy dùng: {', '.join(VIDEO_EXT)}.")
    if role not in ROLE_VN or action not in sport.actions:
        raise UIError("Vai trò hoặc bài tập không hợp lệ.")
    if not data:
        raise UIError("File tải lên rỗng.")
    path = sport.input / (unique_stem(role, action, "up", sport.input) + ext)
    path.write_bytes(data)
    return path


def read_annotation(stem: str, input_dir: Path) -> dict:
    p = Path(input_dir) / f"{stem}.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def write_hand(stem: str, hand: str, input_dir: Path) -> None:
    """Ghi racket_hand vào <Input>/<video>.json (giữ nguyên các khóa khác như person_roi, serve_windows_s). 'auto' = bỏ khai báo."""
    if hand not in HAND_VN:
        raise UIError("Tay thuận phải là auto, right hoặc left.")
    ann = read_annotation(stem, input_dir)
    if hand == "auto":
        ann.pop("racket_hand", None)
    else:
        ann["racket_hand"] = hand
    p = Path(input_dir) / f"{stem}.json"
    if ann or p.exists():
        p.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- chạy lõi
def clean_log(text: str) -> str:
    return "\n".join(l for l in (text or "").splitlines() if l.strip() and not any(n in l for n in _NOISE))


def run_cli(args: Sequence[str], timeout: Optional[int] = None) -> Tuple[int, str]:
    """Chạy `python -m badminton3d <args>` tại thư mục dự án; trả (mã thoát, log đã lọc nhiễu của thư viện)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    try:
        p = subprocess.run([sys.executable, "-m", "badminton3d", *map(str, args)], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise UIError("Quá thời gian chờ. Hãy thử lại với video ngắn hơn hoặc chạy bằng dòng lệnh.")
    return p.returncode, clean_log((p.stdout or "") + "\n" + (p.stderr or ""))


def reference_exists(sport: Sport, action: str) -> bool:
    return sport.reference_file(action).is_file()


def analyze_video(sport: Sport, video_path: Path, hand: str = "auto", force_pose: bool = False) -> dict:
    """Chấm một video bằng engine của môn. Trả về nội dung report (cùng cấu trúc report.json)."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise UIError(f"Không thấy video: {video_path.name}")
    action = sport.action_of(video_path.stem)
    if action is None:
        keys = ", ".join(sport.actions) or "(chưa có bài nào)"
        raise UIError(f"Tên video phải chứa mã bài tập của môn {sport.name} ({keys}), ví dụ trainee_{next(iter(sport.actions), 'bai')}_01.mp4.")
    eng = _engine(sport, action)
    if not reference_exists(sport, action):
        raise ReferenceMissing(f"Chưa có chuẩn HLV cho bài «{sport.label(action)}» (file {sport.reference_file(action).name} trong thư mục chuẩn của môn). "
                               f"Hãy dựng chuẩn ở tab «Chuẩn HLV & Cấu hình».")
    if sport.actions[action].hand:
        write_hand(video_path.stem, hand, video_path.parent)
    try:
        rep = eng.analyze(sport, video_path, action, hand, force_pose)
    except (IOError, OSError) as exc:
        raise UIError(f"Không đọc được video (codec lạ?). Thử chuyển sang MP4 H.264. Chi tiết: {exc}")
    rep = rep or load_report(sport, video_path.stem)
    if not rep:
        raise UIError("Chạy xong nhưng không thấy báo cáo kết quả; xem log trong cửa sổ dòng lệnh.")
    return rep


def load_report(sport: Sport, stem: str) -> Optional[dict]:
    p = sport.output / stem / "report.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def report_files(sport: Sport, stem: str) -> Dict[str, Optional[Path]]:
    d = sport.output / stem
    names = {"overlay": "overlay.mp4", "dashboard": "dashboard.png", "timeline": "angles_timeline.png", "quality": "quality.png", "html": "report.html", "json": "report.json"}
    return {k: (d / v if (d / v).is_file() else None) for k, v in names.items()}


def fmt_val(v: Optional[float], unit: str) -> str:
    if v is None or v != v:
        return "—"
    if unit == "°":
        return f"{v:.0f}°"
    if unit == "m":
        return f"{v * 100:.0f} cm"
    if unit == "x":
        return f"{v * 100:.0f}% vai"
    return f"{v:.2f}"


def summary(rep: dict) -> dict:
    sc = rep.get("score") or {}
    return {"status": rep.get("status"), "overall": sc.get("overall"), "grade": sc.get("grade"), "warnings": rep.get("warnings", []),
            "message": rep.get("message"), "action": rep.get("exercise") or rep.get("action"), "duration": rep.get("duration_s")}


def error_rows(rep: dict) -> List[dict]:
    """Bảng «Cần sửa gì để tăng điểm»: góc sai, giá trị thực vs chuẩn HLV, hướng dẫn sửa, điểm lấy lại được."""
    rows = []
    for e in (rep.get("score") or {}).get("errors", []):
        one_sided = e.get("feature") in ("follow_through", "heel_lift") or e.get("kind") in ("PHẠM LUẬT", "LỖI HIỆU QUẢ")
        std = f"≤ {fmt_val(e['tol'], e['unit'])}" if one_sided else f"{fmt_val(e['center'], e['unit'])} ± {fmt_val(e['tol'], e['unit'])}"
        rows.append({"Mức": e.get("severity", ""), "Phần": e.get("where", ""), "Lỗi": e.get("label", ""), "Hiện tại": fmt_val(e.get("value"), e["unit"]),
                     "Chuẩn HLV": std, "Hướng dẫn sửa": e.get("advice", ""), "Điểm lấy lại được": round(float(e.get("potential_gain", 0.0)), 1)})
    return rows


# ---------------------------------------------------------------- video xem được trên trình duyệt
def _ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def web_video(src: Optional[Path]) -> Optional[Path]:
    """overlay.mp4 ghi bằng mp4v mà trình duyệt thường không phát được. Chuyển sang H.264 (hoặc VP8/WebM) bằng ffmpeg, lưu cạnh file gốc
    (overlay_web.mp4); dùng lại nếu đã có và mới hơn. Không có ffmpeg thì trả None (giao diện hiện khung hình chính + nút tải)."""
    if src is None or not Path(src).is_file():
        return None
    src = Path(src)
    dst = src.with_name(src.stem + "_web.mp4")
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size > 0:
        return dst
    ff = _ffmpeg()
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(dst)]
        try:
            if subprocess.run(cmd, capture_output=True, timeout=600).returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
                return dst
        except (subprocess.TimeoutExpired, OSError):
            pass
        if dst.exists():
            dst.unlink()
        webm = src.with_name(src.stem + "_web.webm")
        try:
            if subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libvpx", "-b:v", "1500k", str(webm)], capture_output=True, timeout=600).returncode == 0 and webm.is_file():
                return webm
        except (subprocess.TimeoutExpired, OSError):
            pass
    return None


def key_frames(video: Optional[Path], n: int = 6) -> list:
    """n khung hình (RGB) rải đều video, dùng khi không phát được video trong trình duyệt."""
    if video is None or not Path(video).is_file():
        return []
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---------------------------------------------------------------- nhúng HTML
def inline_images(html_path: Optional[Path]) -> str:
    """Đọc file HTML và nhúng ảnh tương đối thành data URI để hiển thị được trong iframe của Streamlit."""
    if html_path is None or not Path(html_path).is_file():
        return ""
    html_path = Path(html_path)
    text = html_path.read_text(encoding="utf-8", errors="replace")

    def sub(m):
        src = m.group(2)
        if re.match(r"^(https?:|data:|//)", src):
            return m.group(0)
        f = (html_path.parent / src).resolve()
        if not f.is_file():
            return m.group(0)
        mime = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        return f"{m.group(1)}data:{mime};base64,{base64.b64encode(f.read_bytes()).decode('ascii')}{m.group(3)}"

    return re.sub(r"""(<img[^>]*?\ssrc=["'])([^"']+)(["'])""", sub, text, flags=re.I)


# ---------------------------------------------------------------- chuẩn HLV
def reference_rows(sport: Sport, action: str) -> Tuple[Optional[dict], List[dict]]:
    """(thông tin chung, các dòng chuẩn ± dung sai). Do engine của môn diễn giải file chuẩn; môn chưa có engine thì chỉ báo file có/không."""
    p = sport.reference_file(action)
    if not p.is_file():
        return None, []
    e = None
    try:
        e = sport.engine
    except SportError:
        pass
    if e is not None and hasattr(e, "reference_rows"):
        return e.reference_rows(sport, action)
    try:
        ref = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        raise UIError(f"File chuẩn {p.name} bị hỏng (không phải JSON hợp lệ).")
    return {"built_from": ref.get("built_from", []), "n_videos": ref.get("n_videos", 0), "built_on": ref.get("built_on", "")}, []


def coach_videos(sport: Sport, action: str) -> List[dict]:
    return [v for v in list_input_videos(sport) if v["role"] == "coach" and v["action"] == action]


def build_reference(sport: Sport, action: str, videos: Sequence[Path]) -> Tuple[int, str]:
    if action not in sport.actions:
        raise UIError("Bài tập không hợp lệ.")
    if not 1 <= len(videos) <= 3:
        raise UIError("Hãy chọn từ 1 đến 3 video HLV tập đúng.")
    for v in videos:
        if not Path(v).is_file():
            raise UIError(f"Không thấy video: {Path(v).name}")
        if sport.action_of(Path(v).name) != action:
            raise UIError(f"Video «{Path(v).name}» không phải bài «{sport.label(action)}».")
    return _engine(sport, action).build_reference(sport, action, [Path(v) for v in videos])


# ---------------------------------------------------------------- quay trực tiếp và chạy toàn bộ
def record_live(sport: Sport, action: str, source: str, role: str = "trainee", hand: str = "auto", max_seconds: int = 180) -> Tuple[int, str, Optional[str]]:
    """Mở camera bằng engine của môn (cửa sổ trên máy đang chạy giao diện); trả (mã, log, tên video đã quay/chấm)."""
    if action not in sport.actions:
        raise UIError("Bài tập không hợp lệ.")
    if not str(source).strip():
        raise UIError("Hãy nhập nguồn camera (số webcam hoặc địa chỉ URL).")
    eng = _engine(sport, action)
    if not hasattr(eng, "record"):
        raise UIError(f"Engine của môn {sport.name} chưa hỗ trợ quay trực tiếp.")
    if not reference_exists(sport, action):
        raise ReferenceMissing(f"Chưa có chuẩn HLV cho bài «{sport.label(action)}». Hãy dựng chuẩn ở tab «Chuẩn HLV & Cấu hình» trước.")
    code, out = eng.record(sport, action, str(source).strip(), role, hand, int(max_seconds))
    return code, out, parse_record_stem(out)


def parse_record_stem(log: str) -> Optional[str]:
    m = re.findall(r"\[(?:SCORED|[A-Z_]+)\]\s+(\S+)\s+->", log or "")
    return m[-1] if m else None


def run_all(sport: Sport) -> Tuple[int, str]:
    eng = _engine(sport)
    if not hasattr(eng, "run_all"):
        raise UIError(f"Engine của môn {sport.name} chưa hỗ trợ chạy toàn bộ.")
    return eng.run_all(sport)


def evaluation_table(sport: Sport) -> list:
    p = sport.tong_hop / "evaluation.csv"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def summary_images(sport: Sport, limit: int = 8) -> List[Path]:
    """Các biểu đồ PNG trong thư mục tổng hợp của môn (không phụ thuộc tên cụ thể)."""
    d = sport.tong_hop
    return sorted(d.glob("*.png"))[:limit] if d.is_dir() else []


# ---------------------------------------------------------------- cấu hình YAML (giữ nguyên chú thích)
def _find_line(lines: List[str], keys: Sequence[str]) -> int:
    start, end, indent = 0, len(lines), -1
    idx = -1
    for depth, key in enumerate(keys):
        want = indent + 2 if depth else 0
        idx = -1
        for i in range(start, end):
            ln = lines[i]
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            cur = len(ln) - len(ln.lstrip())
            if depth and cur <= indent:
                end = i
                break
            if cur == want and re.match(rf"^\s*{re.escape(str(key))}\s*:", ln):
                idx = i
                break
        if idx < 0:
            raise KeyError(".".join(map(str, keys)))
        indent, start = want, idx + 1
        end = len(lines)
    return idx


def yaml_get(keys: Sequence[str], path: Path):
    import yaml
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in keys:
        node = node[k]
    return node


def yaml_set(keys: Sequence[str], value, path: Path) -> None:
    """Sửa một giá trị đơn trong YAML mà KHÔNG làm mất chú thích/định dạng (đổi đúng dòng đó)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    i = _find_line(lines, keys)
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    elif isinstance(value, float):
        text = repr(round(value, 6))
    elif isinstance(value, str):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    key = str(keys[-1])
    m = re.match(rf"^(\s*{re.escape(key)}\s*:\s*)([^#]*?)(\s*(?:#.*)?)$", lines[i])
    if not m:
        raise UIError(f"Không sửa được dòng cấu hình «{'.'.join(map(str, keys))}».")
    pad = m.group(3) if m.group(3).strip() else ""
    old = m.group(2)
    if m.group(3).startswith(" ") and len(text) < len(old):
        pad = " " * (len(old) - len(text)) + m.group(3)
    lines[i] = m.group(1) + text + pad
    path.write_text("\n".join(lines), encoding="utf-8")


def yaml_leaves(path: Path) -> List[dict]:
    """Các tham số đơn (bool / số / chuỗi / null) của file YAML: [{keys, value, comment}] — dùng cho trình sửa cấu hình chung của mọi môn."""
    import yaml
    path = Path(path)
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lines = path.read_text(encoding="utf-8").split("\n")
    out: List[dict] = []

    def walk(node, keys):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, keys + [str(k)])
        elif isinstance(node, (bool, int, float, str)) or node is None:
            try:
                ln = lines[_find_line(lines, keys)]
            except (KeyError, IndexError):
                return                                                    # nằm trong khối {...} một dòng: chỉ đọc, không sửa nhanh
            c = re.search(r"#\s*(.*)$", ln)
            out.append({"keys": keys, "value": node, "comment": c.group(1).strip() if c else ""})

    walk(data, [])
    return out


def save_config(changes: Dict[Tuple[str, ...], object], path: Path) -> None:
    """Áp nhiều thay đổi cùng lúc; kiểm tra file vẫn đọc được, nếu hỏng thì khôi phục bản sao lưu (.yaml.bak)."""
    path = Path(path)
    backup = path.with_suffix(".yaml.bak")
    shutil.copy2(path, backup)
    try:
        for keys, val in changes.items():
            yaml_set(keys, val, path)
        import yaml
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.name == "badminton3d.yaml":
            from badminton3d.config import load_config
            load_config(str(path))
    except Exception as exc:
        shutil.copy2(backup, path)
        raise UIError(f"Cấu hình không hợp lệ, đã khôi phục bản cũ. Chi tiết: {exc}")
