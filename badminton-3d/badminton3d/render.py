"""Video overlay: làm rõ ảnh -> khung xương xanh/đỏ theo điểm -> bảng dashboard + icon phản hồi.
Chữ tiếng Việt có dấu vẽ bằng PIL (cv2.putText không hỗ trợ dấu).

Cấu trúc:
  enhance_frame      xử lý ảnh (tương phản, sáng, làm nét)
  build_frame_eval   gom kết quả chấm điểm thành "đánh giá theo từng frame"
  _skeleton          vẽ khung xương (màu theo điểm)
  draw_panel         bảng dashboard góc trái trên (nền mờ + chữ)
  draw_status_icon   vòng tròn + tick / X góc phải dưới
  draw_text_pil      chữ tiếng Việt (PIL)
  render_overlay     vòng lặp video
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from badminton3d.analyze import Context
from badminton3d.pose_io import JOINTS
from badminton3d.scoring import severity
from badminton3d.segment import READY_STATES

# ------------------------------------------------------------------ hằng số
EDGES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]

# Màu BGR cho OpenCV; RGB cho PIL
GREEN_BGR, RED_BGR, NEUTRAL_BGR, DARK_BGR = (0, 255, 0), (0, 0, 255), (235, 200, 120), (20, 20, 20)
GREEN, RED, WHITE, GREY, ORANGE = (0, 255, 0), (255, 60, 60), (255, 255, 255), (190, 190, 190), (255, 170, 0)
LIGHT_GREEN, LIGHT_RED = (120, 255, 120), (255, 120, 120)

PASS_SCORE = 70.0      # >= 70: form đạt (xanh); < 70: sai (đỏ)

font_path = "arial.ttf"        # đổi thành "Roboto.ttf" nếu muốn; không tìm thấy sẽ tự dò font khác có tiếng Việt
FONT_FALLBACKS = [
    "arial.ttf", "Arial.ttf", "Roboto-Regular.ttf", "Roboto.ttf",
    "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
BOLD_FALLBACKS = [
    "arialbd.ttf", "Arial Bold.ttf", "Roboto-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]

# Nhận xét ngắn khi sai: (đặc trưng, hướng lệch) -> câu hiển thị trên video
SHORT_ERR = {
    ("trunk_lean", "high"): "Đổ người về trước!", ("trunk_lean", "low"): "Thân quá thẳng, đổ nhẹ ra trước!",
    ("knee_mean", "high"): "Gối duỗi quá, hạ trọng tâm!", ("knee_mean", "low"): "Gối gập quá sâu, nâng người lên!",
    ("hip_mean", "high"): "Người quá thẳng, gập hông thêm!", ("hip_mean", "low"): "Gập hông quá nhiều, đứng cao hơn!",
    ("stance_ratio", "high"): "Hai chân mở quá rộng!", ("stance_ratio", "low"): "Hai chân quá hẹp, mở rộng ra!",
    ("elbow_mean", "high"): "Khuỷu tay duỗi quá, gập lại!", ("elbow_mean", "low"): "Khuỷu tay gập quá, duỗi bớt!",
    ("shoulder_mean", "high"): "Tay dang quá xa thân!", ("shoulder_mean", "low"): "Tay áp sát thân, đưa vợt ra trước!",
    ("elbow_arm", "high"): "Khuỷu tay duỗi quá, gập lại!", ("elbow_arm", "low"): "Khuỷu tay gập quá, duỗi bớt!",
    ("shoulder_arm", "high"): "Tay nâng quá cao, hạ xuống!", ("shoulder_arm", "low"): "Tay áp sát thân, đưa ra trước!",
    ("wrist_height", "high"): "Tay quá cao, hạ xuống!", ("wrist_height", "low"): "Tay quá thấp, nâng lên!",
    ("wrist_reach", "high"): "Tay đưa quá xa thân!", ("wrist_reach", "low"): "Tay quá sát thân, đưa ra trước!",
    ("follow_through", "high"): "Vung theo quá cao, dừng vợt sớm!", ("heel_lift", "high"): "Nhấc gót chân khi phát!",
}
PHASE_VN = {"chuan_bi": "Chuẩn bị", "rut_vot": "Rút vợt", "danh_cau": "Chạm cầu", "ket_thuc": "Kết thúc"}



# ------------------------------------------------------------------ 1. xử lý ảnh
def enhance_frame(img: np.ndarray, alpha: float = 1.15, beta: int = 15, sharpen: float = 0.6) -> np.ndarray:
    """Tăng tương phản/độ sáng nhẹ rồi làm nét (unsharp mask) để người chơi nổi khỏi nền."""
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)
    return out


# ------------------------------------------------------------------ 2. đánh giá theo frame
@dataclass
class FrameEval:
    score: float
    label: str = ""                                   # vd. "Chạm cầu" (serve)
    rows: Dict[str, dict] = field(default_factory=dict)   # đặc trưng -> {status, center, tol, unit, ...}

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    def worst_error(self) -> Optional[dict]:
        bad = [r for r in self.rows.values() if r.get("status") in ("HIGH", "LOW") and r.get("score") is not None]
        if not bad:
            return None
        special = [r for r in bad if r.get("feature") in ("follow_through", "heel_lift")]        # lỗi riêng đang xảy ra ở đúng frame này được nêu trước
        return special[0] if special else min(bad, key=lambda r: (r["score"], 0 if r.get("level") == "PRIMARY" else 1))


def build_frame_eval(sc: Optional[dict]) -> Dict[int, FrameEval]:
    """Gán đánh giá (điểm, từng góc đúng/sai) cho mọi frame được chấm. Frame không chấm -> không có khoá."""
    ev: Dict[int, FrameEval] = {}
    if not sc:
        return ev
    for u in sc["units"]:
        a, b = u["frames"]
        if "phases" in u:                              # serve: điểm theo pha, thiếu thì dùng điểm cả lần phát
            if u.get("score") is None:
                continue
            for i in range(a, b + 1):
                ev[i] = FrameEval(u["score"])
            for ph, p in u["phases"].items():
                if p.get("score") is None:
                    continue
                fa, fb = p["frames"]
                rows = {r["feature"]: r for r in p["features"]}
                for i in range(fa, fb + 1):
                    ev[i] = FrameEval(p["score"], PHASE_VN.get(ph, ph), rows)
            for f in u.get("faults", []):                  # lỗi riêng (vung xả đà, nhấc gót): hiện trên đúng các frame xảy ra lỗi
                row = {"feature": f["feature"], "status": "HIGH", "dev": f["dev"], "score": 0.0, "level": "PRIMARY", "center": f["center"], "tol": f["tol"], "unit": f["unit"], "severity": f["severity"]}
                for i in range(f["frames"][0], f["frames"][1] + 1):
                    if i in ev:
                        ev[i].rows[f["feature"]] = row
            for i in range(a, b + 1):                      # điểm trên video = điểm lần phát đã trừ lỗi riêng
                if i in ev and u.get("penalty"):
                    ev[i].score = min(ev[i].score, u["score"])
        elif u.get("score") is not None:               # ready
            rows = {r["feature"]: r for r in u["features"]}
            for i in range(a, b + 1):
                ev[i] = FrameEval(u["score"], "", rows)
    return ev


def verdict_text(fe: FrameEval) -> Tuple[str, Tuple[int, int, int]]:
    """Nhận xét + màu chữ, luôn khớp màu khung xương (đạt = xanh, sai = đỏ)."""
    err = fe.worst_error()
    short = SHORT_ERR.get((err["feature"], "high" if err["dev"] > 0 else "low")) if err else None
    sev = f" ({err.get('severity') or severity(err['score'])})" if err else ""
    if fe.passed:
        if not short:                                  # đạt và không góc nào lệch -> hoàn hảo
            return "HOÀN HẢO! Giữ nguyên tư thế", GREEN
        return f"ĐẠT! Chú ý{sev}: {short}", LIGHT_GREEN     # đạt nhưng còn góc lệch nhẹ -> vẫn xanh, kèm lưu ý
    return f"SAI: {short or 'Tư thế lệch chuẩn!'}", RED


# ------------------------------------------------------------------ font + chữ tiếng Việt
@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    cands = ([font_path.replace(".ttf", "bd.ttf")] if bold else [font_path]) + (BOLD_FALLBACKS if bold else FONT_FALLBACKS)
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_texts_pil(img: np.ndarray, items: Sequence[Tuple[str, Tuple[int, int], int, Tuple[int, int, int], bool]]) -> np.ndarray:
    """Vẽ nhiều dòng chữ chỉ với MỘT lần đổi BGR<->PIL (nhanh hơn gọi từng dòng). item = (text, (x,y), size, màu RGB, in_đậm)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, color, bold in items:
        d.text(xy, text, font=_font(size, bold), fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_text_pil(img: np.ndarray, text: str, xy: Tuple[int, int], size: int = 18, color: Tuple[int, int, int] = WHITE, bold: bool = False) -> np.ndarray:
    """Vẽ một dòng chữ tiếng Việt có dấu (color là RGB). Trả về ảnh BGR mới."""
    return draw_texts_pil(img, [(text, xy, size, color, bold)])


# ------------------------------------------------------------------ 3. khung xương (đoạn lệch = đỏ, còn lại xanh/cam)
YELLOW_BGR, AMBER_BGR = (0, 235, 255), (0, 165, 255)
# đặc trưng -> các đoạn xương liên quan ("{s}" = cả hai bên, "{a}" = tay đang đo ở bài serve)
FEATURE_BONES = {
    "heel_lift": [("knee", "ankle")], "knee_mean": [("hip", "knee"), ("knee", "ankle")], "hip_mean": [("shoulder", "hip"), ("hip", "knee")],
    "trunk_lean": [("shoulder", "hip")], "stance_ratio": [("knee", "ankle")],
    "elbow_mean": [("shoulder", "elbow"), ("elbow", "wrist")], "shoulder_mean": [("hip", "shoulder"), ("shoulder", "elbow")],
}
ARM_FEATURES = {"elbow_arm": [("shoulder", "elbow"), ("elbow", "wrist")], "shoulder_arm": [("hip", "shoulder"), ("shoulder", "elbow")],
                "follow_through": [("shoulder", "elbow"), ("elbow", "wrist")], "wrist_height": [("shoulder", "elbow"), ("elbow", "wrist")], "wrist_reach": [("shoulder", "elbow"), ("elbow", "wrist")]}


def faulty_bones(ctx: Context, fe: Optional[FrameEval]) -> set:
    """Tập cạnh xương (a, b) thuộc góc đang lệch khỏi chuẩn."""
    bad = set()
    if fe is None:
        return bad
    side = (ctx.arm or {}).get("side")
    for name, r in fe.rows.items():
        if r.get("status") not in ("HIGH", "LOW"):
            continue
        if name in ARM_FEATURES and side:
            bad |= {(f"{side}_{x}", f"{side}_{y}") for x, y in ARM_FEATURES[name]}
        for x, y in FEATURE_BONES.get(name, []):
            bad |= {(f"{sd}_{x}", f"{sd}_{y}") for sd in ("left", "right")}
    return bad


def _skeleton(img: np.ndarray, ctx: Context, i: int, scale: float, fe: Optional[FrameEval]) -> None:
    """Đoạn xương thuộc góc lệch = ĐỎ; đoạn còn lại = XANH LÁ khi form đạt (score >= 70), CAM khi form sai hoặc chưa chấm.
    Khớp là chấm vàng có viền tối."""
    base = AMBER_BGR if fe is None or not fe.passed else GREEN_BGR
    bad = faulty_bones(ctx, fe)
    thick = max(3, round(min(img.shape[:2]) / 150))
    xy = ctx.clean.xy
    pts = {j: (int(xy[j][i][0] * scale), int(xy[j][i][1] * scale)) for j in JOINTS if not np.isnan(xy[j][i]).any()}
    for a, b in EDGES:
        if a in pts and b in pts:
            col = RED_BGR if ((a, b) in bad or (b, a) in bad) else base
            cv2.line(img, pts[a], pts[b], DARK_BGR, thick + 2, cv2.LINE_AA)
            cv2.line(img, pts[a], pts[b], col, thick, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(img, p, thick + 3, DARK_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, p, thick + 1, YELLOW_BGR, -1, cv2.LINE_AA)


# ------------------------------------------------------------------ 4. dashboard
YELLOW_TXT, ORANGE_TXT = (255, 235, 90), (255, 110, 60)


def _range(r: dict) -> str:
    lo, hi, u = r["center"] - r["tol"], r["center"] + r["tol"], r["unit"]
    if u == "x":
        return f"{lo * 100:.0f}-{hi * 100:.0f}%"
    if u == "m":
        return f"{lo * 100:.0f}-{hi * 100:.0f} cm"
    return f"{lo:.0f}-{hi:.0f}°"


def _fmt_val(v: float, unit: str) -> str:
    if np.isnan(v):
        return "--"
    return f"{v * 100:.0f}%" if unit == "x" else (f"{v * 100:.0f} cm" if unit == "m" else f"{v:.0f}°")


def _num_line(ctx: Context, i: int, name: str, label: str, fe: Optional[FrameEval]):
    f = ctx.feats.get(name)
    v = float(f["value"][i]) if f is not None and i < len(f["value"]) else float("nan")
    row = fe.rows.get(name) if fe else None
    unit = row["unit"] if row else ("x" if name == "stance_ratio" else "°")
    txt = f"{label}: {_fmt_val(v, unit)}"
    if row and row.get("status") in ("OK", "HIGH", "LOW"):
        txt += f" (Chuẩn: {_range(row)})"
        return txt, (YELLOW_TXT if row["status"] == "OK" else ORANGE_TXT)
    return txt, YELLOW_TXT


def _state_line(label: str, names: Sequence[str], fe: Optional[FrameEval]):
    """Dòng trạng thái (Đạt / Quá cao / Quá thấp...) gộp từ một hay nhiều góc."""
    rows = [fe.rows[n] for n in names if fe and n in fe.rows and fe.rows[n].get("status") in ("OK", "HIGH", "LOW")]
    if not rows:
        return f"{label}: --", YELLOW_TXT
    bad = [r for r in rows if r["status"] != "OK"]
    if not bad:
        return f"{label}: Đạt", YELLOW_TXT
    r = max(bad, key=lambda r: abs(r["dev"]) / max(r["tol"], 1e-6))
    return f"{label}: {'Quá cao' if r['dev'] > 0 else 'Quá thấp'}", ORANGE_TXT


def panel_lines(ctx: Context, i: int, fe: Optional[FrameEval]) -> List[Tuple[str, Tuple[int, int, int]]]:
    if ctx.exercise == "ready":
        return [_num_line(ctx, i, "knee_mean", "Góc gối", fe), _num_line(ctx, i, "trunk_lean", "Độ nghiêng lưng", fe),
                _num_line(ctx, i, "stance_ratio", "Độ rộng chân", fe), _state_line("Vị trí tay", ["elbow_mean", "shoulder_mean"], fe)]
    return [_num_line(ctx, i, "elbow_arm", "Góc khuỷu tay", fe), _state_line("Độ cao cổ tay", ["wrist_height"], fe),
            _num_line(ctx, i, "trunk_lean", "Độ nghiêng lưng", fe), _num_line(ctx, i, "knee_mean", "Góc gối", fe)]


def _shade(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float, border: bool = False) -> None:
    """Nền đen bán trong suốt (cv2.addWeighted) trên vùng [x0:x1, y0:y1]."""
    h, w = img.shape[:2]
    x1, y1 = min(x1, w), min(y1, h)
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0)
    if border:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (150, 150, 150), 1)


def draw_panel(img: np.ndarray, ctx: Context, i: int, fe: Optional[FrameEval], idle_text: str) -> np.ndarray:
    """Góc trái trên: khối tiêu đề (FORM CHUẨN + nhận xét) và hộp thông số có khoảng chuẩn, nền bán trong suốt."""
    h, w = img.shape[:2]
    s = min(h, w) / 540.0                              # cỡ chữ theo CẠNH NGẮN (video dọc 540x960 và ngang 960x540 có cùng cỡ chữ)
    f_big, f_sub, f_row = int(28 * s), int(15 * s), int(17 * s)
    pad, gap = int(10 * s), int(27 * s)
    if fe is not None:
        sub, subcol, head = verdict_text(fe)[0], verdict_text(fe)[1], f"FORM CHUẨN: {fe.score:.0f}"
    else:
        sub, subcol, head = idle_text, ORANGE, "FORM CHUẨN: --"
    rows = panel_lines(ctx, i, fe)
    wid = lambda t, size, bold: _font(size, bold).getlength(t)
    # --- khối tiêu đề
    hw = int(max(wid(head, f_big, False), wid(sub, f_sub, True)) + 2 * pad)
    hh = pad + int(f_big * 1.15) + int(f_sub * 1.3) + pad // 2
    _shade(img, 0, 0, hw, hh, 0.65)
    # --- hộp thông số
    bw = int(max(wid(t, f_row, False) for t, _ in rows) + 2 * pad)
    bh = pad + gap * len(rows)
    by = hh + int(6 * s)
    _shade(img, 4, by, 4 + bw, by + bh, 0.55, border=True)
    items = [(head, (pad, pad // 2), f_big, WHITE, False), (sub, (pad, pad // 2 + int(f_big * 1.15)), f_sub, subcol, True)]
    for k, (t, c) in enumerate(rows):
        items.append((t, (4 + pad, by + pad // 2 + gap * k), f_row, c, False))
    return draw_texts_pil(img, items)


# ------------------------------------------------------------------ 5. icon phản hồi
def draw_status_icon(img: np.ndarray, passed: bool) -> None:
    """Góc phải dưới: vòng tròn trắng + tick xanh (đạt) hoặc vòng tròn đỏ + X trắng (sai)."""
    h, w = img.shape[:2]
    sh = min(h, w)
    r = int(sh * 0.09)
    cx, cy = w - r - int(sh * 0.05), h - r - int(sh * 0.07)
    th = max(5, r // 5)
    cv2.circle(img, (cx, cy), r + 2, DARK_BGR, -1, cv2.LINE_AA)
    if passed:
        cv2.circle(img, (cx, cy), r, (245, 245, 245), -1, cv2.LINE_AA)
        pts = np.array([[cx - 0.46 * r, cy + 0.02 * r], [cx - 0.12 * r, cy + 0.38 * r], [cx + 0.48 * r, cy - 0.36 * r]], np.int32)
        cv2.polylines(img, [pts], False, (90, 180, 30), th, cv2.LINE_AA)       # xanh lá đậm (BGR)
    else:
        cv2.circle(img, (cx, cy), r, (50, 50, 235), -1, cv2.LINE_AA)
        d = int(0.38 * r)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), (255, 255, 255), th, cv2.LINE_AA)
        cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), (255, 255, 255), th, cv2.LINE_AA)


# ------------------------------------------------------------------ trạng thái khi chưa chấm
def _idle_reason(ctx: Context, i: int) -> str:
    if i >= ctx.pose.n:
        return ""
    if ctx.exercise == "ready":
        inv = {v: k for k, v in READY_STATES.items()}
        return {"MOVING": "Đang di chuyển đỡ cầu", "SETTLING": "Đang ổn định lại tư thế...", "INVALID": "Thiếu dữ liệu khớp", "READY": "Chuẩn bị chấm..."}[inv[ctx.ready_seg.state[i]]]
    return serve_idle_label(ctx, i)


def _tag_text(ctx: Context, i: int, fe: Optional[FrameEval]) -> str:
    base = "SẴN SÀNG" if ctx.exercise == "ready" else "PHÁT CẦU"
    return f"{base} · {fe.label}" if fe is not None and fe.label else base


# ------------------------------------------------------------------ vòng lặp video
def render_overlay(video_path: str, ctx: Context, res: dict, cfg: dict, out_path: str, max_height: int = 540) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or ctx.pose.fps
    sc = res.get("score")
    overall = sc["overall"] if sc else None
    evals = build_frame_eval(sc)
    writer, i = None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h0, w0 = fr.shape[:2]
        scale = min(1.0, max_height / min(h0, w0))      # cạnh NGẮN = max_height (540): video dọc 1080x1920 -> 540x960, ngang 1920x1080 -> 960x540
        if scale < 1.0:                                  # đổi cỡ TRƯỚC khi vẽ để chữ/đường vẽ sắc nét ở độ phân giải xuất
            fr = cv2.resize(fr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        fr = enhance_frame(fr)
        fe = evals.get(i)
        if i < ctx.pose.n and ctx.pose.detected[i]:
            _skeleton(fr, ctx, i, scale, fe)
        fr = draw_panel(fr, ctx, i, fe, _idle_reason(ctx, i))
        if fe is not None:
            draw_status_icon(fr, fe.passed)
        h, w = fr.shape[:2]
        k = min(h, w) / 540.0
        if overall is not None:
            strip = fr[h - int(36 * k):, :int(230 * k)]
            strip[:] = cv2.addWeighted(strip, 0.35, np.zeros_like(strip), 0.65, 0)
        extra = [(_tag_text(ctx, i, fe), (w - int(210 * k), 12), int(15 * k), WHITE, True)]
        if overall is not None:
            extra.append((f"TỔNG: {overall:.0f}/100 ({sc['grade']})", (10, h - int(30 * k)), int(17 * k), WHITE, True))
        fr = draw_texts_pil(fr, extra)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fr.shape[1], fr.shape[0]))
        writer.write(fr)
        i += 1
    cap.release()
    if writer:
        writer.release()


def serve_idle_label(ctx: Context, i: int) -> str:
    """Frame phát cầu không được chấm: phân loại để người xem biết hệ thống đang thấy gì (chỉ nhãn, không chấm điểm)."""
    from badminton3d.signals import body_speed
    cache = getattr(ctx, "_idle_cache", None)
    if cache is None:
        cache = body_speed(ctx.clean, ctx.cfg_serve_speed_window if hasattr(ctx, "cfg_serve_speed_window") else 0.2)
        ctx._idle_cache = cache
    reps = ctx.serve_seg.reps
    if reps and i > reps[-1].end:
        return "Sau cú phát (không chấm)"
    sp = cache[i] if i < len(cache) else float("nan")
    if np.isnan(sp):
        return "Thiếu dữ liệu (không chấm)"
    return "Đang di chuyển / nhặt cầu" if sp >= 0.30 else "Đang đứng chuẩn bị / sẵn sàng phát"
