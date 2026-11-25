"""Ghi kết quả 1 video ra thư mục Output/<video>/: report.json, report.html (tự chứa), angles.csv, biểu đồ, overlay."""
from __future__ import annotations

import base64
import html
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from badminton3d import __version__
from badminton3d.analyze import Context
from badminton3d.segment import READY_STATES


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return to_py(o.item())
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def vn(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None or x != x else f"{x:.{nd}f}".replace(".", ",")


def fmt_val(x: Optional[float], unit: str) -> str:
    if x is None or x != x:
        return "—"
    if unit == "m":
        return f"{x * 100:.0f} cm".replace(".", ",")
    if unit == "°":
        return f"{x:.0f}°"
    return vn(x, 2)


def img_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def write_angles_csv(ctx: Context, path: str) -> None:
    p = ctx.pose
    df = pd.DataFrame({"frame": p.frames, "t_s": np.round(np.arange(p.n) / p.fps, 3)})
    if ctx.exercise == "ready":
        inv = {v: k for k, v in READY_STATES.items()}
        df["state"] = [inv[s] for s in ctx.ready_seg.state]
        df["body_speed_mps"] = np.round(ctx.ready_seg.speed, 3)
    else:
        lab = np.array(["NOT_SCORED"] * p.n, dtype=object)
        for r in ctx.serve_seg.reps:
            for k, (a, b) in r.phases.items():
                lab[a:b + 1] = k
        df["phase"] = lab
        df["arm_activity_mps"] = np.round(ctx.serve_seg.activity, 3)
    for n, f in ctx.feats.items():
        df[n] = np.round(f["value"], 3)
        df[n + "_rel"] = np.round(f["rel"], 2)
    df.to_csv(path, index=False)


def _status_cell(f: dict) -> str:
    s = f["status"]
    return {"OK": '<span class="ok">✓ Đạt</span>', "HIGH": '<span class="bad">▲ Cao hơn chuẩn</span>', "LOW": '<span class="bad">▼ Thấp hơn chuẩn</span>',
            "NOT_ASSESSED": '<span class="na">— không đánh giá</span>'}[s]


def _feature_table(rows: list) -> str:
    out = ['<table><tr><th>Góc</th><th>Đo được</th><th>Chuẩn HLV ± dung sai</th><th>Lệch</th><th>Điểm</th><th>Độ tin cậy</th><th>Kết quả</th></tr>']
    for f in rows:
        dev = "—" if f.get("dev") is None else (("+" if f["dev"] > 0 else "−") + fmt_val(abs(f["dev"]), f["unit"]))
        sub = ' <span class="tag">tay/bên thay thế</span>' if f.get("substituted", 0) > 0.5 else ""
        out.append(f"<tr><td>{html.escape(f['label'])}{sub}</td><td>{fmt_val(f.get('value'), f['unit'])}</td>"
                   f"<td>{fmt_val(f['center'], f['unit'])} ± {fmt_val(f['tol'], f['unit']).replace('+', '')}</td><td>{dev}</td>"
                   f"<td>{vn(f['score'], 0)}</td><td>{vn(f['rel'] * 100, 0)}%</td><td>{_status_cell(f)}</td></tr>")
    out.append("</table>")
    return "".join(out)


CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:2px solid #dde3e8;padding-bottom:6px}h3{margin-bottom:6px}.sub{color:#667}
.card{background:#fff;border-radius:10px;padding:18px 22px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin:14px 0}
.score{font-size:64px;font-weight:700;line-height:1}.good{color:#2e9e5b}.mid{color:#e8a33d}.low{color:#d64545}.grey{color:#9aa0a6}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid #e6eaee;text-align:left}th{background:#f0f3f6}
.ok{color:#2e9e5b;font-weight:600}.bad{color:#d64545;font-weight:600}.na{color:#999}.tag{background:#fff3d6;color:#8a5a00;border-radius:4px;padding:1px 6px;font-size:11px}
.warn{background:#fff7e0;border-left:4px solid #e8a33d;padding:10px 14px;margin:8px 0;border-radius:4px}img{max-width:100%;border-radius:6px}
.grid{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.muted{color:#778;font-size:13px}"""


def build_html(res: dict, cfg: dict, img_dir: str, errors_n: int = 8) -> str:
    sc = res.get("score")
    overall = sc["overall"] if sc else None
    cls = "grey" if overall is None else ("good" if overall >= 85 else ("mid" if overall >= 60 else "low"))
    ex_name = "Tư thế sẵn sàng (ready)" if res["exercise"] == "ready" else "Phát cầu trái tay (serve)"
    parts = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Kết quả — {html.escape(res['video'])}</title><style>{CSS}</style></head><body><div class='wrap'>",
             f"<h1>Kết quả chấm tư thế</h1><div class='sub'>{html.escape(res['video'])} · {ex_name} · {res['duration_s']:.1f} giây · {res['fps']:.0f} FPS</div>"]
    # ---- tóm tắt
    parts.append("<div class='card'><div class='grid'>")
    parts.append(f"<div><div class='score {cls}'>{'—' if overall is None else f'{overall:.0f}'}</div><div class='sub'>/ 100 · <b class='{cls}'>{html.escape(sc['grade']) if sc else html.escape(res['status'])}</b></div></div>")
    errs = (sc or {}).get("errors", [])
    if sc and sc["units"]:
        n_scored = len([u for u in sc["units"] if u["score"] is not None])
        what = f"{n_scored} đoạn sẵn sàng ({sc['scored_seconds']:.1f} giây)" if res["exercise"] == "ready" else f"{n_scored} lần phát"
        top = errs[:3]
        gain = sum(e["potential_gain"] for e in top)
        parts.append(f"<div><b>Đã chấm:</b> {what}<br>")
        if top:
            parts.append(f"<b>Việc cần sửa nhất:</b><ol>" + "".join(f"<li>{html.escape(e['advice'])} <span class='muted'>(+{e['potential_gain']:.1f} điểm)</span></li>" for e in top) + "</ol>")
            parts.append(f"<span class='muted'>Sửa được 3 việc này có thể tăng khoảng +{gain:.1f} điểm (ước tính; các lỗi được coi là độc lập).</span>")
        else:
            parts.append("<span class='ok'>Mọi góc đều nằm trong dung sai so với chuẩn HLV.</span>")
        parts.append("</div>")
    else:
        parts.append(f"<div><b>Không chấm được.</b><br>{html.escape(res.get('message', ''))}</div>")
    parts.append("</div></div>")
    for w in res.get("warnings", []):
        parts.append(f"<div class='warn'>⚠ {html.escape(w)}</div>")
    # ---- lỗi cần sửa
    if errs:
        parts.append("<h2>Cần sửa gì để tăng điểm</h2><div class='card'><table><tr><th>#</th><th>Phần</th><th>Lỗi</th><th>Mức</th><th>Hiện tại</th><th>Chuẩn</th><th>Cần làm</th><th>Lấy lại được</th></tr>")
        for i, e in enumerate(errs[:errors_n], 1):
            parts.append(f"<tr><td>{i}</td><td>{html.escape(e['where'])}</td><td>{html.escape(e['label'])}</td><td>{html.escape(e.get('severity', ''))}{' · ' + html.escape(e['kind'].lower()) if e.get('kind') and e['kind'] != 'LỖI CƠ BẢN' else ''}</td><td>{fmt_val(e['value'], e['unit'])}</td>"
                         f"<td>{('≤ ' + fmt_val(e['tol'], e['unit'])) if e['feature'] in ('follow_through', 'heel_lift') else (fmt_val(e['center'], e['unit']) + ' ± ' + fmt_val(e['tol'], e['unit']))}</td><td>{html.escape(e['advice'])}</td><td><b>+{e['potential_gain']:.1f}</b></td></tr>")
        parts.append("</table></div>")
    parts.append(f"<h2>Tổng quan</h2><div class='card'><img src='{img_b64(os.path.join(img_dir, 'dashboard.png'))}'></div>")
    # ---- chi tiết
    if sc and sc["units"]:
        parts.append("<h2>Chi tiết từng góc</h2>")
        if res["exercise"] == "ready":
            for u in sc["units"]:
                parts.append(f"<div class='card'><h3>Đoạn {u['id']} — frame {u['frames'][0]}–{u['frames'][1]} ({u['duration_s']:.1f} giây) — điểm {vn(u['score'], 0)}</h3>{_feature_table(u['features'])}</div>")
        else:
            labels = cfg["features"]["serve"]["phase_labels"]
            for u in sc["units"]:
                parts.append(f"<div class='card'><h3>Lần phát {u['id']} — điểm {vn(u['score'], 0)}</h3>")
                if u.get("faults"):
                    parts.append("<p>" + "".join(f"<span class='warn' style='display:block'>⚠ {html.escape(f['kind'].title())}: {html.escape(f['label'])} — trừ {f['penalty']:.0f} điểm. {html.escape(f['advice'])}</span>" for f in u["faults"]) + "</p>")
                if u.get("fault_metrics"):
                    parts.append("<p class='muted'>Số đo kiểm tra lỗi: " + ", ".join(f"{k} = {'—' if v is None else v}" for k, v in u["fault_metrics"].items()) + "</p>")
                for ph, r in u["phases"].items():
                    parts.append(f"<h4>{labels[ph]} <span class='muted'>(trọng số {r['weight']:.2f}, điểm {vn(r['score'], 0)})</span></h4>{_feature_table(r['features'])}")
                parts.append("</div>")
    # ---- chất lượng dữ liệu
    q = res["quality"]
    vq = res.get("video_qc")
    if vq and "fps" in vq:
        parts.append("<h2>Chất lượng video (FPS, rung, nhiễu)</h2><div class='card'><table><tr><th>Chỉ số</th><th>Giá trị</th><th>Ý nghĩa</th></tr>")
        for k, v, m in [("FPS", f"{vq['fps']:.0f} ({vq['width']}×{vq['height']})", "≥ 24 là đủ cho phát cầu; càng cao càng ít mờ ở cú vung"),
                        ("Frame trùng lặp", f"{vq['duplicate_frame_pct']:.0f}%", "cao = video bị nội suy tăng FPS / đứt nhịp"),
                        ("Rung máy", "—" if vq["shake_rms_pct"] != vq["shake_rms_pct"] else f"{vq['shake_rms_pct']:.2f}% bề rộng khung", "< 0,25% là ổn (máy trên chân đế)"),
                        ("Chiều cao người", f"{vq['subject_height_frac'] * 100:.0f}% khung (~{vq['subject_height_px']} px)", "≥ 50% khung thì khớp chính xác nhất"),
                        ("Nhiễu khớp (thô → sau làm mượt)", f"{vq['jitter_raw_pct_torso']:.1f}% → {vq['jitter_clean_pct_torso']:.1f}% chiều dài thân", f"làm mượt giảm {vq['jitter_reduction_pct']:.0f}% nhiễu, cửa sổ {vq['smooth_window_frames']} frame (≈ 0,23 giây)")]:
            parts.append(f"<tr><td>{k}</td><td>{html.escape(v)}</td><td class='muted'>{html.escape(m)}</td></tr>")
        parts.append("</table></div>")
    parts.append("<h2>Chất lượng dữ liệu và xử lý mất khớp</h2><div class='card'>")
    parts.append(f"<p>Chỉ số %Valid cũ (cả 10 khớp phải thấy cùng lúc): <b>{q['legacy_valid_pct']:.0f}%</b>. Khi tính <b>Valid theo từng góc</b> (chỉ cần các khớp của góc đó) "
                 f"và xử lý mất khớp (nội suy khoảng mất ≤ {cfg['quality']['gap_interp_max_s']} giây; dùng bên còn lại / tay thấy rõ nhất), độ phủ như biểu đồ dưới.</p>")
    if q.get("arm"):
        parts.append(f"<p>Tay dùng để đo (serve): <b>{q['arm']['side']}</b> — chế độ <b>{q['arm']['mode']}</b>.</p>")
    parts.append(f"<img src='{img_b64(os.path.join(img_dir, 'quality.png'))}'></div>")
    parts.append(f"<h2>Góc theo thời gian so với chuẩn</h2><div class='card'><img src='{img_b64(os.path.join(img_dir, 'angles_timeline.png'))}'></div>")
    ref = res.get("reference")
    parts.append("<h2>Thông tin kỹ thuật</h2><div class='card muted'>")
    if ref:
        parts.append(f"Chuẩn HLV dựng từ {ref['n_videos']} video: {html.escape(', '.join(ref['built_from']))}.<br>")
    parts.append(f"Phiên bản badminton3d {__version__} · MediaPipe Pose Landmarker (3D world) · ngưỡng visibility {cfg['quality']['visibility_min']} · "
                 f"điểm một góc = 100 khi trong dung sai, giảm tuyến tính về 0 khi lệch thêm {cfg['scoring']['decay_factor']}× dung sai.</div></div></body></html>")
    return "".join(parts)
