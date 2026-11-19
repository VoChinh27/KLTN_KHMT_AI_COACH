"""Báo cáo TXT + PDF (QTextDocument + QPrinter)."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter

from gui_app import constants as C


def build_txt(ctx) -> str:
    head = [f"Ngày chấm : {datetime.now():%d/%m/%Y %H:%M}",
            f"Học viên  : {ctx.student_code} - {ctx.student_name}",
            f"Quay lúc  : {ctx.recorded_at:%d/%m/%Y %H:%M}",
            f"Video     : {ctx.input_video.name if ctx.input_video else ''}",
            f"Môn       : {ctx.sport}   YAML: {ctx.yaml_path}", ""]
    tail = (["", "Cảnh báo đo lường (không phải lỗi học viên):"] + [f" - {q}" for q in ctx.quality]) if ctx.quality else []
    body = [ctx.result.render_text(), ""]
    for ph in ctx.result.sub_phase_order:
        sp = ctx.result.sub_phase_results.get(ph)
        if sp and sp.has_score():
            body += [f"--- {C.PHASE_VI.get(ph, ph)} ---", sp.result.feedback.render_text(), ""]
    return "\n".join(head + body + tail)


def build_html(ctx, snapshots: dict | None = None) -> str:
    r = ctx.result
    score = "—" if r.overall_score != r.overall_score else f"{r.overall_score:.0f}/100"
    h = [f"<h2>Báo cáo huấn luyện — {escape(ctx.student_name)} ({escape(ctx.student_code)})</h2>",
         f"<p>Điểm tổng: <b>{score}</b> · tin cậy: {r.score_confidence} · "
         f"{r.n_phases_scored}/{len(r.sub_phase_order)} pha · nguồn pha: {escape(r.phase_source)}</p>"]
    for ph in r.sub_phase_order:
        sp = r.sub_phase_results.get(ph)
        h.append(f"<h3>{C.PHASE_VI.get(ph, ph)}</h3>")
        if not sp or not sp.has_score():
            h.append(f"<p><i>Bỏ qua — {escape(sp.skip_reason if sp else '')}</i></p>"); continue
        if snapshots and ph in snapshots:
            h.append(f"<img src='{snapshots[ph]}' width='320'/>")
        h.append("<table border='1' cellspacing='0' cellpadding='4'><tr><th></th><th>Chỉ số</th>"
                 "<th>Của bạn</th><th>Chuẩn</th><th>Lời khuyên</th></tr>")
        for it in sp.result.feedback.items:
            h.append(f"<tr><td>{it.status_symbol}</td><td>{escape(it.display_name)}</td>"
                     f"<td>{it.student_value:.1f}{it.unit}</td><td>{it.reference_center:.1f}{it.unit}</td>"
                     f"<td>{escape(it.message)}</td></tr>")
        h.append("</table>")
    flags = [k for k, v in r.quality_flags.items() if v]
    if flags:
        h.append(f"<p><b>Cảnh báo chất lượng dữ liệu:</b> {', '.join(flags)}</p>")
    if getattr(ctx, "quality", None):
        h.append("<h3>Cảnh báo đo lường (không phải lỗi của học viên)</h3><ul>" +
                 "".join(f"<li>{escape(q)}</li>" for q in ctx.quality) + "</ul>")
    return "".join(h)


def build_pdf(ctx, path: Path, snapshots: dict | None = None) -> None:
    doc = QTextDocument()
    doc.setHtml(build_html(ctx, snapshots))
    pr = QPrinter(QPrinter.HighResolution)
    pr.setOutputFormat(QPrinter.PdfFormat)
    pr.setOutputFileName(str(path))
    doc.print_(pr)
