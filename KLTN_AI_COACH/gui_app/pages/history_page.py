"""Màn Lịch sử: lọc, mở báo cáo/video, xem chi tiết, xóa, xuất CSV."""
import csv
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from gui_app import constants as C

HEAD = ["ID", "Ngày chấm", "Mã HV", "Họ tên", "Môn", "Điểm", "Tin cậy"]


class HistoryPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app, self.rows = app, []
        self.q = QLineEdit(); self.q.setPlaceholderText("Lọc theo mã/tên học viên…")
        self.sport = QComboBox(); self.sport.addItem("Tất cả môn", "")
        self.table = QTableWidget(0, len(HEAD)); self.table.setHorizontalHeaderLabels(HEAD)
        self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.q.textChanged.connect(self.refresh); self.sport.currentIndexChanged.connect(self.refresh)
        self.table.cellDoubleClicked.connect(lambda *_: self.detail())

        def mk(t, f):
            b = QPushButton(t); b.clicked.connect(f); return b
        top = QHBoxLayout(); top.addWidget(self.q, 1); top.addWidget(self.sport)
        bot = QHBoxLayout()
        for t, f in (("Xem chi tiết", self.detail), ("Mở báo cáo", self.open_report), ("Mở video", self.open_video),
                     ("Xóa", self.delete), ("Xuất CSV", self.export_csv)):
            bot.addWidget(mk(t, f))
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.table, 1); lay.addLayout(bot)

    def refresh(self, *_):
        sport = self.sport.currentData() or ""
        self.rows = self.app.db.list_sessions(self.q.text().strip(), sport)
        known = {self.sport.itemData(i) for i in range(self.sport.count())}
        for r in self.rows:
            if r["sport"] not in known:
                self.sport.blockSignals(True); self.sport.addItem(r["sport"], r["sport"]); self.sport.blockSignals(False)
                known.add(r["sport"])
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            sc = r["overall_score"]
            vals = [r["id"], r["scored_at"], r["student_code"], r["full_name"], r["sport"],
                    "—" if sc is None else f"{sc:.0f}", r["score_confidence"] or ""]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if j == 5 and sc is not None: it.setForeground(QColor(C.score_color(sc)))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()

    def _cur(self):
        i = self.table.currentRow()
        return self.rows[i] if 0 <= i < len(self.rows) else None

    @staticmethod
    def _open(path):
        if path and Path(path).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))); return True
        return False

    def open_report(self):
        r = self._cur()
        if r and not (self._open(r["report_pdf_path"]) or self._open(r["report_txt_path"])):
            QMessageBox.information(self, "Báo cáo", "Phiên này chưa xuất báo cáo (hoặc file đã bị xóa).")

    def open_video(self):
        r = self._cur()
        if r and not (self._open(r["output_video_path"]) or self._open(r["input_video_path"])):
            QMessageBox.information(self, "Video", "Không tìm thấy file video.")

    def detail(self):
        r = self._cur()
        if not r: return
        h = [f"<h3>#{r['id']} — {r['full_name']} ({r['student_code']})</h3><p>{r['scored_at']} · điểm: {r['overall_score']}</p>"]
        for ph in self.app.db.phases_of(r["id"]):
            h.append(f"<h4>{C.PHASE_VI.get(ph['phase'], ph['phase'])} — " +
                     (f"{ph['score']:.0f}" if ph["score"] is not None else f"bỏ qua ({ph['skip_reason']})") + "</h4>")
        h.append("<table border='1' cellpadding='3'><tr><th></th><th>Pha</th><th>Chỉ số</th><th>Của bạn</th><th>Chuẩn</th><th>Lời khuyên</th></tr>")
        for f in self.app.db.findings_of(r["id"]):
            sv = "" if f["student_value"] is None else f"{f['student_value']:.1f}"
            rc = "" if f["reference_center"] is None else f"{f['reference_center']:.1f}"
            h.append(f"<tr><td>{f['status']}</td><td>{f['phase']}</td><td>{f['display_name']}</td><td>{sv}</td><td>{rc}</td><td>{f['message']}</td></tr>")
        h.append("</table>")
        d = QDialog(self); d.setWindowTitle("Chi tiết phiên"); d.resize(900, 600)
        b = QTextBrowser(); b.setHtml("".join(h)); QVBoxLayout(d).addWidget(b); d.exec_()   # xem lại không cần chấm lại

    def delete(self):
        r = self._cur()
        if r and QMessageBox.question(self, "Xóa", f"Xóa bản ghi #{r['id']}? (file trong thư mục phiên được giữ lại)") == QMessageBox.Yes:
            self.app.db.delete_session(r["id"]); self.refresh()

    def export_csv(self):
        p, _ = QFileDialog.getSaveFileName(self, "Xuất CSV", "lich_su.csv", "CSV (*.csv)")
        if not p: return
        with open(p, "w", newline="", encoding="utf-8-sig") as f:       # utf-8-sig để Excel đọc đúng tiếng Việt
            w = csv.writer(f); w.writerow(HEAD + ["Báo cáo PDF", "Video"])
            for r in self.rows:
                w.writerow([r["id"], r["scored_at"], r["student_code"], r["full_name"], r["sport"], r["overall_score"],
                            r["score_confidence"], r["report_pdf_path"], r["output_video_path"] or r["input_video_path"]])
        QMessageBox.information(self, "Xuất CSV", f"Đã xuất {len(self.rows)} dòng.")
