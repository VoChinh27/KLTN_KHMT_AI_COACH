"""Tab ④ Báo cáo & Lưu."""
from pathlib import Path

from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser,
                             QVBoxLayout, QWidget)

from gui_app.services import report_service as R
from gui_app.widgets.busy_overlay import BusyOverlay
from gui_app.workers.report_worker import ReportWorker


class ReportTab(QWidget):
    again = pyqtSignal(bool)          # True = cùng học viên
    home = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, app, ctx, get_renderer):
        super().__init__()
        self.app, self.ctx, self.get_renderer = app, ctx, get_renderer
        self.exported, self._w, self._out = False, None, None
        self.view = QTextBrowser(); self.db_label = QLabel()
        self.c_pdf, self.c_txt, self.c_img = QCheckBox("PDF"), QCheckBox("TXT"), QCheckBox("Kèm ảnh pha")
        for c in (self.c_pdf, self.c_txt, self.c_img): c.setChecked(True)
        b_exp = QPushButton("Xuất báo cáo"); b_exp.clicked.connect(self.export)
        b_again = QPushButton("Chấm tiếp"); b_again.clicked.connect(self._again)
        b_home = QPushButton("Về trang chủ"); b_home.clicked.connect(self._home)
        opts = QHBoxLayout()
        for w in (QLabel("Tùy chọn xuất:"), self.c_pdf, self.c_txt, self.c_img): opts.addWidget(w)
        opts.addStretch(); opts.addWidget(b_exp)
        foot = QHBoxLayout(); foot.addWidget(self.db_label); foot.addStretch(); foot.addWidget(b_again); foot.addWidget(b_home)
        root = QVBoxLayout(self); root.addWidget(self.view, 1); root.addLayout(opts); root.addLayout(foot)
        self.busy = BusyOverlay(self)

    def resizeEvent(self, e):
        super().resizeEvent(e); self.busy.resize(self.size())

    def refresh(self):
        self.exported = bool(self.ctx.report_pdf or self.ctx.report_txt)
        self.view.setHtml(R.build_html(self.ctx))
        self.db_label.setText(f"Đã lưu bản ghi #{self.ctx.db_session_id} ✓" if self.ctx.db_session_id else "⚠ Chưa lưu CSDL")

    # ---- xuất ----
    def export(self):
        if not (self.c_pdf.isChecked() or self.c_txt.isChecked()):
            QMessageBox.information(self, "Xuất báo cáo", "Hãy chọn ít nhất PDF hoặc TXT."); return
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu báo cáo", str(self.ctx.session_dir))   # D15
        if not d: return
        self._out = Path(d)
        for name in ("report.pdf", "report.txt"):                                                         # D19
            if (self._out / name).exists():
                box = QMessageBox(self); box.setText(f"'{name}' đã tồn tại.")
                ow = box.addButton("Ghi đè", QMessageBox.AcceptRole); box.addButton("Hủy", QMessageBox.RejectRole); box.exec_()
                if box.clickedButton() is not ow: return
                break
        self._w = ReportWorker(self.ctx, self.get_renderer(), self._out, self.c_txt.isChecked(), self.c_img.isChecked())
        self.busy.show_busy("Đang xuất báo cáo…"); self.busy_changed.emit(True)
        self._w.finished_ok.connect(self._done); self._w.failed.connect(self._failed); self._w.start()

    def _done(self, res):
        try:
            if self.c_pdf.isChecked():                                  # PDF dựng ở luồng chính
                pdf = self._out / "report.pdf"
                R.build_pdf(self.ctx, pdf, res.get("snaps") if self.c_img.isChecked() else None)
                self.ctx.report_pdf = pdf
            self.ctx.report_txt = res.get("txt")
        except Exception as e:
            self._failed(str(e), ""); return
        self.busy.done(); self.busy_changed.emit(False); self.exported = True
        if self.ctx.db_session_id:
            self.app.db.update_paths(self.ctx.db_session_id, report_txt_path=self.ctx.report_txt, report_pdf_path=self.ctx.report_pdf)
        box = QMessageBox(self); box.setIcon(QMessageBox.Information); box.setText("Đã xuất báo cáo.")      # D16
        b1 = box.addButton("Mở thư mục", QMessageBox.ActionRole)
        b2 = box.addButton("Mở PDF", QMessageBox.ActionRole) if self.ctx.report_pdf else None
        box.addButton("Đóng", QMessageBox.RejectRole); box.exec_()
        if box.clickedButton() is b1: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._out)))
        elif b2 is not None and box.clickedButton() is b2: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.ctx.report_pdf)))

    def _failed(self, msg, tb):
        self.busy.done(); self.busy_changed.emit(False)
        m = QMessageBox(self); m.setIcon(QMessageBox.Critical); m.setText(f"Lỗi ghi file: {msg}"); m.setDetailedText(tb); m.exec_()

    # ---- rời tab ----
    def _confirm_leave(self) -> bool:                                                                    # D18
        if self.exported or self.ctx.overlay_path:
            return True
        box = QMessageBox(self); box.setText("Bạn chưa xuất báo cáo hoặc lưu video.")
        now = box.addButton("Xuất ngay", QMessageBox.AcceptRole)
        skip = box.addButton("Bỏ qua", QMessageBox.DestructiveRole); box.addButton("Hủy", QMessageBox.RejectRole); box.exec_()
        if box.clickedButton() is now: self.export(); return False
        return box.clickedButton() is skip

    def _again(self):                                                                                    # D17
        if not self._confirm_leave(): return
        box = QMessageBox(self); box.setText("Chấm tiếp cho:")
        same = box.addButton("Cùng học viên (video mới)", QMessageBox.AcceptRole)
        box.addButton("Học viên mới", QMessageBox.ActionRole); box.addButton("Hủy", QMessageBox.RejectRole); box.exec_()
        b = box.clickedButton()
        if b is same: self.again.emit(True)
        elif box.buttonRole(b) == QMessageBox.ActionRole: self.again.emit(False)

    def _home(self):
        if self._confirm_leave(): self.home.emit()
