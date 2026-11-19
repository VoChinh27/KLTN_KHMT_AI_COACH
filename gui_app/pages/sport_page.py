"""Trang 2: 4 tab, khóa/mở theo bước; giữ SessionContext; điều phối ScoreWorker."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QTabWidget

from gui_app.session import SessionContext, Step
from gui_app.tabs.phase_tab import PhaseTab
from gui_app.tabs.report_tab import ReportTab
from gui_app.tabs.result_tab import ResultTab
from gui_app.tabs.upload_tab import UploadTab
from gui_app.workers.score_worker import ScoreWorker


class SportPage(QTabWidget):
    home_requested = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.ctx = SessionContext()
        self._w = None
        self._build()

    def _build(self):
        while self.count():
            w = self.widget(0); self.removeTab(0); w.deleteLater()
        self.up, self.ph = UploadTab(self.app, self.ctx), PhaseTab(self.app, self.ctx)
        self.rs = ResultTab(self.app, self.ctx)
        self.rp = ReportTab(self.app, self.ctx, lambda: self.rs.renderer)
        for t, n in ((self.up, "① Video & Cấu hình"), (self.ph, "② Trích xuất pha"),
                     (self.rs, "③ Kết quả"), (self.rp, "④ Báo cáo & Lưu")):
            self.addTab(t, n)
        self._lock_from(Step.PHASE)
        self.up.proceed.connect(self._to_phase)
        self.up.video_changed.connect(self._video_changed)
        self.ph.back.connect(lambda: self.setCurrentIndex(Step.UPLOAD))
        self.ph.approved.connect(self._score)
        self.rs.back.connect(self._back_to_phase)
        self.rs.next_report.connect(self._to_report)
        self.rp.again.connect(self._again)
        self.rp.home.connect(self._go_home)
        for t in (self.ph, self.rs, self.rp):
            t.busy_changed.connect(self._set_busy)

    def start(self, sport: str):
        """Mỗi lần vào môn = phiên mới (tránh rò trạng thái giữa các môn/học viên)."""
        self.ctx = SessionContext(sport=sport)
        self._build(); self.setCurrentIndex(Step.UPLOAD)

    # ---- khóa / bận ----
    def _lock_from(self, step: int):
        for i in range(int(step), 4):
            self.setTabEnabled(i, False)

    def _set_busy(self, b: bool):
        self.tabBar().setEnabled(not b); self.busy_changed.emit(b)

    def _video_changed(self):
        if self.ctx.phase_set is not None or self.ctx.frames:
            self.ctx.reset_from(Step.PHASE)
            self.ph.review = None
        self._lock_from(Step.PHASE)

    # ---- luồng ----
    def _to_phase(self):
        self.setTabEnabled(Step.PHASE, True); self.setCurrentIndex(Step.PHASE); self.ph.load()

    def _score(self):
        self.ph.busy.show_busy("Đang chấm điểm…"); self._set_busy(True)
        self._w = ScoreWorker(self.app.pipeline, self.ctx, include_warn=self.rs.warn.isChecked())
        self._w.finished_ok.connect(self._scored); self._w.failed.connect(self._score_failed); self._w.start()

    def _scored(self, res):
        self.ph.busy.done(); self._set_busy(False)
        if res is None: return
        self.ctx.result, self.ctx.pauses, self.ctx.quality = res
        r = self.ctx.result
        if getattr(r, "needs_review", False) or r.n_phases_scored == 0:
            QMessageBox.warning(self, "Không chấm được",
                                "Tất cả pha bị bỏ qua (thiếu dữ liệu hoặc cần duyệt lại). Hãy kiểm tra lại các pha ở tab ②.")
            self.ctx.result = None; return
        try:
            self.ctx.db_session_id = self.app.db.save_session(self.ctx)
        except Exception as e:                                                       # D20
            box = QMessageBox(self); box.setIcon(QMessageBox.Warning); box.setText(f"Lỗi ghi CSDL: {e}")
            retry = box.addButton("Thử lại", QMessageBox.AcceptRole); box.addButton("Tiếp tục không lưu", QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is retry:
                try: self.ctx.db_session_id = self.app.db.save_session(self.ctx)
                except Exception: pass
        self.setTabEnabled(Step.RESULT, True); self.rs.show_result(); self.setCurrentIndex(Step.RESULT)

    def _score_failed(self, msg, tb):                                                # D12 + D21
        self.ph.busy.done(); self._set_busy(False)
        box = QMessageBox(self); box.setIcon(QMessageBox.Critical); box.setText(f"Lỗi chấm điểm: {msg}")
        box.setDetailedText(tb)
        retry = box.addButton("Thử lại", QMessageBox.AcceptRole); box.addButton("Về tab ②", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is retry: self._score()

    def _back_to_phase(self):
        if QMessageBox.question(self, "Sửa pha", "Thay đổi sẽ xóa kết quả chấm hiện tại. Tiếp tục?") == QMessageBox.Yes:
            self.rs.player.release()
            self.ctx.reset_from(Step.RESULT); self._lock_from(Step.RESULT); self.setCurrentIndex(Step.PHASE)

    def _to_report(self):
        self.setTabEnabled(Step.REPORT, True); self.rp.refresh(); self.setCurrentIndex(Step.REPORT)

    def _again(self, same_student: bool):
        keep = (self.ctx.student_code, self.ctx.student_name) if same_student else ("", "")
        self.rs.player.release(); self.ph.player.release(); self.up.player.release()
        self.ctx = SessionContext(sport=self.ctx.sport); self._build()
        self.up.code.setText(keep[0]); self.up.name.setText(keep[1]); self.setCurrentIndex(Step.UPLOAD)

    def _go_home(self):
        for t in (self.up, self.ph, self.rs): t.player.release()
        self.home_requested.emit()

    # ---- đóng app ----
    def busy(self) -> bool:
        return any(w is not None and w.isRunning() for w in (self._w, self.ph.worker, self.rs._w, self.rp._w))

    def cancel_all(self):
        for w in (self._w, self.ph.worker, self.rs._w, self.rp._w):
            if w is not None and w.isRunning():
                w.cancel(); w.wait(3000)
