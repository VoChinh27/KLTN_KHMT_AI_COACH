"""Tab ③ Kết quả: video overlay dừng tại lỗi + danh sách phát hiện."""
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from gui_app import constants as C
from gui_app.services.config_service import ConfigService
from gui_app.services.event_builder import build_events
from gui_app.services.overlay_renderer import OverlayRenderer
from gui_app.widgets.busy_overlay import BusyOverlay
from gui_app.widgets.error_card import ErrorCard
from gui_app.widgets.findings_panel import FindingsPanel
from gui_app.widgets.phase_score_cards import PhaseScoreCards
from gui_app.widgets.video_player import VideoPlayer
from gui_app.workers.export_video_worker import ExportVideoWorker


class ResultTab(QWidget):
    back, next_report = pyqtSignal(), pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, app, ctx):
        super().__init__()
        self.app, self.ctx, self.renderer, self._w = app, ctx, None, None
        self.summary = QLabel(); self.summary.setWordWrap(True)
        self.cards = PhaseScoreCards()
        self.player, self.panel = VideoPlayer(), FindingsPanel()
        self.card = ErrorCard(self._continue, self._next_error)
        self.auto = QCheckBox("Tự động dừng tại lỗi"); self.auto.setChecked(True)
        self.warn = QCheckBox("Dừng cả cảnh báo ⚠")
        self.resume = QSpinBox(); self.resume.setRange(0, 30); self.resume.setSpecialValueText("chờ bấm"); self.resume.setSuffix(" giây")
        self.speed = QComboBox(); self.speed.addItems(["0.25x", "0.5x", "1x", "2x"]); self.speed.setCurrentText("1x")
        self.quality = QPlainTextEdit(readOnly=True); self.quality.setMaximumHeight(90)
        gq = QGroupBox("Cảnh báo chất lượng (không phải lỗi của học viên)"); QVBoxLayout(gq).addWidget(self.quality)
        self.b_save = QPushButton("Lưu video…"); self.b_save.clicked.connect(self.save_video)
        b_back = QPushButton("← Sửa pha"); b_back.clicked.connect(self.back.emit)
        b_next = QPushButton("Sang báo cáo →"); b_next.clicked.connect(self.next_report.emit)
        ctrl = QHBoxLayout()
        for w in (self.auto, self.warn, QLabel("Tự tiếp tục sau:"), self.resume, QLabel("Tốc độ:"), self.speed):
            ctrl.addWidget(w)
        ctrl.addStretch()
        left = QVBoxLayout(); left.addWidget(self.player, 1); left.addWidget(self.card); left.addLayout(ctrl)
        right = QVBoxLayout(); right.addWidget(self.panel, 3); right.addWidget(gq, 1)
        foot = QHBoxLayout(); foot.addWidget(self.b_save); foot.addStretch(); foot.addWidget(b_back); foot.addWidget(b_next)
        body = QHBoxLayout(); body.addLayout(left, 3); body.addLayout(right, 2)
        root = QVBoxLayout(self)
        for w in (self.summary, self.cards): root.addWidget(w)
        root.addLayout(body, 1); root.addLayout(foot)
        self.busy = BusyOverlay(self)
        self.auto.toggled.connect(lambda v: setattr(self.player, "auto_pause", v))
        self.warn.toggled.connect(self._rebuild_pauses)
        self.speed.currentTextChanged.connect(lambda t: self.player.set_speed(float(t[:-1])))
        self.player.paused_at_event.connect(self._show_card)
        self.panel.jump_to_frame.connect(self._jump)

    def resizeEvent(self, e):
        super().resizeEvent(e); self.busy.resize(self.size())

    # ---- hiển thị ----
    def show_result(self):
        c, r = self.ctx, self.ctx.result
        nan = r.overall_score != r.overall_score
        score = "—" if nan else f"{r.overall_score:.0f}/100"
        flags = [k for k, v in r.quality_flags.items() if v]
        txt = (f"<b>Điểm tổng {score}</b> · tin cậy: {r.score_confidence} · {r.n_phases_scored}/{len(r.sub_phase_order)} pha · "
               f"nguồn pha: {r.phase_source}")
        if nan: txt += "<br><span style='color:#f1c40f'>⚠ Không đủ dữ liệu cho điểm tổng (cần ≥ 2 pha được chấm)</span>"
        if flags: txt += f"<br>⚠ cờ chất lượng: {', '.join(flags)}"
        self.summary.setText(txt)
        weights = {}
        try: weights = ConfigService(c.yaml_path).cfg.phase_weights or {}
        except Exception: pass
        self.cards.set_result(r, weights)
        windows = self._windows()
        self.renderer = OverlayRenderer(c.frames, r, windows, c.fps, c.total_frames,
                                        source=c.phase_set.marked_by if c.phase_set else "auto_reviewed")
        self.player.load(str(c.input_video))
        self.player.frame_hook = self.renderer.render
        self.player.pauses, self.player.auto_pause = c.pauses, self.auto.isChecked()
        self.panel.set_result(r, c.pauses)
        self.quality.setPlainText("\n".join(c.quality) or "Không có.")
        if not c.pauses: self.card.message("Không phát hiện lỗi kỹ thuật cần dừng.")
        else: self.card.hide()

    def _windows(self):
        return {n: (w.start_frame, w.end_frame) for n, w in self.ctx.phase_set.reps[0].present()}

    def _rebuild_pauses(self, *_):
        if not self.ctx.result: return
        self.ctx.pauses, self.ctx.quality = build_events(self.ctx.result, self._windows(), self.warn.isChecked())
        self.player.pauses = self.ctx.pauses
        self.panel.set_result(self.ctx.result, self.ctx.pauses)
        self.quality.setPlainText("\n".join(self.ctx.quality) or "Không có.")

    # ---- dừng / tiếp tục ----
    def _show_card(self, ev):
        self.card.show_event(ev)
        if self.resume.value() > 0:
            QTimer.singleShot(self.resume.value() * 1000, lambda e=ev: self.card.ev is e and self._continue())

    def _continue(self):
        self.card.hide(); self.player.continue_play()

    def _jump(self, frame: int):
        self.player.stop(); self.player.seek(frame)
        ev = next((e for e in self.ctx.pauses if e.frame == frame), None)
        if ev: self.card.show_event(ev)

    def _next_error(self):
        if not self.ctx.pauses: return
        nxt = next((e for e in self.ctx.pauses if e.frame > self.player.idx), self.ctx.pauses[0])
        self.player.stop(); self.player.seek(nxt.frame); self.card.show_event(nxt)

    # ---- lưu video ----
    def save_video(self):
        p, _ = QFileDialog.getSaveFileName(self, "Lưu video", str(self.ctx.session_dir / "overlay.mp4"), "MP4 (*.mp4)")  # D13
        if not p: return
        self._w = ExportVideoWorker(self.ctx, self.renderer, p)
        self.busy.cancel_cb = self._w.cancel
        self.busy.show_busy("Đang ghi video…"); self.busy_changed.emit(True)           # D14
        self._w.progress.connect(self.busy.update_progress)
        self._w.finished_ok.connect(self._saved); self._w.failed.connect(self._save_failed); self._w.start()

    def _saved(self, out):
        self.busy.done(); self.busy_changed.emit(False)
        if out is None: return
        self.ctx.overlay_path = out
        if self.ctx.db_session_id: self.app.db.update_paths(self.ctx.db_session_id, output_video_path=out)
        QMessageBox.information(self, "Đã lưu", f"Đã lưu video:\n{out}")

    def _save_failed(self, msg, tb):
        self.busy.done(); self.busy_changed.emit(False)
        m = QMessageBox(self); m.setIcon(QMessageBox.Critical); m.setText(f"Không ghi được video: {msg}"); m.setDetailedText(tb); m.exec_()
