"""Tab ② Trích xuất pha: đề xuất -> người duyệt -> OK mới chấm. Hỗ trợ chế độ đánh dấu thủ công."""
import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                             QShortcut, QSpinBox, QVBoxLayout, QWidget)

from gui_app import constants as C
from gui_app.services.config_service import ConfigService
from gui_app.services.phase_review_service import ManualProposal, PhaseReviewService
from gui_app.widgets.busy_overlay import BusyOverlay
from gui_app.widgets.phase_table import PhaseTable
from gui_app.widgets.phase_timeline import PhaseTimeline
from gui_app.widgets.video_player import VideoPlayer
from gui_app.workers.pose_phase_worker import PosePhaseWorker

BOUNDARY_LABELS = ["Đầu clip → Vào chỗ", "Vào chỗ | Sẵn sàng", "Sẵn sàng | Bộc phát", "Bộc phát | Chạy lao", "Chạy lao → Cuối clip"]


class PhaseTab(QWidget):
    approved = pyqtSignal()
    back = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, app, ctx):
        super().__init__()
        self.app, self.ctx, self.review, self.worker, self.cfg = app, ctx, None, None, None
        self.player, self.table, self.timeline = VideoPlayer(), PhaseTable(), PhaseTimeline()
        self.desc = QPlainTextEdit(readOnly=True); self.desc.setMaximumHeight(150)
        self.flags = QLabel(); self.flags.setWordWrap(True)
        self.mode = QLabel(); self.mode.setStyleSheet("color:#f1c40f")
        self.sel = QComboBox(); self.sel.addItems(BOUNDARY_LABELS)
        self.step = QSpinBox(); self.step.setRange(1, 60); self.step.setValue(1)
        self.absent = [QCheckBox(C.PHASE_VI[p]) for p in C.PHASE_ORDER]
        for i, cb in enumerate(self.absent):
            cb.toggled.connect(lambda v, i=i: self._set_absent(i, v))
        self.b_extract = QPushButton("Trích xuất pha"); self.b_extract.clicked.connect(self.extract)
        self.b_ok = QPushButton("OK – Chạy chấm điểm"); self.b_ok.setEnabled(False); self.b_ok.clicked.connect(self.ok)
        b_back = QPushButton("← Quay lại"); b_back.clicked.connect(self.back.emit)

        def mk(text, fn):
            b = QPushButton(text); b.clicked.connect(fn); return b
        row1 = QHBoxLayout()
        for w in (QLabel("Biên:"), self.sel, mk("−", lambda: self._move(-1)), mk("+", lambda: self._move(1)),
                  self.step, QLabel("khung"), mk("Đặt biên tại khung hiện tại", self._set_here)):
            row1.addWidget(w)
        row2 = QHBoxLayout()
        for w in (mk("Hoàn tác", lambda: self._do("undo")), mk("Làm lại", lambda: self._do("redo")),
                  mk("Khôi phục đề xuất", lambda: self._do("reset"))):
            row2.addWidget(w)
        row2.addStretch(); row2.addWidget(QLabel("Pha vắng mặt:"))
        for cb in self.absent: row2.addWidget(cb)
        left = QVBoxLayout(); left.addWidget(self.player, 1); left.addWidget(self.timeline)
        left.addLayout(row1); left.addLayout(row2)
        right = QVBoxLayout(); right.addWidget(self.mode); right.addWidget(self.table)
        right.addWidget(QLabel("Mô tả pha đang chọn")); right.addWidget(self.desc); right.addWidget(self.flags)
        foot = QHBoxLayout(); foot.addWidget(self.b_extract); foot.addStretch(); foot.addWidget(b_back); foot.addWidget(self.b_ok)
        body = QHBoxLayout(); body.addLayout(left, 3); body.addLayout(right, 2)
        root = QVBoxLayout(self); root.addLayout(body, 1); root.addLayout(foot)
        self.busy = BusyOverlay(self, on_cancel=self._cancel)
        self.timeline.seek.connect(self.player.seek)
        self.player.frame_changed.connect(self.timeline.set_current)
        self.timeline.boundary_selected.connect(self.sel.setCurrentIndex)
        self.timeline.boundary_dragged.connect(self._dragged)
        self.table.cellClicked.connect(lambda r, _: self._describe(r))
        self.player.frame_hook = self._hook
        for key, fn in (("Ctrl+Z", lambda: self._do("undo")), ("Ctrl+Y", lambda: self._do("redo"))):
            sc = QShortcut(QKeySequence(key), self); sc.activated.connect(fn)
        self._load_cfg()

    # ---- nạp ----
    def _load_cfg(self):
        try:
            self.cfg = ConfigService(self.ctx.yaml_path) if self.ctx.yaml_path else None
        except Exception:
            self.cfg = None

    def load(self):
        self._load_cfg()
        self.player.load(str(self.ctx.input_video))
        self.b_extract.setText("Trích xuất lại" if self.review else "Trích xuất pha")

    def resizeEvent(self, e):
        super().resizeEvent(e); self.busy.resize(self.size())

    # ---- worker ----
    def _cancel(self):
        if self.worker: self.worker.cancel()
        self.busy.done(); self.busy_changed.emit(False)

    def extract(self):
        if self.review and self.review.edits:
            if QMessageBox.question(self, "Trích xuất lại", "Bỏ các chỉnh sửa hiện tại?") != QMessageBox.Yes:
                return
        self.worker = PosePhaseWorker(self.app.pipeline, self.ctx)
        self.worker.progress.connect(self.busy.update_progress)
        self.worker.finished_ok.connect(self._extracted); self.worker.failed.connect(self._failed)
        self.busy.show_busy("Đang xử lý…"); self.busy_changed.emit(True); self.worker.start()

    def _extracted(self, res):
        self.busy.done(); self.busy_changed.emit(False)
        if res is None:
            return
        frames, proposal, err = res
        c = self.ctx
        c.frames = frames
        c.manual_mode = proposal is None
        if proposal is None:
            if not self._ask_manual(err):                                          # D10
                return
            proposal = ManualProposal(n_frames=c.total_frames)
        c.proposal = proposal
        self.review = PhaseReviewService(proposal, c.input_video.name, c.fps, c.total_frames, manual=c.manual_mode)
        c.auto_phases_path = c.session_dir / "phases_auto.phases.json"
        self.review.save_auto(c.auto_phases_path)
        self.mode.setText("✎ CHẾ ĐỘ ĐÁNH DẤU THỦ CÔNG — chưa có mô hình phân pha. Hãy tự đặt 4 biên." if c.manual_mode
                          else f"Đề xuất của mô hình {proposal.model_version}")
        self.b_extract.setText("Trích xuất lại")
        self._refresh()
        gaps = bool(frames) and (frames[-1].frame_idx + 1 != len(frames))
        extra = ["pose_frame_gaps (pose thiếu khung — biên có thể lệch)"] if gaps else []
        self.flags.setText("⚠ Cờ: " + (", ".join(list(proposal.flags) + extra) or "không") if (proposal.flags or extra) else "")
        self.player.refresh()

    def _ask_manual(self, err: str) -> bool:
        box = QMessageBox(self); box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Không phân pha tự động được")
        box.setText("Chưa có mô hình phân pha." if err.startswith("no_model") else "Mô hình phân pha báo lỗi.")
        box.setDetailedText(err)
        man = box.addButton("Đánh dấu thủ công", QMessageBox.AcceptRole); box.addButton("Hủy", QMessageBox.RejectRole)
        box.exec_()
        return box.clickedButton() is man

    def _failed(self, msg, tb):
        self.busy.done(); self.busy_changed.emit(False)
        if "Không phát hiện được người" in msg:                                      # D09
            box = QMessageBox(self); box.setIcon(QMessageBox.Warning); box.setText("Không phát hiện người trong video.")
            box.addButton("Chọn video khác", QMessageBox.AcceptRole); box.addButton("Đóng", QMessageBox.RejectRole)
            box.exec_()
            if box.buttonRole(box.clickedButton()) == QMessageBox.AcceptRole: self.back.emit()
        else:                                                                       # D21
            m = QMessageBox(self); m.setIcon(QMessageBox.Critical); m.setText(f"Lỗi: {msg}"); m.setDetailedText(tb); m.exec_()

    # ---- chỉnh biên ----
    def _dragged(self, i, f):
        self.sel.setCurrentIndex(i)
        if self.review: self.review.set_boundary(i, f); self._refresh()

    def _move(self, sign):
        if self.review: self.review.move_boundary(self.sel.currentIndex(), sign * self.step.value()); self._refresh()

    def _set_here(self):
        if self.review: self.review.set_boundary(self.sel.currentIndex(), self.player.idx); self._refresh()

    def _set_absent(self, i, v):
        if self.review and self.review.is_absent(i) != v:
            self.review.set_absent(i, v); self._refresh()

    def _do(self, name):
        if self.review: getattr(self.review, name)(); self._refresh()

    def _refresh(self):
        r = self.review
        self.table.refresh(r, self.ctx.fps)
        self.timeline.set_data(r.b, r.proposal.boundary_conf, self.ctx.total_frames)
        for i, cb in enumerate(self.absent):
            cb.blockSignals(True); cb.setChecked(r.is_absent(i)); cb.blockSignals(False)
        errs, warns = r.validate(), r.warnings()
        self.b_ok.setEnabled(not errs)                                              # lỗi -> khóa OK
        base = self.flags.text().split("\n")[0] if self.flags.text().startswith("⚠ Cờ") else ""
        extra = ("✗ " + "; ".join(errs)) if errs else (("⚠ " + "; ".join(warns)) if warns else "")
        self.flags.setText("\n".join(x for x in (base, extra) if x))
        self.flags.setStyleSheet("color:#e74c3c" if errs else "color:#f1c40f")
        self.player.refresh()

    def _describe(self, row):
        ph = C.PHASE_ORDER[row]
        text = f"{C.PHASE_VI[ph]}\n"
        if self.cfg:
            text += self.cfg.phase_description(ph) + "\n"
            feats = self.cfg.scored_features(ph)
            if feats:
                text += "\nChỉ số sẽ chấm: " + ", ".join(f"{k} (×{v})" for k, v in feats.items())
        self.desc.setPlainText(text)

    def _hook(self, frame, idx):
        """Vẽ khung xương + viền màu pha hiện tại lên video."""
        if not self.review or not self.ctx.frames:
            return frame
        from render_overlay import PHASE_FOCUS_JOINTS, draw_skeleton
        out = frame.copy()
        phase = next((p for p, (s, e) in self.review.windows().items() if s <= idx <= e), None)
        pf = next((f for f in self.ctx.frames if f.frame_idx == idx), None) if len(self.ctx.frames) < 5000 else None
        if pf is not None:
            draw_skeleton(out, pf, phase or "none", PHASE_FOCUS_JOINTS.get(phase, set()))
        if phase:
            r, g, b = C.PHASE_RGB[phase]
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (b, g, r), 6)
        return out

    # ---- OK ----
    def ok(self):
        r = self.review
        if not self.ctx.manual_mode and r.low_conf_boundaries() and r.edits == 0:   # D11
            box = QMessageBox(self); box.setText("Còn biên độ tin cậy thấp. Bạn đã kiểm tra chưa?")
            box.addButton("Kiểm tra lại", QMessageBox.RejectRole)
            go = box.addButton("Xác nhận & chấm", QMessageBox.AcceptRole); box.exec_()
            if box.clickedButton() is not go:
                return
        c = self.ctx
        try:
            c.phases_path = c.session_dir / "phases.phases.json"
            c.phase_set = r.save_reviewed(c.auto_phases_path, c.phases_path)
        except Exception as e:
            QMessageBox.critical(self, "Không lưu được nhãn pha", str(e)); return
        self.approved.emit()
