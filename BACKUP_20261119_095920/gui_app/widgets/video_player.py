"""Trình phát: cv2.VideoCapture + QTimer + QLabel.
- frame_hook(frame_bgr, idx) -> frame_bgr : vẽ overlay.
- Chặn tại PauseEvent khi auto_pause bật. Đọc tuần tự (không seek mỗi khung) để mượt.
- Phím tắt (khi widget có focus trong cửa sổ): Space, ←/→, Shift+←/→."""
from typing import Callable, List, Optional

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QShortcut, QSlider, QVBoxLayout, QWidget


class VideoPlayer(QWidget):
    frame_changed = pyqtSignal(int)
    paused_at_event = pyqtSignal(object)      # PauseEvent
    playing_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps, self.total, self.idx = 30.0, 0, 0
        self.frame_hook: Optional[Callable] = None
        self.pauses: List = []
        self.auto_pause = False
        self.speed = 1.0
        self._skip: set = set()
        self._last = None
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)
        self.view = QLabel(alignment=Qt.AlignCenter); self.view.setMinimumSize(560, 315)
        self.view.setStyleSheet("background:#000;")
        self.slider = QSlider(Qt.Horizontal); self.slider.sliderMoved.connect(self.seek)
        self.btn = QPushButton("⏯"); self.btn.setFixedWidth(44); self.btn.clicked.connect(self.toggle)
        self.b_prev = QPushButton("⏮"); self.b_prev.setFixedWidth(44); self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next = QPushButton("⏭"); self.b_next.setFixedWidth(44); self.b_next.clicked.connect(lambda: self.step(1))
        self.time = QLabel("0.0s / 0.0s"); self.time.setMinimumWidth(110)
        bar = QHBoxLayout()
        for w in (self.b_prev, self.btn, self.b_next, self.slider, self.time):
            bar.addWidget(w)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1); lay.addLayout(bar)
        for key, fn in (("Space", self.toggle), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)),
                        ("Shift+Left", lambda: self.step(-10)), ("Shift+Right", lambda: self.step(10))):
            sc = QShortcut(QKeySequence(key), self); sc.setContext(Qt.WidgetWithChildrenShortcut); sc.activated.connect(fn)

    # ---- nạp / điều khiển ----
    def load(self, path: str):
        self.stop()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, max(self.total - 1, 0)); self._skip.clear()
        self.seek(0)

    def release(self):
        self.stop()
        if self.cap:
            self.cap.release(); self.cap = None

    def set_speed(self, s: float):
        self.speed = s
        if self.timer.isActive():
            self.timer.start(self._interval())

    def _interval(self) -> int:
        return max(1, int(1000 / self.fps / self.speed))

    def is_playing(self) -> bool:
        return self.timer.isActive()

    def play(self):
        if self.cap and not self.timer.isActive():
            self.timer.start(self._interval()); self.playing_changed.emit(True)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop(); self.playing_changed.emit(False)

    def toggle(self):
        self.stop() if self.timer.isActive() else self.play()

    def step(self, n: int):
        self.stop(); self.seek(self.idx + n)

    def continue_play(self):
        self._skip.add(self.idx); self.play()

    def seek(self, i: int):
        if not self.cap or self.total <= 0:
            return
        self.idx = max(0, min(self.total - 1, int(i)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
        ok, frame = self.cap.read()
        if ok:
            self._render(frame)

    def refresh(self):
        """Vẽ lại khung hiện tại (sau khi đổi frame_hook/biên pha)."""
        if self._last is not None:
            self._render(self._last, keep=True)

    def _tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok or self.idx + 1 >= self.total:
            self.stop(); return
        self.idx += 1
        self._render(frame)
        if self.auto_pause and self.idx not in self._skip:
            for ev in self.pauses:
                if ev.frame == self.idx:
                    self.stop(); self.paused_at_event.emit(ev); break

    def _render(self, frame, keep=False):
        self._last = frame
        out = self.frame_hook(frame, self.idx) if self.frame_hook else frame
        h, w = out.shape[:2]
        img = QImage(out.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.view.setPixmap(QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.slider.blockSignals(True); self.slider.setValue(self.idx); self.slider.blockSignals(False)
        self.time.setText(f"{self.idx / self.fps:.1f}s / {self.total / self.fps:.1f}s")
        self.frame_changed.emit(self.idx)
