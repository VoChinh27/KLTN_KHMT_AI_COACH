from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui_app import constants as C


class PhaseTimeline(QWidget):
    """Vẽ 4 đoạn pha + 5 mốc biên (màu theo độ tin cậy). Click = tua; kéo mốc = chỉnh biên."""
    seek = pyqtSignal(int)
    boundary_dragged = pyqtSignal(int, int)    # idx, frame
    boundary_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(56)
        self.b, self.conf, self.total, self.cur, self._drag = [0] * 5, [1.0] * 5, 1, 0, None

    def set_data(self, b, conf, total):
        self.b, self.conf, self.total = list(b), list(conf), max(total, 1); self.update()

    def set_current(self, f):
        self.cur = f; self.update()

    def _x(self, f):
        return int(10 + (self.width() - 20) * f / self.total)

    def _f(self, x):
        return int(max(0, min(self.total, (x - 10) * self.total / max(self.width() - 20, 1))))

    def paintEvent(self, _):
        p = QPainter(self)
        for i, ph in enumerate(C.PHASE_ORDER):
            if self.b[i + 1] > self.b[i]:
                p.fillRect(self._x(self.b[i]), 12, self._x(self.b[i + 1]) - self._x(self.b[i]), 26,
                           QColor(*C.PHASE_RGB[ph]))
        for i, f in enumerate(self.b):
            p.setPen(QPen(QColor(C.conf_color(self.conf[min(i, len(self.conf) - 1)])), 3))
            p.drawLine(self._x(f), 6, self._x(f), 44)
        p.setPen(QPen(Qt.white, 2)); p.drawLine(self._x(self.cur), 0, self._x(self.cur), self.height())

    def mousePressEvent(self, e):
        near = [i for i, f in enumerate(self.b) if abs(self._x(f) - e.x()) <= 6]
        if near:
            self._drag = near[0]; self.boundary_selected.emit(self._drag)
        else:
            self.seek.emit(self._f(e.x()))

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.boundary_dragged.emit(self._drag, self._f(e.x()))

    def mouseReleaseEvent(self, _):
        self._drag = None
