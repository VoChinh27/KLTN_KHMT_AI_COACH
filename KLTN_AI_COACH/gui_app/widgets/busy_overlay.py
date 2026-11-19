from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget


class BusyOverlay(QWidget):
    """Lớp phủ tiến trình; percent=-1 => dạng không xác định."""

    def __init__(self, parent, on_cancel=None):
        super().__init__(parent)
        self.setStyleSheet("background: rgba(0,0,0,170);")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self.msg = QLabel("Đang xử lý…"); self.msg.setAlignment(Qt.AlignCenter)
        self.bar = QProgressBar(); self.bar.setFixedWidth(360)
        self.cancel_cb = on_cancel
        self.btn = QPushButton("Hủy"); self.btn.clicked.connect(lambda: self.cancel_cb and self.cancel_cb())
        for w in (self.msg, self.bar, self.btn):
            lay.addWidget(w, alignment=Qt.AlignCenter)
        self.hide()

    def show_busy(self, text="Đang xử lý…"):
        self.resize(self.parent().size()); self.msg.setText(text)
        self.bar.setRange(0, 0); self.raise_(); self.show()

    def update_progress(self, pct: int, text: str):
        self.msg.setText(text)
        self.bar.setRange(0, 0) if pct < 0 else (self.bar.setRange(0, 100), self.bar.setValue(pct))

    def done(self):
        self.hide()
