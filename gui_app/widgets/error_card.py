from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui_app import constants as C


class ErrorCard(QFrame):
    """Thẻ lỗi khi dừng: tiêu đề pha, đỏ = lỗi, xanh = động tác đúng; phân trang '1/N' nếu nhiều lỗi."""

    def __init__(self, on_continue, on_next_error):
        super().__init__()
        self.setObjectName("ErrorCard")
        self.title, self.err, self.ok = QLabel(), QLabel(), QLabel()
        for l in (self.err, self.ok):
            l.setWordWrap(True)
        self.err.setStyleSheet("color:#e74c3c"); self.ok.setStyleSheet("color:#2ecc71")
        self.page = QLabel(); self.ev, self.i = None, 0
        b_prev, b_nx = QPushButton("◀"), QPushButton("▶")
        b_prev.clicked.connect(lambda: self._go(-1)); b_nx.clicked.connect(lambda: self._go(1))
        b_c, b_n = QPushButton("Tiếp tục"), QPushButton("Lỗi tiếp ▶")
        b_c.clicked.connect(on_continue); b_n.clicked.connect(on_next_error)
        nav = QHBoxLayout()
        for w in (b_prev, self.page, b_nx): nav.addWidget(w)
        nav.addStretch(); nav.addWidget(b_c); nav.addWidget(b_n)
        lay = QVBoxLayout(self)
        for w in (self.title, self.err, self.ok): lay.addWidget(w)
        lay.addLayout(nav); self.hide()

    def show_event(self, ev):
        self.ev, self.i = ev, 0; self._fill(); self.show()

    def message(self, text: str):
        self.ev = None; self.title.setText(text); self.err.setText(""); self.ok.setText(""); self.page.setText(""); self.show()

    def _go(self, d):
        if self.ev: self.i = (self.i + d) % len(self.ev.items); self._fill()

    def _fill(self):
        it = self.ev.items[self.i]
        sev = "✗ Lỗi kỹ thuật" if self.ev.severity == "error" else "⚠ Cảnh báo"
        self.title.setText(f"<b>{C.PHASE_VI.get(self.ev.phase, self.ev.phase)}</b> — {sev}")
        self.err.setText("✗ " + it["message_error"]); self.ok.setText("✓ Động tác đúng: " + it["message_correct"])
        self.page.setText(f"{self.i + 1}/{len(self.ev.items)}")
