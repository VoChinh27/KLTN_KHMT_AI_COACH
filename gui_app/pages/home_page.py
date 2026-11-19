"""Trang 1: Chọn môn."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from gui_app import constants as C
from gui_app.widgets.sport_card import SportCard


class HomePage(QWidget):
    sport_selected = pyqtSignal(str)
    add_sport_requested = pyqtSignal()
    history_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.grid = QGridLayout(); self.grid.setSpacing(18)
        title = QLabel("AI PERSONAL SPORTS COACH"); title.setObjectName("Title"); title.setAlignment(Qt.AlignCenter)
        sub = QLabel("Chọn môn huấn luyện"); sub.setAlignment(Qt.AlignCenter)
        self.warn = QLabel(); self.warn.setStyleSheet("color:#f1c40f"); self.warn.setWordWrap(True)
        b_add = QPushButton("＋ Thêm môn huấn luyện"); b_add.clicked.connect(self.add_sport_requested.emit)
        b_hist = QPushButton("Lịch sử"); b_hist.clicked.connect(self.history_requested.emit)
        b_quit = QPushButton("Thoát"); b_quit.clicked.connect(self.quit_requested.emit)
        foot = QHBoxLayout()
        for w in (b_add, b_hist): foot.addWidget(w)
        foot.addStretch(); foot.addWidget(b_quit)
        center = QHBoxLayout(); center.addStretch(); center.addLayout(self.grid); center.addStretch()
        root = QVBoxLayout(self)
        for w in (title, sub, self.warn): root.addWidget(w)
        root.addStretch(); root.addLayout(center); root.addStretch(); root.addLayout(foot)
        self.refresh()

    def refresh(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w: w.deleteLater()
        for i, info in enumerate(self.app.registry.scan()):
            card = SportCard(info); card.clicked.connect(self._clicked)
            self.grid.addWidget(card, i // 4, i % 4)
        miss = []
        if not (C.ROOT / "weights" / "rtmpose-m.onnx").exists(): miss.append("weights/rtmpose-m.onnx")
        if not self.app.registry.has_phase_model(): miss.append("models/phase_segmenter/registry.json")
        if not C.SYSTEM_CONFIG.exists(): miss.append("config/system_config.yaml")
        self.warn.setText("⚠ Thiếu: " + ", ".join(miss) + " — một số chức năng sẽ không chạy được." if miss else "")   # D03

    def _clicked(self, info):
        if info.ready:
            self.sport_selected.emit(info.code); return
        box = QMessageBox(self); box.setWindowTitle("Chưa sẵn sàng")                                                 # D02
        box.setText(f"Môn '{info.display}' chưa có mô hình hoặc chưa hoàn thiện.")
        add = box.addButton("Thêm môn", QMessageBox.AcceptRole); box.addButton("Đóng", QMessageBox.RejectRole); box.exec_()
        if box.clickedButton() is add: self.add_sport_requested.emit()
