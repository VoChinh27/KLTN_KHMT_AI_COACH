from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout


class SportCard(QFrame):
    clicked = pyqtSignal(object)

    def __init__(self, info):
        super().__init__()
        self.info = info
        self.setFixedSize(200, 150)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("SportCard")
        self.setProperty("ready", info.ready)
        lay = QVBoxLayout(self)
        t = QLabel(info.display); t.setAlignment(Qt.AlignCenter); t.setObjectName("CardTitle")
        b = QLabel(("● " if info.ready else "") + info.badge); b.setAlignment(Qt.AlignCenter)
        lay.addWidget(t); lay.addWidget(b)
        if not info.ready:
            self.setStyleSheet("QFrame#SportCard{opacity:.4;} QLabel{color:#777;}")

    def mousePressEvent(self, e):
        self.clicked.emit(self.info)
