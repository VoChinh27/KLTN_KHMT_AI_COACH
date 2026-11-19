"""Cửa sổ log chạy tiến trình ngoài (huấn luyện mô hình pha, đánh dấu pha...) bằng QProcess."""
import sys

from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout

from gui_app import constants as C


class ProcessDialog(QDialog):
    def __init__(self, parent, title: str, module: str, args: list):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(760, 460)
        self.log = QPlainTextEdit(readOnly=True)
        self.b_stop, self.b_close = QPushButton("Dừng"), QPushButton("Đóng")
        self.b_close.setEnabled(False)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(self.b_stop); row.addWidget(self.b_close)
        lay = QVBoxLayout(self); lay.addWidget(self.log); lay.addLayout(row)
        self.p = QProcess(self); self.p.setWorkingDirectory(str(C.ROOT)); self.p.setProcessChannelMode(QProcess.MergedChannels)
        self.p.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(bytes(self.p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        self.p.finished.connect(lambda code, _: (self.log.appendPlainText(f"\n[Kết thúc, mã thoát {code}]"), self.b_close.setEnabled(True), self.b_stop.setEnabled(False)))
        self.b_stop.clicked.connect(self.p.kill); self.b_close.clicked.connect(self.accept)
        self.log.appendPlainText("$ " + " ".join([sys.executable, "-m", module] + args))
        self.p.start(sys.executable, ["-m", module] + args)

    def reject(self):                                   # chặn đóng khi tiến trình còn chạy
        if self.p.state() == QProcess.NotRunning:
            super().reject()
