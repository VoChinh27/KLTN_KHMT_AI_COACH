"""Điểm vào: python -m gui_app   (hoặc python -m gui_app.main)"""
import sys

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

from gui_app import constants as C
from gui_app.app_context import AppContext
from gui_app.pages.add_sport_page import AddSportPage
from gui_app.pages.history_page import HistoryPage
from gui_app.pages.home_page import HomePage
from gui_app.pages.sport_page import SportPage


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(C.APP_NAME); self.setMinimumSize(1280, 720)
        self.home, self.sport = HomePage(ctx), SportPage(ctx)
        self.add_sport, self.history = AddSportPage(ctx), HistoryPage(ctx)
        self.stack = QStackedWidget()
        for w in (self.home, self.sport, self.add_sport, self.history): self.stack.addWidget(w)
        # thanh trên: đường dẫn + nút Trang chủ / Lịch sử
        self.crumb = QLabel("Trang chủ")
        self.b_home, self.b_hist = QPushButton("Trang chủ"), QPushButton("Lịch sử")
        self.b_home.clicked.connect(self._go_home); self.b_hist.clicked.connect(self._open_history)
        bar = QHBoxLayout(); bar.addWidget(self.crumb, 1); bar.addWidget(self.b_home); bar.addWidget(self.b_hist)
        box = QWidget(); lay = QVBoxLayout(box); lay.addLayout(bar); lay.addWidget(self.stack, 1)
        self.setCentralWidget(box)
        self.home.sport_selected.connect(self._open_sport)
        self.home.add_sport_requested.connect(lambda: self._show(self.add_sport, "Trang chủ › Thêm môn"))
        self.home.history_requested.connect(self._open_history)
        self.home.quit_requested.connect(self.close)
        self.sport.home_requested.connect(self._go_home)
        self.sport.busy_changed.connect(self._busy)
        self.add_sport.finished.connect(self._go_home)
        self.statusBar().showMessage(ctx.pipeline.describe())

    def _show(self, w, crumb):
        self.stack.setCurrentWidget(w); self.crumb.setText(crumb)

    def _busy(self, b: bool):                       # khóa điều hướng khi worker chạy
        self.b_home.setEnabled(not b); self.b_hist.setEnabled(not b)
        self.statusBar().showMessage("Đang xử lý…" if b else self.ctx.pipeline.describe())

    def _go_home(self):
        if self.stack.currentWidget() is self.sport and self.sport.busy(): return
        self.home.refresh(); self._show(self.home, "Trang chủ")

    def _open_history(self):
        if self.sport.busy(): return
        self.history.refresh(); self._show(self.history, "Trang chủ › Lịch sử")

    def _open_sport(self, sport: str):
        self.sport.start(sport)
        name = {v: k for k, v in C.SPORT_ALIAS.items()}.get(sport, sport)
        self._show(self.sport, f"Trang chủ › {name}")

    def closeEvent(self, e):                         # D01
        if QMessageBox.question(self, "Thoát", "Thoát ứng dụng?") != QMessageBox.Yes:
            e.ignore(); return
        if self.sport.busy():
            if QMessageBox.question(self, "Đang xử lý", "Tác vụ đang chạy. Hủy và thoát?") != QMessageBox.Yes:
                e.ignore(); return
            self.sport.cancel_all()
        e.accept()


def _install_excepthook():
    """Lỗi trong slot PyQt5 sẽ làm app abort nếu không có excepthook -> hiện hộp thoại D21 (chi tiết lỗi)."""
    import traceback

    def hook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        print(text, file=sys.stderr)
        box = QMessageBox(); box.setIcon(QMessageBox.Critical); box.setWindowTitle("Lỗi không mong muốn")
        box.setText(f"{et.__name__}: {ev}"); box.setDetailedText(text); box.exec_()
    sys.excepthook = hook


def main():
    _install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName(C.APP_NAME)
    if C.FONT_PATH.exists():
        fid = QFontDatabase.addApplicationFont(str(C.FONT_PATH))
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0], 10))
    if C.QSS_PATH.exists():                          # nạp THẬT file QSS
        app.setStyleSheet(C.QSS_PATH.read_text(encoding="utf-8"))
    win = MainWindow(AppContext())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
