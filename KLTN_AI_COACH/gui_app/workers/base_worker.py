"""BaseWorker(QThread): progress / finished / failed. Hủy = bỏ kết quả khi worker kết thúc."""
import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    progress = pyqtSignal(int, str)     # percent (-1 = không xác định), message
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str, str)       # message, traceback

    def __init__(self):
        super().__init__()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def work(self):                     # lớp con ghi đè, trả kết quả
        raise NotImplementedError

    def run(self):
        try:
            res = self.work()
            if not self.cancelled:
                self.finished_ok.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
