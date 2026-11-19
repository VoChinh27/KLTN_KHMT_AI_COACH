import yaml
from PyQt5.QtWidgets import QPlainTextEdit


class YamlViewer(QPlainTextEdit):
    """Chỉ đọc. load() trả (ok, thông báo lỗi cú pháp)."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def load(self, path):
        try:
            text = open(path, encoding="utf-8").read()
            self.setPlainText(text)
            yaml.safe_load(text)
            return True, ""
        except Exception as e:
            return False, str(e)
