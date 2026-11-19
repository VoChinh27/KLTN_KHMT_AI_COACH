from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QComboBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from gui_app import constants as C

FILTERS = ["Tất cả", "Lỗi kỹ thuật", "Cảnh báo đo lường", "Đạt"]


class FindingsPanel(QWidget):
    jump_to_frame = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.combo = QComboBox(); self.combo.addItems(FILTERS)
        self.combo.currentIndexChanged.connect(self._apply)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["", "Chỉ số", "Giá trị / chuẩn", "Lời khuyên"])
        self.tree.itemClicked.connect(self._click)
        lay = QVBoxLayout(self); lay.addWidget(self.combo); lay.addWidget(self.tree)
        self.result, self.pauses = None, {}

    def set_result(self, result, pauses):
        self.result, self.pauses = result, {e.phase: e.frame for e in pauses}
        self._apply()

    def _apply(self):
        self.tree.clear()
        if not self.result: return
        f = self.combo.currentIndex()
        for ph in self.result.sub_phase_order:
            sp = self.result.sub_phase_results.get(ph)
            name = C.PHASE_VI.get(ph, ph)
            if not sp or not sp.has_score():
                QTreeWidgetItem(self.tree, ["", f"{name} — Bỏ qua", sp.skip_reason if sp else "", ""]); continue
            sc = sp.result.feedback.overall_score
            note = "  (chỉ tham khảo, không tính vào tổng)" if ph == "drive_phase" else ""
            top = QTreeWidgetItem(self.tree, ["", f"{name}  {sc:.0f}{note}", "", ""])
            top.setForeground(1, QColor(C.score_color(sc)))
            top.setData(0, 256, self.pauses.get(ph))
            for it in sp.result.feedback.items:
                et = getattr(it.error_type, "value", it.error_type)
                if f == 1 and et != "technique_error": continue
                if f == 2 and et != "measurement_error": continue
                if f == 3 and it.status_symbol != "✓": continue
                row = QTreeWidgetItem(top, [it.status_symbol, it.display_name,
                                            f"{it.student_value:.1f} / {it.reference_center:.1f}{it.unit}", it.message])
                row.setData(0, 256, self.pauses.get(ph))
            top.setExpanded(True)

    def _click(self, item, _):
        fr = item.data(0, 256)
        if fr is not None: self.jump_to_frame.emit(int(fr))
