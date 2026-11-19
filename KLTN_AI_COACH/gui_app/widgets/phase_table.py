from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

from gui_app import constants as C


class PhaseTable(QTableWidget):
    HEAD = ["#", "Pha", "Bắt đầu", "Kết thúc", "Tin cậy (đầu→cuối)", "Trạng thái"]

    def __init__(self):
        super().__init__(4, len(self.HEAD))
        self.setHorizontalHeaderLabels(self.HEAD)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def refresh(self, review, fps: float):
        b, conf = review.b, review.proposal.boundary_conf
        for i, ph in enumerate(C.PHASE_ORDER):
            absent = b[i + 1] <= b[i]
            cells = [str(i + 1), C.PHASE_VI[ph],
                     "—" if absent else f"{b[i]} ({b[i]/fps:.2f}s)",
                     "—" if absent else f"{b[i+1]-1} ({(b[i+1]-1)/fps:.2f}s)",
                     f"{conf[i]:.2f} → {conf[min(i+1, len(conf)-1)]:.2f}",
                     "Vắng" if absent else "Có mặt"]
            for j, t in enumerate(cells):
                it = QTableWidgetItem(t)
                if j == 4:
                    it.setForeground(QColor(C.conf_color(min(conf[i], conf[min(i + 1, len(conf) - 1)]))))
                self.setItem(i, j, it)
        self.resizeColumnsToContents()
