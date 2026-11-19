from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui_app import constants as C


class PhaseScoreCards(QWidget):
    """4 thẻ: tên pha, điểm (màu theo ngưỡng), trọng số; pha bỏ qua hiện skip_reason."""

    def __init__(self):
        super().__init__()
        self.lay = QHBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0)

    def set_result(self, result, weights: dict):
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w: w.deleteLater()
        for ph in result.sub_phase_order:
            sp = result.sub_phase_results.get(ph)
            f = QFrame(); f.setObjectName("PhaseCard"); v = QVBoxLayout(f)
            v.addWidget(QLabel(f"<b>{C.PHASE_VI.get(ph, ph)}</b>"))
            if sp and sp.has_score():
                sc = sp.result.feedback.overall_score
                lab = QLabel(f"<span style='font-size:22px;color:{C.score_color(sc)}'>{sc:.0f}</span>")
                v.addWidget(lab)
                w = weights.get(ph)
                note = "chỉ tham khảo — không tính vào điểm tổng" if (w == 0 or ph == "drive_phase") \
                    else (f"trọng số {w:.2f}" if w is not None else "")
                v.addWidget(QLabel(note))
            else:
                v.addWidget(QLabel("<span style='font-size:18px;color:#888'>Bỏ qua</span>"))
                r = QLabel(sp.skip_reason if sp else ""); r.setWordWrap(True); v.addWidget(r)
            self.lay.addWidget(f)
