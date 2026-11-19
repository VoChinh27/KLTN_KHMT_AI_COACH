"""Trang 3: Thêm môn — trình hướng dẫn 4 bước. Giai đoạn hiện tại: chỉ lưu YAML ở trạng thái Nháp
(engine phân pha đang gắn cứng 4 pha xuất phát thấp)."""
import re
import sys
from pathlib import Path

import yaml
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget, QDoubleSpinBox)

from gui_app import constants as C
from gui_app.widgets.process_dialog import ProcessDialog

STEPS = ["1. Thông tin môn", "2. Video mẫu", "3. Chọn mô hình", "4. Cấu hình YAML"]


class AddSportPage(QWidget):
    finished = pyqtSignal()

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.title = QLabel(); self.title.setObjectName("CardTitle")
        banner = QLabel("⚠ Engine hiện gắn cứng 4 pha của xuất phát thấp: môn mới chỉ được lưu ở trạng thái Nháp cho đến khi mở rộng engine.")
        banner.setStyleSheet("color:#f1c40f"); banner.setWordWrap(True)
        self.stack = QStackedWidget()
        for p in (self._p1(), self._p2(), self._p3(), self._p4()): self.stack.addWidget(p)
        self.b_prev, self.b_next, b_cancel = QPushButton("Quay lại"), QPushButton("Tiếp"), QPushButton("Hủy")
        self.b_prev.clicked.connect(lambda: self._go(-1)); self.b_next.clicked.connect(lambda: self._go(1))
        b_cancel.clicked.connect(self.finished.emit)
        nav = QHBoxLayout(); nav.addWidget(b_cancel); nav.addStretch(); nav.addWidget(self.b_prev); nav.addWidget(self.b_next)
        root = QVBoxLayout(self)
        for w in (self.title, banner): root.addWidget(w)
        root.addWidget(self.stack, 1); root.addLayout(nav)
        self._show(0)

    # ---- các bước ----
    def _p1(self):
        w = QWidget(); lay = QVBoxLayout(w); f = QFormLayout()
        self.name, self.code, self.desc = QLineEdit(), QLineEdit(), QPlainTextEdit(); self.desc.setMaximumHeight(80)
        self.code.setPlaceholderText("vd: long_jump (chữ thường không dấu, số, _)")
        f.addRow("Tên hiển thị *", self.name); f.addRow("Mã môn *", self.code); f.addRow("Mô tả", self.desc)
        self.ptable = QTableWidget(0, 5); self.ptable.setHorizontalHeaderLabels(["Mã pha", "Tên pha", "Mô tả", "Thứ tự", "Trọng số"])
        b_add = QPushButton("＋ Thêm pha"); b_add.clicked.connect(self._add_phase_row)
        b_del = QPushButton("Xóa pha chọn"); b_del.clicked.connect(lambda: self.ptable.removeRow(self.ptable.currentRow()))
        row = QHBoxLayout(); row.addWidget(b_add); row.addWidget(b_del); row.addStretch()
        lay.addLayout(f); lay.addWidget(QLabel("Các pha kỹ thuật")); lay.addWidget(self.ptable); lay.addLayout(row)
        self._add_phase_row()
        return w

    def _add_phase_row(self):
        r = self.ptable.rowCount(); self.ptable.insertRow(r)
        for j, v in enumerate(["", "", "", str(r + 1), "1.0"]): self.ptable.setItem(r, j, QTableWidgetItem(v))

    def _p2(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.videos = QListWidget()
        b = QPushButton("Tải video…"); b.clicked.connect(self._add_videos)
        b_rm = QPushButton("Bỏ video chọn"); b_rm.clicked.connect(lambda: self.videos.takeItem(self.videos.currentRow()))
        b_mark = QPushButton("Đánh dấu pha (mở công cụ gán nhãn)"); b_mark.clicked.connect(self._mark)
        row = QHBoxLayout()
        for x in (b, b_rm, b_mark): row.addWidget(x)
        lay.addLayout(row); lay.addWidget(self.videos)
        return w

    def _add_videos(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Video mẫu", "", C.VIDEO_FILTER)
        for p in ps: self.videos.addItem(p)

    def _mark(self):
        it = self.videos.currentItem()
        if not it:
            QMessageBox.information(self, "Đánh dấu pha", "Chọn một video trong danh sách trước."); return
        ProcessDialog(self, "Công cụ đánh dấu pha", "tools.marking.mark_phases",
                      ["--video", it.text(), "--out-dir", str(C.ROOT / "data" / "phases")]).exec_()

    def _p3(self):
        w = QWidget(); lay = QVBoxLayout(w); f = QFormLayout()
        self.pose, self.pmodel = QComboBox(), QComboBox()
        self.pose.addItems(C.POSE_BACKENDS)
        self.onnx = QLineEdit(str(C.ROOT / "weights" / "rtmpose-m.onnx"))
        b_onnx = QPushButton("…"); b_onnx.setFixedWidth(34)
        b_onnx.clicked.connect(lambda: self.onnx.setText(QFileDialog.getOpenFileName(self, "ONNX", "", "ONNX (*.onnx)")[0] or self.onnx.text()))
        r = QHBoxLayout(); r.addWidget(self.onnx); r.addWidget(b_onnx)
        f.addRow("Mô hình pose", self.pose); f.addRow("Đường dẫn ONNX", r); f.addRow("Mô hình phân pha", self.pmodel)
        b_train = QPushButton("Huấn luyện mô hình pha…"); b_train.clicked.connect(self._train)
        self.train_hint = QLabel("Cần các file nhãn trong data/phases và pose cache trong data/pose_cache.")
        self.train_hint.setWordWrap(True)
        lay.addLayout(f); lay.addWidget(b_train); lay.addWidget(self.train_hint); lay.addStretch()
        return w

    def _train(self):
        ProcessDialog(self, "Huấn luyện mô hình phân pha", "tools.training.train_phase_model", []).exec_()
        self.app.pipeline.reload_segmenter(); self._load_models()

    def _load_models(self):
        self.pmodel.clear()
        vs = self.app.registry.phase_model_versions()
        self.pmodel.addItems(vs or ["Chưa có"])
        cur = self.app.registry.current_phase_model()
        if cur and cur in vs: self.pmodel.setCurrentText(cur)

    def _p4(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.yaml = QPlainTextEdit(); self.ystate = QLabel()
        b_tpl = QPushButton("Tạo từ mẫu sprint_start.yaml"); b_tpl.clicked.connect(self._from_template)
        b_imp = QPushButton("Nhập file YAML…"); b_imp.clicked.connect(self._import)
        b_chk = QPushButton("Kiểm tra"); b_chk.clicked.connect(self._check)
        row = QHBoxLayout()
        for x in (b_tpl, b_imp, b_chk): row.addWidget(x)
        self.ref = QTableWidget(0, 4); self.ref.setHorizontalHeaderLabels(["Pha", "Chỉ số", "Min", "Max"]); self.ref.setMaximumHeight(160)
        lay.addLayout(row); lay.addWidget(self.yaml, 1); lay.addWidget(self.ystate)
        lay.addWidget(QLabel("Khoảng chuẩn cố định (đọc từ YAML)")); lay.addWidget(self.ref)
        return w

    def _from_template(self):
        tpl = C.SPORT_CONFIG_DIR / "sprint_start.yaml"
        text = tpl.read_text(encoding="utf-8") if tpl.exists() else "sport: \nphases: []\ncyclic: {}\n"
        code = self.code.text().strip()
        if code: text = re.sub(r"(?m)^sport:.*$", f"sport: {code}", text, count=1)
        self.yaml.setPlainText(text); self._check()

    def _import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Nhập YAML", "", "YAML (*.yaml *.yml)")
        if p: self.yaml.setPlainText(Path(p).read_text(encoding="utf-8")); self._check()

    def _check(self) -> bool:
        try:
            raw = yaml.safe_load(self.yaml.toPlainText()) or {}
            from core.sport_interface import SportConfig
            errs = SportConfig._from_dict(raw).validate()
        except Exception as e:
            self.ystate.setText(f"✗ {e}"); return False
        self.ref.setRowCount(0)
        for ph, rows in (raw.get("fixed_reference_ranges") or {}).items():
            for r in rows or []:
                i = self.ref.rowCount(); self.ref.insertRow(i)
                for j, v in enumerate([ph, r.get("feature_key"), r.get("min"), r.get("max")]):
                    self.ref.setItem(i, j, QTableWidgetItem(str(v)))
        self.ystate.setText("✓ Hợp lệ" if not errs else "⚠ Cảnh báo: " + "; ".join(errs))
        return True            # validate() chỉ trả cảnh báo; lỗi cú pháp/thiếu trường đã bị bắt ở trên

    # ---- điều hướng ----
    def _show(self, i: int):
        self.stack.setCurrentIndex(i); self.title.setText(STEPS[i])
        self.b_prev.setEnabled(i > 0); self.b_next.setText("Lưu môn" if i == 3 else "Tiếp")
        if i == 2: self._load_models()

    def _go(self, d: int):
        i = self.stack.currentIndex()
        if d > 0 and not self._validate_step(i): return
        if d > 0 and i == 3: return self._save()
        self._show(max(0, min(3, i + d)))

    def _validate_step(self, i: int) -> bool:
        if i == 0:
            code = self.code.text().strip()
            if not self.name.text().strip() or not re.fullmatch(r"[a-z][a-z0-9_]*", code):
                QMessageBox.warning(self, "Thông tin môn", "Nhập tên hiển thị và mã môn (chữ thường không dấu, số, _)."); return False
            if (C.SPORT_CONFIG_DIR / f"{code}.yaml").exists():
                if QMessageBox.question(self, "Trùng mã", "Mã môn đã tồn tại. Tiếp tục (sẽ hỏi ghi đè khi lưu)?") != QMessageBox.Yes: return False
        return True

    def _save(self):
        if not self._check():
            QMessageBox.warning(self, "YAML", "YAML chưa hợp lệ."); return
        code = self.code.text().strip()
        raw = yaml.safe_load(self.yaml.toPlainText())
        raw["sport"] = code; raw["display_name"] = self.name.text().strip()
        raw["status"] = "draft"
        dst = C.SPORT_CONFIG_DIR / f"{code}.yaml"
        if dst.exists():                                                                    # D19
            box = QMessageBox(self); box.setText(f"'{dst.name}' đã tồn tại.")
            ow = box.addButton("Ghi đè", QMessageBox.AcceptRole); box.addButton("Hủy", QMessageBox.RejectRole); box.exec_()
            if box.clickedButton() is not ow: return
        try:
            dst.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Lỗi ghi file", str(e)); return
        QMessageBox.information(self, "Đã lưu", f"Đã lưu môn ở trạng thái Nháp:\n{dst}")
        self.finished.emit()
