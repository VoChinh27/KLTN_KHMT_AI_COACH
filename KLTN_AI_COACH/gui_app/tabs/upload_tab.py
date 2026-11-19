"""Tab ① Video & Cấu hình."""
from pathlib import Path

from PyQt5.QtCore import QDateTime, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDateTimeEdit, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from gui_app import constants as C
from gui_app.services import file_service
from gui_app.services.config_service import ConfigService
from gui_app.widgets.video_player import VideoPlayer
from gui_app.widgets.yaml_viewer import YamlViewer


class UploadTab(QWidget):
    proceed = pyqtSignal()
    video_changed = pyqtSignal()

    def __init__(self, app, ctx):
        super().__init__()
        self.app, self.ctx = app, ctx
        self._meta, self.yaml_ok, self.min_fps = None, False, 30.0
        self.player = VideoPlayer()
        self.info = QLabel("Chưa chọn video"); self.info.setWordWrap(True)
        b_pick = QPushButton("Chọn video…  (Ctrl+O)"); b_pick.setShortcut("Ctrl+O"); b_pick.clicked.connect(self.pick_video)
        b_re = QPushButton("Chọn lại"); b_re.clicked.connect(self.pick_video)
        b_clear = QPushButton("Xóa"); b_clear.clicked.connect(self.clear_video)
        self.code, self.name, self.note = QLineEdit(), QLineEdit(), QLineEdit()
        self.code.setPlaceholderText("VD: HV001")
        self.code.editingFinished.connect(self._autofill)
        self.when = QDateTimeEdit(QDateTime.currentDateTime()); self.when.setCalendarPopup(True)
        self.when.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.yaml_box = QComboBox()
        for p in sorted(C.SPORT_CONFIG_DIR.glob("*.yaml")):
            self.yaml_box.addItem(p.name, str(p))
        i = self.yaml_box.findText(f"{ctx.sport}.yaml")
        if i >= 0: self.yaml_box.setCurrentIndex(i)
        b_yaml = QPushButton("Chọn file YAML…"); b_yaml.clicked.connect(self.pick_yaml)
        self.yaml_box.currentIndexChanged.connect(self._load_yaml)
        self.yaml = YamlViewer(); self.yaml_state = QLabel(); self.yaml_state.setWordWrap(True)
        self.next = QPushButton("Tiếp tục → Trích xuất pha"); self.next.setEnabled(False)
        self.next.clicked.connect(self._proceed)
        for w in (self.code, self.name, self.note):
            w.textChanged.connect(self._validate)

        gv = QGroupBox("Video đầu vào"); lv = QVBoxLayout(gv)
        row = QHBoxLayout()
        for w in (b_pick, b_re, b_clear): row.addWidget(w)
        lv.addLayout(row); lv.addWidget(self.info); lv.addWidget(self.player, 1)
        gs = QGroupBox("Học viên"); fs = QFormLayout(gs)
        fs.addRow("Mã học viên *", self.code); fs.addRow("Họ tên *", self.name)
        fs.addRow("Thời gian quay", self.when); fs.addRow("Ghi chú", self.note)
        left = QVBoxLayout(); left.addWidget(gv, 3); left.addWidget(gs)
        gc = QGroupBox("Cấu hình môn"); lc = QVBoxLayout(gc)
        r2 = QHBoxLayout(); r2.addWidget(self.yaml_box, 1); r2.addWidget(b_yaml)
        lc.addLayout(r2); lc.addWidget(self.yaml, 1); lc.addWidget(self.yaml_state)
        right = QVBoxLayout(); right.addWidget(gc, 1); right.addWidget(self.next)
        root = QHBoxLayout(self); root.addLayout(left, 3); root.addLayout(right, 2)
        self._load_yaml()

    # ---- video ----
    def pick_video(self):
        start = str(self.ctx.source_video.parent) if self.ctx.source_video else ""
        p, _ = QFileDialog.getOpenFileName(self, "Chọn video", start, C.VIDEO_FILTER)   # D04
        if not p:
            return
        try:
            self._meta = self.app.pipeline.read_meta(p)
        except Exception as e:
            self._meta = None; self.info.setText(f"⚠ {e}"); self._validate(); return
        m = self._meta
        self.ctx.source_video = Path(p)
        self.ctx.input_video, self.ctx.session_dir = None, None       # buộc tạo phiên mới khi bấm Tiếp tục
        self.when.setDateTime(QDateTime.fromSecsSinceEpoch(int(Path(p).stat().st_mtime)))
        self.info.setText(f"{Path(p).name}  ·  {m['width']}×{m['height']}  ·  {m['fps']:.1f} fps  ·  "
                          f"{m['frames']} khung  ·  {m['frames'] / m['fps']:.1f}s")
        self.player.load(p)
        self.video_changed.emit(); self._validate()
        if m["fps"] < self.min_fps:                                                     # D06 (không chặn)
            box = QMessageBox(self); box.setIcon(QMessageBox.Warning); box.setWindowTitle("fps thấp")
            box.setText(f"fps = {m['fps']:.1f} < {self.min_fps:.0f}: các chỉ số thời gian có thể kém tin cậy.")
            keep = box.addButton("Vẫn tiếp tục", QMessageBox.AcceptRole)
            other = box.addButton("Chọn video khác", QMessageBox.RejectRole); box.exec_()
            if box.clickedButton() is other:
                self.clear_video(); self.pick_video()

    def clear_video(self):
        self.player.release()
        self.ctx.source_video, self.ctx.input_video, self.ctx.session_dir, self._meta = None, None, None, None
        self.info.setText("Chưa chọn video"); self.video_changed.emit(); self._validate()

    # ---- YAML ----
    def pick_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn YAML", str(C.SPORT_CONFIG_DIR), "YAML (*.yaml *.yml)")   # D05
        if p:
            self.yaml_box.addItem(Path(p).name, p); self.yaml_box.setCurrentIndex(self.yaml_box.count() - 1)

    def _load_yaml(self):
        p = self.yaml_box.currentData()
        if not p:
            self.yaml_ok = False; self.yaml_state.setText("✗ Chưa có file YAML"); self._validate(); return
        ok, err = self.yaml.load(p)
        errs, warns = ([err], []) if not ok else ([], [])
        if ok:
            try:
                cs = ConfigService(Path(p))
                errs, warns = cs.validate()
                self.min_fps = cs.min_fps
                if cs.cfg.sport != self.ctx.sport:
                    errs.append(f"sport='{cs.cfg.sport}' không khớp môn đã chọn '{self.ctx.sport}'")
            except Exception as e:
                errs = [str(e)]
        self.yaml_ok = not errs
        msg = "✓ Hợp lệ" if not errs else "✗ " + "; ".join(errs)
        if warns: msg += "\n⚠ " + "; ".join(warns)
        self.yaml_state.setText(msg)
        self.ctx.yaml_path = Path(p)
        self._validate()

    # ---- học viên ----
    def _autofill(self):
        code = self.code.text().strip()
        s = self.app.db.find_student(code) if code else None
        if not s:
            return
        cur = self.name.text().strip()
        if not cur:
            self.name.setText(s["full_name"])
        elif cur != s["full_name"]:                                                     # D07
            box = QMessageBox(self); box.setText(f"Mã '{code}' đã gán cho '{s['full_name']}'.")
            old = box.addButton("Dùng tên cũ", QMessageBox.AcceptRole); box.addButton("Sửa mã", QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is old: self.name.setText(s["full_name"])
            else: self.code.clear(); self.code.setFocus()

    def _validate(self, *_):
        self.next.setEnabled(bool(self._meta) and bool(self.code.text().strip())
                             and bool(self.name.text().strip()) and self.yaml_ok)

    def _proceed(self):
        c, m = self.ctx, self._meta
        c.student_code, c.student_name, c.note = self.code.text().strip(), self.name.text().strip(), self.note.text()
        c.recorded_at = self.when.dateTime().toPyDateTime()
        c.fps, c.width, c.height, c.total_frames = m["fps"], m["width"], m["height"], m["frames"]
        if c.input_video is None:                    # chỉ sao chép 1 lần cho mỗi video
            try:
                c.session_dir = file_service.create_session_dir(c.student_code)
                c.input_video = file_service.copy_video_unique(c.source_video, c.session_dir)
            except OSError as e:                     # D19-like: đĩa đầy / bị khóa
                QMessageBox.critical(self, "Lỗi ghi file", str(e)); return
        self.proceed.emit()
