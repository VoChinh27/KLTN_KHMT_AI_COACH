"""Smoke test toàn luồng GUI (offscreen) với pipeline giả: chọn môn -> video -> pha -> chấm -> kết quả -> báo cáo -> lịch sử."""
import json
import time
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from gui_app import constants as C
from gui_app.tests import fakes

YAML = """sport: sprint_start
phases: [static, sprint_start]
cyclic: {static: false, sprint_start: false}
"""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "config" / "sport_configs").mkdir(parents=True)
    (tmp_path / "config" / "sport_configs" / "sprint_start.yaml").write_text(YAML, encoding="utf-8")
    (tmp_path / "models" / "phase_segmenter").mkdir(parents=True)
    (tmp_path / "models" / "phase_segmenter" / "registry.json").write_text(json.dumps({"current": "v1", "versions": {"v1": {}}}))
    for k, v in dict(ROOT=tmp_path, APP_DATA=tmp_path / "app_data", SESSIONS_DIR=tmp_path / "app_data" / "sessions",
                     DB_PATH=tmp_path / "app_data" / "t.db", SYSTEM_CONFIG=tmp_path / "config" / "system_config.yaml",
                     SPORT_CONFIG_DIR=tmp_path / "config" / "sport_configs",
                     PHASE_MODEL_DIR=tmp_path / "models" / "phase_segmenter",
                     PHASE_FEEDBACK_LOG=tmp_path / "log.jsonl").items():
        monkeypatch.setattr(C, k, v)
    video = tmp_path / "clip.mp4"; fakes.make_video(str(video))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(video), "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path / "out")))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "o.mp4"), "")))
    for n in ("information", "warning", "question"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    return tmp_path


def wait(cond, timeout=20):
    app = QApplication.instance(); t = time.time()
    while not cond():
        app.processEvents(); time.sleep(0.01)
        from gui_app.tests.conftest import ERRORS
        assert not ERRORS, ERRORS[-1]
        assert time.time() - t < timeout, "timeout"


def test_full_flow(env):
    app = QApplication.instance() or QApplication([])
    from gui_app.app_context import AppContext
    from gui_app.main import MainWindow
    win = MainWindow(AppContext()); win.show()
    infos = win.ctx.registry.scan()
    assert any(i.ready for i in infos) and any(i.badge == "Chưa có mô hình" for i in infos)

    win._open_sport("sprint_start"); sp = win.sport
    assert not sp.isTabEnabled(1)
    sp.up.pick_video(); sp.up.code.setText("HV01"); sp.up.name.setText("Nguyễn A")
    assert sp.up.next.isEnabled()
    sp.up.next.click(); assert sp.currentIndex() == 1 and sp.ctx.input_video.exists()

    sp.ph.extract(); wait(lambda: sp.ph.review is not None)
    assert sp.ph.b_ok.isEnabled() and sp.ph.table.rowCount() == 4
    sp.ph.sel.setCurrentIndex(1); sp.ph._move(1); assert sp.ph.review.edits == 1
    sp.ph.ok(); wait(lambda: sp.ctx.result is not None)
    assert sp.currentIndex() == 2 and sp.ctx.db_session_id
    assert len(sp.ctx.pauses) == 1 and sp.ctx.pauses[0].frame == 20       # khung giữa pha 10..30
    assert any("Cổ chân" in q for q in sp.ctx.quality)                    # measurement_error -> cảnh báo chất lượng
    sp.rs.player.seek(20); sp.rs._next_error(); assert sp.rs.card.isVisible()
    sp.rs.player.auto_pause = True; sp.rs.player.seek(19); sp.rs.player.play()
    wait(lambda: not sp.rs.player.is_playing(), 5)
    assert sp.rs.player.idx == 20                                          # tự dừng đúng khung lỗi

    sp.rs.save_video(); wait(lambda: sp.ctx.overlay_path is not None)
    assert Path(sp.ctx.overlay_path).exists()

    sp.rs.next_report.emit(); sp.rp.export(); wait(lambda: sp.ctx.report_pdf is not None)
    assert Path(sp.ctx.report_pdf).stat().st_size > 500 and Path(sp.ctx.report_txt).exists()

    win._open_history(); assert win.history.table.rowCount() == 1
    win.history.table.selectRow(0); assert win.history._cur()["report_pdf_path"]
    sp.cancel_all(); win.sport.up.player.release(); win.sport.ph.player.release(); win.sport.rs.player.release()


def test_manual_mode_when_no_model(env, monkeypatch):
    app = QApplication.instance() or QApplication([])
    from gui_app.app_context import AppContext
    from gui_app.main import MainWindow
    from gui_app.services.pipeline_service import PipelineService
    monkeypatch.setattr(PipelineService, "propose_phases",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("registry.json")))
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[0] if self.buttons() else None, raising=False)
    from gui_app.tabs import phase_tab
    monkeypatch.setattr(phase_tab.PhaseTab, "_ask_manual", lambda self, err: True)
    win = MainWindow(AppContext()); win.show(); win._open_sport("sprint_start"); sp = win.sport
    sp.up.pick_video(); sp.up.code.setText("HV02"); sp.up.name.setText("B"); sp.up.next.click()
    sp.ph.extract(); wait(lambda: sp.ph.review is not None)
    assert sp.ctx.manual_mode and "THỦ CÔNG" in sp.ph.mode.text()
    sp.ph.sel.setCurrentIndex(2); sp.ph.player.seek(35); sp.ph._set_here()
    assert sp.ph.review.b[2] == 35
    sp.ph._set_absent(3, True); assert sp.ph.review.is_absent(3)
    for p in (sp.up.player, sp.ph.player, sp.rs.player): p.release()
