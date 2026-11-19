from types import SimpleNamespace as NS

import numpy as np

from gui_app.services.phase_review_service import PhaseReviewService


def _svc():
    prop = NS(boundaries=[0, 100, 200, 220, 400], boundary_conf=np.array([1, .9, .5, .8, 1.0]),
              auto_meta=lambda: {"flags": []})
    return PhaseReviewService(prop, "v.mp4", 30.0, 400)


def test_move_clamped_and_undo():
    s = _svc(); s.move_boundary(1, 500)
    assert s.b[1] == 200 and s.edits == 1
    s.undo(); assert s.b[1] == 100


def test_windows_inclusive_end():
    assert _svc().windows()["static"] == (100, 199)


def test_low_conf():
    assert _svc().low_conf_boundaries() == [2]
