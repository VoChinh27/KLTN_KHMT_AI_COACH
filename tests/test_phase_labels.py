"""Kiểm thử stage3_features/phase_labels.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core.phase_schema import PhaseSet
from stage3_features.phase_model.core.phase_labels import (rep_to_frame_labels, frame_labels_to_rep,
                                          CLASS_TO_ID, IGNORE_LABEL, class_frequencies)
from tests.test_phase_schema import V1_SAMPLE

def test_labels_roundtrip_sample():
    ps = PhaseSet.from_dict(V1_SAMPLE)
    lab = rep_to_frame_labels(ps.reps[0], ps.total_frames)
    assert len(lab) == 743 and lab[0] == CLASS_TO_ID["outside_pre"] and lab[539] == 0 and lab[540] == 1
    assert lab[600] == 1 and lab[601] == 2 and lab[708] == 3 and lab[734] == 4 and lab[735] == 5 and lab[-1] == 5
    assert (lab != IGNORE_LABEL).all()
    rep2 = frame_labels_to_rep(lab)
    assert {p: (w.start_frame, w.end_frame) for p, w in rep2.phases.items()} == \
           {p: (w.start_frame, w.end_frame) for p, w in ps.reps[0].phases.items()}
    assert class_frequencies(lab)[0] > 0.7   # outside_pre chiếm ~72%: cần class weighting

def test_absent_leaves_ignore_gap_when_not_adjacent():
    from core.phase_schema import Rep
    rep = Rep.from_windows({"vao_cho": (10, 50), "static": (60, 90), "sprint_start": None, "drive_phase": (95, 120)})
    lab = rep_to_frame_labels(rep, 200)
    assert (lab[51:60] == IGNORE_LABEL).all() and (lab[91:95] == IGNORE_LABEL).all()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK", n)
