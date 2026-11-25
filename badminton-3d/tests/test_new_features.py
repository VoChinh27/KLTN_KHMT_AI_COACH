import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from badminton3d.cleaning import smooth_window_frames
from badminton3d.config import load_config
from badminton3d.faults import apply_faults, follow_through, serve_faults
from badminton3d.pose_io import PersonTracker, _tiles, decode_yolov8
from badminton3d.record import next_name

CFG = load_config()


class TrackerTests(unittest.TestCase):
    A, B = (100.0, 100.0, 200.0, 400.0), (700.0, 150.0, 760.0, 330.0)      # A: người lớn (trái), B: người nhỏ (phải)

    def test_initial_choice_largest_center_and_roi(self):
        c = [(0, self.A), (1, self.B)]
        self.assertEqual(PersonTracker(1000, 600, "largest").pick(c), 0)
        self.assertEqual(PersonTracker(1000, 600, "center").pick([(0, self.A), (1, (480, 200, 540, 380))]), 1)
        self.assertEqual(PersonTracker(1000, 600, "largest", roi=[0.6, 0, 1, 1]).pick(c), 1)       # ROI thắng "largest"

    def test_follows_target_even_if_order_swaps(self):
        t = PersonTracker(1000, 600, "largest", roi=[0.6, 0, 1, 1])
        self.assertEqual(t.pick([(0, self.A), (1, self.B)]), 1)
        moved = (705.0, 152.0, 766.0, 335.0)
        self.assertEqual(t.pick([(0, moved), (1, self.A)]), 0)                                   # chỉ số đổi chỗ nhưng vẫn bám đúng người
        self.assertEqual(t.switches, 0)

    def test_lost_then_reacquire_counts_switch(self):
        t = PersonTracker(1000, 600, "largest", lost_frames=3)
        t.pick([(0, self.A)])
        for _ in range(3):
            self.assertIsNone(t.pick([(0, self.B)]))                                              # chỉ còn người khác: chưa vội đổi
        self.assertEqual(t.pick([(0, self.B)]), 0)                                               # quá lost_frames: chọn lại
        self.assertEqual(t.switches, 1)

    def test_no_candidates(self):
        self.assertIsNone(PersonTracker(1000, 600).pick([]))


class DetectorHelpersTests(unittest.TestCase):
    def test_tiles_cover_frame(self):
        for w, h in ((1920, 540), (1080, 1920), (960, 960), (1280, 720)):
            t = _tiles(w, h)
            side = min(w, h)
            self.assertTrue(all(s == side for _, _, s in t))
            cover = np.zeros((h, w), bool)
            for x, y, s in t:
                cover[y:y + s, x:x + s] = True
            self.assertTrue(cover.all(), (w, h))
        self.assertEqual(len(_tiles(960, 960)), 1)

    def test_decode_yolov8_person_boxes_and_nms(self):
        n = 6
        out = np.zeros((1, 84, n), np.float32)
        out[0, :4, 0] = [320, 320, 100, 200]; out[0, 4, 0] = 0.90        # người rõ
        out[0, :4, 1] = [322, 322, 100, 200]; out[0, 4, 1] = 0.80        # trùng người trên -> bị NMS loại
        out[0, :4, 2] = [100, 100, 40, 80]; out[0, 4, 2] = 0.10          # điểm thấp -> loại
        boxes = decode_yolov8(out, scale=0.5, pad=(0, 0), conf=0.35)
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1, sc = boxes[0]
        self.assertAlmostEqual(x0, 540.0, delta=1)                       # (320-50)/0.5
        self.assertAlmostEqual(y1, 840.0, delta=1)                       # (320+100)/0.5
        self.assertAlmostEqual(sc, 0.9, places=3)

    def test_decode_yolov8_none(self):
        self.assertEqual(decode_yolov8(np.zeros((1, 84, 10), np.float32), 1.0, (0, 0)), [])


class SmoothingFpsTests(unittest.TestCase):
    def test_window_follows_time_not_frames(self):
        q = CFG["quality"]
        self.assertEqual(smooth_window_frames(q, 30), 7)
        self.assertEqual(smooth_window_frames(q, 60), 15)
        self.assertEqual(smooth_window_frames(q, 24), 7)
        self.assertEqual(smooth_window_frames(q, 15), 5)
        self.assertEqual(smooth_window_frames({"smooth_poly": 2, "smooth_window": 9}, 120), 9)      # null smooth_window_s: dùng số frame cũ


def fake_ctx(wrist_above_shoulder_m: float, n: int = 90, fps: float = 30.0):
    """Ngữ cảnh giả: cổ tay phải cao hơn vai `wrist_above_shoulder_m` sau cú chạm (frame 40)."""
    w = {j: np.zeros((n, 3)) for j in ("right_wrist", "right_shoulder")}
    w["right_wrist"][:, 1] = 0.3                                          # y XUỐNG: 0,3 = thấp hơn vai
    w["right_wrist"][45:60, 1] = -wrist_above_shoulder_m
    xy = {j: np.zeros((n, 2)) for j in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")}
    rep = SimpleNamespace(peak=40, end=60, swing_start=30, prep_start=5)
    return SimpleNamespace(pose=SimpleNamespace(fps=fps, n=n, foot=None), arm={"side": "right"}, clean=SimpleNamespace(w=w, xy=xy), serve_seg=SimpleNamespace(reps=[rep]))


class FaultTests(unittest.TestCase):
    def test_follow_through_measure(self):
        v, fr = follow_through(fake_ctx(0.40), fake_ctx(0.4).serve_seg.reps[0], CFG)
        self.assertAlmostEqual(v, 0.40, places=3)
        self.assertEqual(fr[0], 40)

    def test_levels_and_penalty(self):
        for h, sev, pen in ((0.05, None, 0), (0.20, "nhẹ", 4), (0.40, "nặng", 10)):
            f, m = serve_faults(fake_ctx(h), CFG)
            if sev is None:
                self.assertEqual(f[0], [])
            else:
                self.assertEqual((f[0][0]["severity"], f[0][0]["penalty"]), (sev, pen))
            self.assertAlmostEqual(m[0]["follow_through_above_shoulder_m"], h, places=3)

    def test_heel_check_skipped_without_foot_data(self):
        _, m = serve_faults(fake_ctx(0.0), CFG)
        self.assertIsNone(m[0]["heel_lift_ratio"])

    def test_disabled_check(self):
        import copy
        cfg = copy.deepcopy(CFG)
        cfg["faults"]["follow_through"]["enabled"] = False
        f, _ = serve_faults(fake_ctx(0.5), cfg)
        self.assertEqual(f[0], [])

    def test_apply_faults_reduces_score_and_lists_error(self):
        sc = {"overall": 90.0, "grade": "Tốt", "errors": [{"feature": "knee_mean", "lost_points": 1.0, "potential_gain": 1.0}],
              "units": [{"id": "serve_1", "score": 90.0}]}
        f, m = serve_faults(fake_ctx(0.40), CFG)
        out = apply_faults(sc, f, m, CFG)
        self.assertAlmostEqual(out["overall"], 80.0)
        self.assertEqual(out["grade"], "Khá")
        self.assertEqual(out["errors"][0]["feature"], "follow_through")                         # lỗi mất nhiều điểm nhất lên đầu
        self.assertEqual(out["units"][0]["score_before_faults"], 90.0)


class RecordNameTests(unittest.TestCase):
    def test_next_name_increments(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(next_name(d, "trainee", "serve"), "trainee_serve_live01")
            for n in ("trainee_serve_live01.mp4", "trainee_serve_live02.json", "trainee_ready_live07.mp4"):
                open(os.path.join(d, n), "w").close()
            self.assertEqual(next_name(d, "trainee", "serve"), "trainee_serve_live03")
            self.assertEqual(next_name(d, "trainee", "ready"), "trainee_ready_live08")


if __name__ == "__main__":
    unittest.main()
