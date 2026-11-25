import unittest

import numpy as np

from badminton3d.angles import bilateral, choose_arm, angle3, ready_features, trunk_lean, stance_ratio, serve_features
from badminton3d.cleaning import clean_pose, fill_short_gaps, smooth_segments
from badminton3d.config import load_config
from badminton3d.pose_io import PoseData
from tests.helpers import burst, make_df

CFG = load_config()


def clean(df, **kw):
    return clean_pose(PoseData.from_df(df), CFG, **kw)


class CleaningTests(unittest.TestCase):
    def test_short_gap_is_interpolated_long_gap_is_not(self):
        n = 100
        m = np.zeros(n, bool)
        m[20:23] = True                       # 3 frame = 0,1 s  (<= 0,25 s)
        m[50:65] = True                       # 15 frame = 0,5 s (> 0,25 s)
        c = clean(make_df(n, hide={"left_knee": m}), smooth=False)
        self.assertTrue(c.valid["left_knee"][20:23].all())
        self.assertTrue(c.interpolated["left_knee"][20:23].all())
        self.assertFalse(c.valid["left_knee"][50:65].any())
        self.assertAlmostEqual(c.rel["left_knee"][21], 0.7 * 0.95, places=3)    # độ tin cậy nội suy giảm

    def test_no_interpolation_at_clip_edges(self):
        m = np.zeros(60, bool)
        m[0:3] = True
        c = clean(make_df(60, hide={"left_knee": m}), smooth=False)
        self.assertFalse(c.valid["left_knee"][0:3].any())

    def test_fill_short_gaps_helper(self):
        v = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], bool)
        fill, L, R = fill_short_gaps(v, 2)
        self.assertTrue(fill[2] and fill[3])
        self.assertFalse(fill[6:11].any())

    def test_smooth_keeps_nan_gap(self):
        x = np.sin(np.linspace(0, 6, 80))
        x[30:40] = np.nan
        y = smooth_segments(x, 7, 2)
        self.assertTrue(np.isnan(y[30:40]).all() and not np.isnan(y[:30]).any())


class AngleTests(unittest.TestCase):
    def test_knee_angle_matches_geometry(self):
        """Hông (-0,10; 0; 0) – gối (-0,10; 0,45; 0,05) – cổ chân (-0,12; 0,90; 0): tính tay = 167,0°."""
        c = clean(make_df(40))
        v, r = angle3(c, "left_hip", "left_knee", "left_ankle")
        self.assertAlmostEqual(float(np.nanmedian(v)), 167.0, delta=0.6)
        self.assertGreater(np.nanmin(r), 0.9)

    def test_trunk_lean_vertical_and_forward(self):
        c0 = clean(make_df(40))
        self.assertLess(np.nanmedian(trunk_lean(c0)[0]), 1.0)
        shift = {"left_shoulder": (-0.18, -0.50, 0.2), "right_shoulder": (0.18, -0.50, 0.2)}          # vai lệch 0,2 m ra trước
        c1 = clean(make_df(40, override=shift))
        self.assertAlmostEqual(float(np.nanmedian(trunk_lean(c1)[0])), np.degrees(np.arctan2(0.2, 0.5)), delta=0.5)

    def test_stance_ratio(self):
        c = clean(make_df(40))
        self.assertAlmostEqual(float(np.nanmedian(stance_ratio(c)[0])), 0.24 / 0.36, places=2)

    def test_bilateral_substitutes_missing_side(self):
        m = np.ones(60, bool)
        c = clean(make_df(60, hide={"left_knee": m}), smooth=False)
        f = ready_features(c, CFG)["knee_mean"]
        self.assertFalse(np.isnan(f["value"]).any())                       # vẫn có giá trị nhờ bên phải
        self.assertTrue(f["substituted"].all())
        self.assertLess(f["rel"].max(), 0.95 * 0.7 + 1e-6)                 # độ tin cậy bị giảm khi phải thay thế

    def test_both_sides_missing_gives_not_valid(self):
        m = np.ones(60, bool)
        c = clean(make_df(60, hide={"left_knee": m, "right_knee": m}), smooth=False)
        self.assertTrue(np.isnan(ready_features(c, CFG)["knee_mean"]["value"]).all())

    def test_valid_per_angle_is_not_all_or_nothing(self):
        """Mất cổ tay trái KHÔNG làm hỏng góc gối: đây chính là lý do tính Valid theo từng góc."""
        m = np.ones(60, bool)
        c = clean(make_df(60, hide={"left_wrist": m}), smooth=False)
        self.assertFalse(np.isnan(angle3(c, "left_hip", "left_knee", "left_ankle")[0]).any())
        self.assertTrue(np.isnan(angle3(c, "left_shoulder", "left_elbow", "left_wrist")[0]).all())


class ArmChoiceTests(unittest.TestCase):
    def test_both_visible_picks_most_active(self):
        n = 120
        c = clean(make_df(n, wrist={"right": burst(n, 40, 12, 0.5)}))
        self.assertEqual(choose_arm(c, "auto")["side"], "right")
        self.assertEqual(choose_arm(c, "auto")["mode"], "most_active_arm")

    def test_one_arm_hidden_uses_visible_arm(self):
        n = 120
        hide = {j: np.ones(n, bool) for j in ("left_shoulder", "left_elbow", "left_wrist")}
        c = clean(make_df(n, hide=hide))
        a = choose_arm(c, "auto")
        self.assertEqual((a["side"], a["mode"]), ("right", "best_visible"))

    def test_declared_racket_hand_hidden_is_substituted(self):
        n = 120
        hide = {j: np.ones(n, bool) for j in ("left_shoulder", "left_elbow", "left_wrist")}
        a = choose_arm(clean(make_df(n, hide=hide)), "left")
        self.assertEqual((a["side"], a["mode"], a["substituted"]), ("right", "best_visible_substitute", True))

    def test_declared_racket_hand_visible_is_used(self):
        a = choose_arm(clean(make_df(60)), "left")
        self.assertEqual((a["side"], a["mode"]), ("left", "racket_arm"))

    def test_serve_features_use_chosen_arm(self):
        n = 60
        c = clean(make_df(n))
        arm = choose_arm(c, "right")
        f = serve_features(c, CFG, arm)
        self.assertEqual(set(f), {"elbow_arm", "shoulder_arm", "wrist_height", "wrist_reach", "knee_mean", "hip_mean", "trunk_lean"})


if __name__ == "__main__":
    unittest.main()
