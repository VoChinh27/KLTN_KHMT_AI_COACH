import unittest

import numpy as np

from badminton3d.analyze import analyze_context, build_context
from badminton3d.config import load_config
from badminton3d.pose_io import PoseData
from badminton3d.cleaning import clean_pose
from badminton3d.reference import build_reference
from badminton3d.scoring import feature_score, score_features, grade
from badminton3d.segment import READY_STATES, segment_ready, segment_serve
from tests.helpers import burst, make_df

CFG = load_config()
FPS = 30.0


def clean(df):
    return clean_pose(PoseData.from_df(df), CFG)


class ReadySegmentTests(unittest.TestCase):
    def test_moving_then_still_scores_only_after_settling(self):
        n = 240
        x = np.zeros(n)
        i = np.arange(n)
        x[:90] = 250 * np.sin(2 * np.pi * i[:90] / 30.0)          # 3 giây di chuyển mạnh (đỡ cầu), tốc độ đỉnh ~2 m/s
        x[90:] = x[89]
        sg = segment_ready(clean(make_df(n, hip_dx=x)), CFG)
        self.assertEqual(len(sg.episodes), 1)
        a, b = sg.episodes[0]
        self.assertGreaterEqual(a, 90 + int(0.3 * FPS) - 3)          # chỉ chấm SAU khi ổn định lại
        self.assertTrue((sg.state[:80] == READY_STATES["MOVING"]).mean() > 0.9)

    def test_continuous_movement_has_no_episode(self):
        n = 200
        x = 250 * np.sin(2 * np.pi * np.arange(n) / 30.0)
        self.assertEqual(segment_ready(clean(make_df(n, hip_dx=x)), CFG).episodes, [])

    def test_short_calm_is_not_scored(self):
        n = 200
        base = 250 * np.sin(2 * np.pi * np.arange(n) / 30.0)
        x = base.copy()
        x[100:118] = base[100]                                         # chỉ đứng yên ~0,6 s, trong đó 0,3 s đầu là ổn định lại
        x[118:] = base[118:] - base[118] + base[100]                   # rồi lại di chuyển (liền mạch, không nhảy vọt)
        self.assertEqual(segment_ready(clean(make_df(n, hip_dx=x)), CFG).episodes, [])


class ServeSegmentTests(unittest.TestCase):
    def test_burst_after_prep_is_one_rep_with_prep(self):
        n = 150
        c = clean(make_df(n, name="syn_serve_01", wrist={"right": burst(n, 70, 10, 0.5)}))
        reps = segment_serve(c, CFG).reps
        self.assertEqual(len(reps), 1)
        r = reps[0]
        self.assertIn("chuan_bi", r.phases)
        self.assertGreaterEqual((r.swing_start - r.prep_start) / FPS, 0.29)

    def test_burst_without_preparation_is_not_a_serve_in_strict_mode(self):
        import copy
        cfg = copy.deepcopy(CFG)
        cfg["serve"]["fallback_without_prep"] = False                                           # chế độ nghiêm: bắt buộc có bước chuẩn bị
        n = 120
        c = clean(make_df(n, name="syn_serve_01", wrist={"right": burst(n, 4, 10, 0.5)}))      # vung ngay từ đầu, không có chuẩn bị
        self.assertEqual(segment_serve(c, cfg).reps, [])

    def test_trimmed_clip_falls_back_to_strong_swing_and_drops_fake_prep(self):
        """Clip cắt sát (vung ngay từ đầu): vẫn lấy cú vung mạnh, đánh dấu no_prep và KHÔNG có pha 'chuan_bi' giả."""
        n = 120
        c = clean(make_df(n, name="syn_serve_01", wrist={"right": burst(n, 4, 10, 0.5)}))
        reps = segment_serve(c, CFG).reps
        self.assertEqual(len(reps), 1)
        self.assertTrue(reps[0].no_prep)
        self.assertNotIn("chuan_bi", reps[0].phases)

    def test_full_serve_is_not_flagged_no_prep(self):
        n = 150
        c = clean(make_df(n, name="syn_serve_01", wrist={"right": burst(n, 70, 10, 0.5)}))
        self.assertFalse(segment_serve(c, CFG).reps[0].no_prep)

    def test_picking_up_shuttle_before_prep_is_excluded(self):
        """Thân di chuyển mạnh trong 1,5 giây đầu (nhặt cầu / đi lại), sau đó đứng yên chuẩn bị rồi phát: chỉ chấm từ lúc chuẩn bị."""
        n = 240
        x = np.zeros(n)
        x[:45] = 250 * np.sin(2 * np.pi * np.arange(45) / 22.0)
        x[45:] = x[44]
        c = clean(make_df(n, name="syn_serve_01", hip_dx=x, wrist={"right": burst(n, 120, 10, 0.5)}))
        reps = segment_serve(c, CFG).reps
        self.assertEqual(len(reps), 1)
        self.assertGreater(reps[0].prep_start, 45 - 3)                 # phần nhặt cầu nằm ngoài vùng chấm

    def test_no_motion_no_rep(self):
        self.assertEqual(segment_serve(clean(make_df(120, name="syn_serve_01")), CFG).reps, [])


def stats(v, rel=0.95):
    return {"value": v, "rel": rel, "coverage": 1.0, "substituted": 0.0, "n_frames": 30}


def ref_feats(center=130.0, tol=10.0, level="PRIMARY"):
    return {"knee_mean": {"center": center, "tol": tol, "level": level, "unit": "°", "label": "Góc gối"},
            "hip_mean": {"center": 120.0, "tol": 10.0, "level": "SECONDARY", "unit": "°", "label": "Góc hông"}}


class ScoringTests(unittest.TestCase):
    def test_score_curve(self):
        d = CFG["scoring"]["decay_factor"]
        self.assertEqual(feature_score(130, 130, 10, d), 100.0)
        self.assertEqual(feature_score(140, 130, 10, d), 100.0)                       # sát biên dung sai vẫn đạt
        self.assertAlmostEqual(feature_score(140 + 0.75 * 10, 130, 10, d), 100 * (1 - 0.75 / d))
        self.assertEqual(feature_score(130 + 10 * (1 + d) + 1, 130, 10, d), 0.0)

    def test_deviation_direction_and_status(self):
        r = score_features({"knee_mean": stats(150), "hip_mean": stats(120)}, ref_feats(), CFG)
        by = {f["feature"]: f for f in r["features"]}
        self.assertEqual(by["knee_mean"]["status"], "HIGH")
        self.assertEqual(by["hip_mean"]["status"], "OK")
        self.assertLess(r["score"], 100)

    def test_low_reliability_is_not_assessed_and_not_penalised(self):
        r = score_features({"knee_mean": stats(170, rel=0.2), "hip_mean": stats(120)}, ref_feats(), CFG)
        by = {f["feature"]: f for f in r["features"]}
        self.assertEqual(by["knee_mean"]["status"], "NOT_ASSESSED")
        self.assertEqual(r["score"], 100.0)                                              # không đo được thì KHÔNG phạt

    def test_reliability_weights_the_average(self):
        hi = score_features({"knee_mean": stats(170, rel=0.95), "hip_mean": stats(120, rel=0.95)}, ref_feats(), CFG)["score"]
        lo = score_features({"knee_mean": stats(170, rel=0.45), "hip_mean": stats(120, rel=0.95)}, ref_feats(), CFG)["score"]
        self.assertGreater(lo, hi)                                                       # góc sai nhưng kém tin cậy thì ảnh hưởng ít hơn

    def test_grade_bands(self):
        self.assertEqual(grade(90, CFG), "Tốt")
        self.assertEqual(grade(72, CFG), "Khá")
        self.assertEqual(grade(55, CFG), "Cần cải thiện")
        self.assertEqual(grade(10, CFG), "Chưa đạt")
        self.assertEqual(grade(None, CFG), "Không đủ dữ liệu")


class ReferenceTests(unittest.TestCase):
    def _units(self, knee):
        feats = {k: stats(v) for k, v in {"knee_mean": knee, "hip_mean": 120.0, "elbow_mean": 110.0, "shoulder_mean": 60.0, "trunk_lean": 25.0, "stance_ratio": 1.5}.items()}
        return [{"id": "R1", "frames": (0, 100), "duration_s": 3.3, "features": feats}]

    def test_single_video_uses_floor_times_factor(self):
        ref = build_reference("ready", {"a": self._units(130.0)}, CFG)
        fl = CFG["features"]["ready"]["knee_mean"]["floor"]
        self.assertAlmostEqual(ref["features"]["knee_mean"]["tol"], fl * CFG["scoring"]["single_video_floor_factor"])
        self.assertEqual(ref["features"]["knee_mean"]["center"], 130.0)

    def test_tolerance_covers_spread_between_coaches(self):
        ref = build_reference("ready", {"a": self._units(120.0), "b": self._units(150.0)}, CFG)
        r = ref["features"]["knee_mean"]
        self.assertEqual(r["center"], 135.0)
        self.assertGreaterEqual(r["tol"], 15.0)                                          # nửa khoảng dao động 30° / 2
        self.assertEqual(r["spread"], [120.0, 150.0])


class EndToEndSyntheticTests(unittest.TestCase):
    def test_correct_posture_scores_high_bent_knee_ranks_first(self):
        n = 150
        good = build_context(make_df(n, name="coach_ready_01"), CFG, "ready")
        self.assertEqual(len(good.units), 1)
        ref = build_reference("ready", {"coach_ready_01": good.units}, CFG)
        same = analyze_context(good, CFG, ref, "coach_ready_01")
        self.assertGreater(same["score"]["overall"], 99)
        # gối gập sâu hơn chuẩn nhiều (đẩy gối ra trước 0,25 m)
        bent = build_context(make_df(n, name="trainee_ready_01", override={"left_knee": (-0.10, 0.45, 0.30), "right_knee": (0.10, 0.45, 0.30)}), CFG, "ready")
        res = analyze_context(bent, CFG, ref, "trainee_ready_01")
        self.assertLess(res["score"]["overall"], same["score"]["overall"])
        top = res["score"]["errors"][0]
        self.assertEqual(top["feature"], "knee_mean")
        self.assertEqual(top["direction"], "thấp hơn chuẩn")
        self.assertGreater(top["potential_gain"], 0)
        self.assertIn("gối", top["advice"].lower())

    def test_no_reference_status(self):
        ctx = build_context(make_df(150, name="coach_ready_01"), CFG, "ready")
        self.assertEqual(analyze_context(ctx, CFG, None, "coach_ready_01")["status"], "NO_REFERENCE")

    def test_moving_video_has_no_units(self):
        n = 150
        x = 250 * np.sin(2 * np.pi * np.arange(n) / 30.0)
        ctx = build_context(make_df(n, name="coach_ready_01", hip_dx=x), CFG, "ready")
        ref = build_reference("ready", {"a": build_context(make_df(150, name="a"), CFG, "ready").units}, CFG)
        res = analyze_context(ctx, CFG, ref, "coach_ready_01")
        self.assertEqual(res["status"], "NO_UNITS")
        self.assertIn("di chuyển", res["message"])


if __name__ == "__main__":
    unittest.main()


class CriticalCapTests(unittest.TestCase):
    """Một góc PRIMARY lệch quá xa chuẩn thì điểm đơn vị không thể vượt critical_cap (dù các góc còn lại đúng)."""
    REF = {
        "wrist_height": {"center": 0.4, "tol": 0.1, "level": "PRIMARY", "unit": "m", "label": "cao cổ tay"},
        "elbow_arm": {"center": 130.0, "tol": 15.0, "level": "SECONDARY", "unit": "°", "label": "khuỷu"},
        "knee_mean": {"center": 150.0, "tol": 10.0, "level": "PRIMARY", "unit": "°", "label": "gối"},
        "trunk_lean": {"center": 8.0, "tol": 7.5, "level": "PRIMARY", "unit": "°", "label": "thân"},
    }

    @staticmethod
    def stats(**v):
        base = {"wrist_height": 0.4, "elbow_arm": 130.0, "knee_mean": 150.0, "trunk_lean": 8.0}
        base.update(v)
        return {k: {"value": x, "rel": 1.0, "coverage": 1.0, "substituted": 0.0} for k, x in base.items()}

    def test_perfect_is_100_no_critical(self):
        r = score_features(self.stats(), self.REF, CFG)
        self.assertAlmostEqual(r["score"], 100.0)
        self.assertEqual(r["critical"], [])

    def test_far_primary_angle_caps_score_even_if_others_perfect(self):
        r = score_features(self.stats(wrist_height=0.9), self.REF, CFG)       # lệch 0,5 m > 2,5 x 0,1
        self.assertEqual(r["critical"], ["wrist_height"])
        self.assertGreater(r["raw_score"], 70.0)                              # trung bình thô vẫn cao...
        self.assertLessEqual(r["score"], CFG["scoring"]["critical_cap"])      # ...nhưng bị giới hạn dưới ngưỡng đạt
        self.assertLess(r["score"], 70.0)

    def test_moderate_deviation_is_not_critical(self):
        r = score_features(self.stats(wrist_height=0.58), self.REF, CFG)      # lệch 0,18: ngoài dung sai nhưng chưa quá (1+decay) x tol
        self.assertEqual(r["critical"], [])
        self.assertAlmostEqual(r["score"], r["raw_score"])

    def test_secondary_angle_never_critical(self):
        r = score_features(self.stats(elbow_arm=60.0), self.REF, CFG)
        self.assertEqual(r["critical"], [])
