import json
import tempfile
import unittest
from pathlib import Path

import yaml

from badminton3d import sports as S
from badminton3d import ui_backend as B


class FakeEngine:
    """Engine giả: chứng minh giao diện chạy được với môn khác cầu lông chỉ nhờ khai báo trong sports.yaml."""
    name = "fake"
    supported_actions = ("forehand",)
    quick_settings = [{"title": "Điểm", "items": [{"keys": ["scoring", "pass_score"], "label": "Điểm đạt", "kind": "float", "min": 0.0, "max": 100.0, "step": 1.0}]}]

    def analyze(self, sport, video, action, hand="auto", force=False):
        d = sport.output / Path(video).stem
        d.mkdir(parents=True, exist_ok=True)
        rep = {"status": "SCORED", "exercise": action, "duration_s": 2.0, "warnings": [], "score": {"overall": 77.0, "grade": "Khá", "errors": []}}
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    def build_reference(self, sport, action, videos):
        sport.reference_file(action).write_text(json.dumps({"built_from": [v.name for v in videos], "n_videos": len(videos)}), encoding="utf-8")
        return 0, "ok"


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_registry_is_badminton_legacy_layout(self):
        sp = S.list_sports(self.root)
        self.assertEqual(list(sp), ["badminton"])
        b = sp["badminton"]
        self.assertEqual(list(b.actions), ["ready", "serve"])
        self.assertEqual((b.input, b.reference, b.output, b.config), (self.root / "Input", self.root / "Reference", self.root / "Output", self.root / "config/badminton3d.yaml"))
        self.assertTrue(b.supports("serve"))
        self.assertTrue(b.actions["serve"].hand and not b.actions["ready"].hand)
        self.assertEqual(S.default_sport_key(self.root), "badminton")

    def test_create_sport_builds_folders_config_and_registry(self):
        sp = S.create_sport("tennis", "Quần vợt", "🎾", S.parse_actions("forehand | Cú thuận tay\nbackhand | Cú trái tay"), self.root)
        for d in ("Input/tennis", "Reference/tennis", "Output/tennis"):
            self.assertTrue((self.root / d).is_dir(), d)
        cfg = yaml.safe_load((self.root / "config/tennis.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["sport"]["key"], "tennis")
        self.assertEqual(list(cfg["actions"]), ["forehand", "backhand"])
        self.assertEqual(cfg["scoring"]["pass_score"], 70)
        self.assertEqual((sp.input, sp.reference, sp.output), (self.root / "Input/tennis", self.root / "Reference/tennis", self.root / "Output/tennis"))
        self.assertIsNone(sp.engine)
        self.assertEqual(sp.label("forehand"), "Cú thuận tay")
        self.assertIn("tennis", S.list_sports(self.root))
        self.assertIn("badminton", S.list_sports(self.root))                                   # môn cũ không bị đụng

    def test_validation(self):
        acts = [("a1", "A")]
        for bad in ("1abc", "a", "có dấu", "x" * 30, "input", "_tong_hop", ""):
            with self.assertRaises(S.SportError, msg=bad):
                S.create_sport(bad, "Tên", "🏅", acts, self.root)
        with self.assertRaises(S.SportError):
            S.create_sport("tennis", "", "🏅", acts, self.root)
        with self.assertRaises(S.SportError):
            S.create_sport("tennis", "Tennis", "🏅", [], self.root)
        self.assertEqual(S.create_sport("TENNIS", "Tennis", "🎾", acts, self.root).key, "tennis")      # chữ hoa được chuẩn hóa thành chữ thường
        with self.assertRaises(S.SportError):
            S.create_sport("tennis", "Tennis 2", "🎾", acts, self.root)
        with self.assertRaises(S.SportError):
            S.parse_actions("a1 | A\na1 | B")

    def test_rollback_when_config_exists(self):
        S.load_registry(self.root)
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config/gym.yaml").write_text("x: 1\n", encoding="utf-8")
        with self.assertRaises(S.SportError):
            S.create_sport("gym", "Gym", "🏋️", [("squat", "Squat")], self.root)
        self.assertNotIn("gym", S.list_sports(self.root))
        self.assertFalse((self.root / "Input/gym").exists())
        self.assertEqual((self.root / "config/gym.yaml").read_text(encoding="utf-8"), "x: 1\n")      # không xóa file có sẵn của người dùng

    def test_add_action_updates_registry_and_config(self):
        S.create_sport("run", "Chạy bộ", "🏃", [("sprint", "Chạy nước rút")], self.root)
        sp = S.add_action("run", "stride", "Sải chân", hand=False, root=self.root)
        self.assertEqual(list(sp.actions), ["sprint", "stride"])
        cfg = yaml.safe_load((self.root / "config/run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(sorted(cfg["actions"]), ["sprint", "stride"])
        with self.assertRaises(S.SportError):
            S.add_action("run", "stride", "Trùng", root=self.root)
        with self.assertRaises(S.SportError):
            S.add_action("khong_co", "a1", "A", root=self.root)

    def test_action_of_prefers_longest_key(self):
        sp = S.create_sport("ten", "Tennis", "🎾", [("serve", "Giao"), ("serve_kick", "Giao xoáy")], self.root)
        self.assertEqual(sp.action_of("trainee_serve_kick_01.mp4"), "serve_kick")
        self.assertEqual(sp.action_of("coach_serve_02.mp4"), "serve")
        self.assertIsNone(sp.action_of("video_la.mp4"))

    def test_broken_registry_gives_friendly_error(self):
        (self.root / "config").mkdir()
        (self.root / "config/sports.yaml").write_text("sports: [\n", encoding="utf-8")
        with self.assertRaises(S.SportError):
            S.list_sports(self.root)
        (self.root / "config/sports.yaml").write_text("default: x\n", encoding="utf-8")
        with self.assertRaises(S.SportError):
            S.list_sports(self.root)

    def test_load_engine_errors(self):
        with self.assertRaises(S.SportError):
            S.load_engine("khong.co.module:Lop")
        with self.assertRaises(S.SportError):
            S.load_engine("badminton3d.sports:KhongCoLop")


class MultiSportBackendTests(unittest.TestCase):
    """Giao diện/backend chạy với môn khác cầu lông chỉ nhờ khai báo (engine giả)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        S.create_sport("tennis", "Quần vợt", "🎾", [("forehand", "Cú thuận tay"), ("backhand", "Cú trái tay")], self.root)
        reg = S.load_registry(self.root)
        reg["sports"]["tennis"]["engine"] = "tests.test_sports:FakeEngine"
        S.save_registry(reg, self.root)
        self.sp = S.get_sport("tennis", self.root)
        (self.sp.input / "trainee_forehand_01.mp4").write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_videos_only_from_sport_input(self):
        (self.sp.input / "coach_forehand_01.mp4").write_bytes(b"x")
        (self.sp.input / "video_la.mp4").write_bytes(b"x")
        v = B.list_input_videos(self.sp)
        self.assertEqual({x["name"]: x["action"] for x in v}, {"coach_forehand_01.mp4": "forehand", "trainee_forehand_01.mp4": "forehand", "video_la.mp4": None})
        self.assertEqual([x["name"] for x in B.coach_videos(self.sp, "forehand")], ["coach_forehand_01.mp4"])
        self.assertEqual(B.list_input_videos(S.get_sport("badminton", self.root)), [])           # thư mục môn khác tách biệt

    def test_save_upload_uses_sport_actions(self):
        p = B.save_upload(self.sp, b"abc", "a.MP4", "trainee", "backhand")
        self.assertEqual((p.parent, p.name), (self.sp.input, "trainee_backhand_up01.mp4"))
        with self.assertRaises(B.UIError):
            B.save_upload(self.sp, b"abc", "a.mp4", "trainee", "serve")                          # «serve» không thuộc môn này

    def test_analyze_needs_reference_then_works(self):
        v = self.sp.input / "trainee_forehand_01.mp4"
        with self.assertRaises(B.ReferenceMissing):
            B.analyze_video(self.sp, v)
        code, _ = B.build_reference(self.sp, "forehand", [v.with_name("coach_forehand_01.mp4")]) if False else (0, "")
        self.sp.reference_file("forehand").write_text(json.dumps({"n_videos": 1}), encoding="utf-8")
        rep = B.analyze_video(self.sp, v)
        self.assertEqual(rep["score"]["overall"], 77.0)
        self.assertEqual(B.load_report(self.sp, "trainee_forehand_01")["score"]["grade"], "Khá")
        self.assertTrue(B.list_input_videos(self.sp)[0]["analyzed"])

    def test_unsupported_action_and_missing_engine(self):
        self.assertTrue(self.sp.supports("forehand"))
        self.assertFalse(self.sp.supports("backhand"))
        (self.sp.input / "trainee_backhand_01.mp4").write_bytes(b"x")
        self.sp.reference_file("backhand").write_text("{}", encoding="utf-8")
        with self.assertRaises(B.UIError) as cm:
            B.analyze_video(self.sp, self.sp.input / "trainee_backhand_01.mp4")
        self.assertIn("chưa hỗ trợ", str(cm.exception))
        reg = S.load_registry(self.root)
        reg["sports"]["tennis"]["engine"] = None
        S.save_registry(reg, self.root)
        sp2 = S.get_sport("tennis", self.root)
        with self.assertRaises(B.UIError) as cm:
            B.analyze_video(sp2, self.sp.input / "trainee_forehand_01.mp4")
        self.assertIn("chưa có bộ chấm điểm", str(cm.exception))
        with self.assertRaises(B.UIError):
            B.run_all(sp2)
        with self.assertRaises(B.UIError):
            B.record_live(sp2, "forehand", "0")

    def test_bad_video_name_lists_valid_actions(self):
        p = self.sp.input / "video_la.mp4"
        p.write_bytes(b"x")
        with self.assertRaises(B.UIError) as cm:
            B.analyze_video(self.sp, p)
        self.assertIn("forehand", str(cm.exception))

    def test_build_reference_via_engine_and_rows_fallback(self):
        c = self.sp.input / "coach_forehand_01.mp4"
        c.write_bytes(b"x")
        self.assertEqual(B.build_reference(self.sp, "forehand", [c]), (0, "ok"))
        info, rows = B.reference_rows(self.sp, "forehand")                                      # engine giả không có reference_rows -> dự phòng
        self.assertEqual((info["n_videos"], rows), (1, []))
        with self.assertRaises(B.UIError):
            B.build_reference(self.sp, "forehand", [self.sp.input / "trainee_backhand_01.mp4"])

    def test_config_leaves_and_edit_for_new_sport(self):
        leaves = B.yaml_leaves(self.sp.config)
        keys = {tuple(l["keys"]): l for l in leaves}
        self.assertEqual(keys[("scoring", "pass_score")]["value"], 70)
        self.assertIn("điểm tối thiểu", keys[("scoring", "pass_score")]["comment"])
        B.save_config({("scoring", "pass_score"): 65, ("actions", "forehand", "label"): "Thuận tay (mới)"}, self.sp.config)
        self.assertEqual(B.yaml_get(["scoring", "pass_score"], self.sp.config), 65)
        self.assertEqual(B.yaml_get(["actions", "forehand", "label"], self.sp.config), "Thuận tay (mới)")
        self.assertIn("# điểm tối thiểu", self.sp.config.read_text(encoding="utf-8"))           # chú thích còn nguyên

    def test_summary_images_and_eval_scoped_to_sport(self):
        self.assertEqual(B.summary_images(self.sp), [])
        self.assertEqual(B.evaluation_table(self.sp), [])
        self.sp.tong_hop.mkdir(parents=True)
        (self.sp.tong_hop / "a.png").write_bytes(b"x")
        (self.sp.tong_hop / "evaluation.csv").write_text("video,diem\nv1,80\n", encoding="utf-8")
        self.assertEqual([p.name for p in B.summary_images(self.sp)], ["a.png"])
        self.assertEqual(B.evaluation_table(self.sp), [{"video": "v1", "diem": "80"}])


if __name__ == "__main__":
    unittest.main()
