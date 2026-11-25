import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from badminton3d import sports as S
from badminton3d import ui_backend as B

ROOT = Path(__file__).resolve().parents[1]
REAL_CFG = ROOT / "config" / "badminton3d.yaml"


def tmp_sport(d):
    """Môn cầu lông trên thư mục tạm (bố cục cũ: Input/, Reference/, Output/)."""
    return S.get_sport("badminton", Path(d))


class FileNamingTests(unittest.TestCase):
    def test_list_and_unique_names(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            sp.input.mkdir()
            for n in ("coach_serve_01.mp4", "trainee_ready_03.mov", "ghi_chu.txt", ".an.mp4", "video_la.mp4"):
                (sp.input / n).write_bytes(b"x")
            vids = B.list_input_videos(sp)
            self.assertEqual([v["name"] for v in vids], ["coach_serve_01.mp4", "trainee_ready_03.mov", "video_la.mp4"])
            self.assertEqual([v["action"] for v in vids], ["serve", "ready", None])
            self.assertEqual(B.unique_stem("trainee", "serve", "up", sp.input), "trainee_serve_up01")
            (sp.input / "trainee_serve_up01.mp4").write_bytes(b"x")
            self.assertEqual(B.unique_stem("trainee", "serve", "up", sp.input), "trainee_serve_up02")

    def test_save_upload_validation(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            p = B.save_upload(sp, b"abc", "Video Cua Toi.MOV", "trainee", "ready")
            self.assertEqual(p.name, "trainee_ready_up01.mov")
            self.assertEqual(p.read_bytes(), b"abc")
            for args in ((b"abc", "a.txt", "trainee", "ready"), (b"", "a.mp4", "trainee", "ready"), (b"x", "a.mp4", "boss", "ready"), (b"x", "a.mp4", "coach", "dance")):
                with self.assertRaises(B.UIError):
                    B.save_upload(sp, *args)

    def test_write_hand_keeps_other_keys(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "v.json").write_text(json.dumps({"person_roi": [0, 0, 0.5, 1]}), encoding="utf-8")
            B.write_hand("v", "left", Path(d))
            self.assertEqual(B.read_annotation("v", Path(d)), {"person_roi": [0, 0, 0.5, 1], "racket_hand": "left"})
            B.write_hand("v", "auto", Path(d))
            self.assertEqual(B.read_annotation("v", Path(d)), {"person_roi": [0, 0, 0.5, 1]})
            B.write_hand("w", "auto", Path(d))
            self.assertFalse((Path(d) / "w.json").exists())
            with self.assertRaises(B.UIError):
                B.write_hand("v", "both", Path(d))


class YamlEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = Path(self.tmp) / "c.yaml"
        shutil.copy2(REAL_CFG, self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_set_keeps_comments_and_other_lines(self):
        before = self.cfg.read_text(encoding="utf-8")
        B.yaml_set(["faults", "heel_lift", "threshold"], 0.15, self.cfg)
        after = self.cfg.read_text(encoding="utf-8")
        self.assertEqual(B.yaml_get(["faults", "heel_lift", "threshold"], self.cfg), 0.15)
        self.assertEqual(B.yaml_get(["faults", "follow_through", "warn_m"], self.cfg), 0.15)      # khóa khác giống tên không bị đụng
        changed = [(a, b) for a, b in zip(before.split("\n"), after.split("\n")) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn("# tỉ lệ nhấc gót", changed[0][1])                                         # chú thích còn nguyên

    def test_bool_null_and_int(self):
        B.yaml_set(["faults", "follow_through", "enabled"], False, self.cfg)
        B.yaml_set(["scoring", "critical_cap"], None, self.cfg)
        B.yaml_set(["faults", "follow_through", "penalty_high"], 12, self.cfg)
        self.assertIs(B.yaml_get(["faults", "follow_through", "enabled"], self.cfg), False)
        self.assertIsNone(B.yaml_get(["scoring", "critical_cap"], self.cfg))
        self.assertEqual(B.yaml_get(["faults", "follow_through", "penalty_high"], self.cfg), 12)
        B.yaml_set(["scoring", "critical_cap"], 69, self.cfg)
        self.assertEqual(B.yaml_get(["scoring", "critical_cap"], self.cfg), 69)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            B.yaml_set(["faults", "khong_co", "x"], 1, self.cfg)

    def test_save_config_restores_on_error(self):
        before = self.cfg.read_text(encoding="utf-8")
        with self.assertRaises(B.UIError):
            B.save_config({("faults", "heel_lift", "threshold"): 0.2, ("faults", "khong_co", "x"): 1}, self.cfg)
        self.assertEqual(self.cfg.read_text(encoding="utf-8"), before)
        B.save_config({("faults", "heel_lift", "threshold"): 0.22, ("faults", "heel_lift", "enabled"): False}, self.cfg)
        self.assertEqual(B.yaml_get(["faults", "heel_lift", "threshold"], self.cfg), 0.22)


class ReportHelpersTests(unittest.TestCase):
    REP = {"status": "SCORED", "exercise": "serve", "duration_s": 4.0, "warnings": ["w1"],
           "score": {"overall": 59.0, "grade": "Cần cải thiện", "errors": [
               {"where": "Sau cú chạm", "feature": "follow_through", "label": "Vung theo", "unit": "m", "value": 0.44, "center": 0.0, "tol": 0.15, "severity": "nặng", "advice": "Dừng vợt sớm", "potential_gain": 10.0},
               {"where": "Chạm cầu", "feature": "knee_mean", "label": "Góc gối", "unit": "°", "value": 164.0, "center": 160.0, "tol": 10.0, "severity": "nhẹ", "advice": "Gập gối", "potential_gain": 2.34}]}}

    def test_error_rows_format(self):
        rows = B.error_rows(self.REP)
        self.assertEqual(rows[0]["Chuẩn HLV"], "≤ 15 cm")
        self.assertEqual(rows[0]["Hiện tại"], "44 cm")
        self.assertEqual(rows[1]["Chuẩn HLV"], "160° ± 10°")
        self.assertEqual(rows[1]["Điểm lấy lại được"], 2.3)
        self.assertEqual(B.summary(self.REP)["grade"], "Cần cải thiện")

    def test_no_score(self):
        self.assertEqual(B.error_rows({"status": "NO_UNITS"}), [])
        self.assertIsNone(B.summary({"status": "NO_UNITS"})["overall"])

    def test_fmt_val(self):
        self.assertEqual(B.fmt_val(None, "°"), "—")
        self.assertEqual(B.fmt_val(float("nan"), "m"), "—")
        self.assertEqual(B.fmt_val(1.5, "x"), "150% vai")

    def test_real_reports_load(self):
        sp = S.get_sport("badminton", ROOT)
        for stem in ("trainee_serve_09", "coach_ready_01"):
            if (ROOT / "Output" / stem / "report.json").is_file():
                rep = B.load_report(sp, stem)
                self.assertIn("status", rep)
                B.error_rows(rep)

    def test_parse_record_stem(self):
        log = "Đã lưu x.mp4 ...\n[SCORED] trainee_serve_live03 -> /tmp/Output/trainee_serve_live03/report.html\n  Điểm: 59.0/100"
        self.assertEqual(B.parse_record_stem(log), "trainee_serve_live03")
        self.assertIsNone(B.parse_record_stem("Đã thoát, không lưu."))


class HtmlAndReferenceTests(unittest.TestCase):
    def test_inline_images(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.png").write_bytes(b"\x89PNG\r\n")
            (Path(d) / "r.html").write_text("<img src='a.png'><img src=\"http://x/y.png\"><img alt='k' src='thieu.png'>", encoding="utf-8")
            out = B.inline_images(Path(d) / "r.html")
            self.assertIn("data:image/png;base64,", out)
            self.assertIn("http://x/y.png", out)
            self.assertIn("thieu.png", out)
            self.assertEqual(B.inline_images(Path(d) / "khong_co.html"), "")

    def test_reference_rows_real_files(self):
        sp = S.get_sport("badminton", ROOT)
        for ex in ("ready", "serve"):
            if sp.reference_file(ex).is_file():
                info, rows = B.reference_rows(sp, ex)
                self.assertGreaterEqual(info["n_videos"], 1)
                self.assertTrue(rows and "Khoảng chấp nhận" in rows[0])
                self.assertEqual("Pha" in rows[0], ex == "serve")

    def test_reference_missing(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            sp.input.mkdir()
            info, rows = B.reference_rows(sp, "ready")
            self.assertIsNone(info)
            self.assertEqual(rows, [])
            v = sp.input / "trainee_ready_99.mp4"
            v.write_bytes(b"x")
            with self.assertRaises(B.ReferenceMissing):
                B.analyze_video(sp, v)

    def test_build_reference_validation(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            ok = Path(d) / "coach_ready_01.mp4"
            ok.write_bytes(b"x")
            wrong = Path(d) / "coach_serve_01.mp4"
            wrong.write_bytes(b"x")
            for ex, vids in (("ready", []), ("ready", [ok] * 4), ("ready", [wrong]), ("ready", [Path(d) / "khong_co.mp4"]), ("dance", [ok])):
                with self.assertRaises(B.UIError):
                    B.build_reference(sp, ex, vids)

    def test_record_live_validation(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            with self.assertRaises(B.UIError):
                B.record_live(sp, "dance", "0")
            with self.assertRaises(B.UIError):
                B.record_live(sp, "serve", "  ")
            with self.assertRaises(B.ReferenceMissing):
                B.record_live(sp, "serve", "0")

    def test_analyze_rejects_bad_names(self):
        with tempfile.TemporaryDirectory() as d:
            sp = tmp_sport(d)
            sp.input.mkdir()
            p = sp.input / "video_la.mp4"
            p.write_bytes(b"x")
            with self.assertRaises(B.UIError):
                B.analyze_video(sp, p)
            with self.assertRaises(B.UIError):
                B.analyze_video(sp, sp.input / "khong_co_serve.mp4")


class WebVideoTests(unittest.TestCase):
    def test_none_inputs(self):
        self.assertIsNone(B.web_video(None))
        self.assertIsNone(B.web_video(Path("/khong/co.mp4")))
        self.assertEqual(B.key_frames(None), [])

    @unittest.skipUnless(B._ffmpeg(), "không có ffmpeg")
    def test_transcode_and_cache(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "overlay.mp4"
            w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 96))
            for i in range(20):
                w.write(np.full((96, 160, 3), i * 10, np.uint8))
            w.release()
            out = B.web_video(src)
            self.assertIsNotNone(out)
            self.assertTrue(out.exists() and out.stat().st_size > 0)
            m1 = out.stat().st_mtime
            self.assertEqual(B.web_video(src), out)                    # lần 2 dùng lại bản đã chuyển
            self.assertEqual(out.stat().st_mtime, m1)
            self.assertEqual(len(B.key_frames(src, 4)), 4)


class RunCliTests(unittest.TestCase):
    def test_run_cli_help(self):
        code, out = B.run_cli(["--version"], timeout=120)
        self.assertEqual(code, 0)
        self.assertIn("badminton3d", out)

    def test_clean_log_filters_noise(self):
        raw = "W0000 00:00:1 inference_feedback_manager.cc\nINFO: Created TensorFlow Lite XNNPACK delegate\nDòng thật\n\n"
        self.assertEqual(B.clean_log(raw), "Dòng thật")


if __name__ == "__main__":
    unittest.main()
