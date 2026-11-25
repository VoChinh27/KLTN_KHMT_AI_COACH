import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from badminton3d import charts
from badminton3d.analyze import analyze_context, build_context, make_warnings
from badminton3d.config import ROOT, load_config
from badminton3d.reference import build_reference
from badminton3d.report import build_html, to_py, write_angles_csv
from tests.helpers import burst, make_df

CFG = load_config()


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "badminton3d", *args], cwd=cwd or str(ROOT), capture_output=True, text=True, encoding="utf-8")


class ReportTests(unittest.TestCase):
    def _result(self, exercise):
        if exercise == "ready":
            ref_ctx = build_context(make_df(150, name="coach_ready_01"), CFG, "ready")
            ref = build_reference("ready", {"coach_ready_01": ref_ctx.units}, CFG)
            ctx = build_context(make_df(150, name="trainee_ready_01", override={"left_knee": (-0.10, 0.45, 0.30), "right_knee": (0.10, 0.45, 0.30)}), CFG, "ready")
        else:
            n = 150
            mk = lambda nm: build_context(make_df(n, name=nm, wrist={"right": burst(n, 70, 10, 0.5)}), CFG, "serve", "right")
            ref_ctx = mk("coach_serve_01")
            ref = build_reference("serve", {"coach_serve_01": ref_ctx.units}, CFG, {"arm_side": {"coach_serve_01": "right"}, "view_yaw_deg": {"coach_serve_01": 0.0}})
            ctx = mk("trainee_serve_01")
        res = analyze_context(ctx, CFG, ref, ctx.pose.video)
        res["warnings"] = make_warnings(res, ctx, ref, CFG)
        return ctx, ref, res

    def test_outputs_are_generated_for_both_exercises(self):
        for ex in ("ready", "serve"):
            ctx, ref, res = self._result(ex)
            self.assertEqual(res["status"], "SCORED", res.get("message"))
            with tempfile.TemporaryDirectory() as d:
                charts.dashboard(ctx, res, CFG, os.path.join(d, "dashboard.png"))
                charts.angles_timeline(ctx, ref, res, CFG, os.path.join(d, "angles_timeline.png"))
                charts.quality_chart(res, CFG, os.path.join(d, "quality.png"))
                write_angles_csv(ctx, os.path.join(d, "angles.csv"))
                html = build_html(res, CFG, d)
                for f in ("dashboard.png", "angles_timeline.png", "quality.png", "angles.csv"):
                    self.assertTrue(os.path.getsize(os.path.join(d, f)) > 500, f)
                self.assertIn("Cần sửa gì để tăng điểm" if res["score"]["errors"] else "Kết quả chấm tư thế", html)
                self.assertIn("data:image/png;base64", html)
                json.dumps(to_py(res), ensure_ascii=False)                     # báo cáo ghi được JSON

    def test_html_lists_advice_and_potential_gain(self):
        _, _, res = self._result("ready")
        html = build_html(res, CFG, self._tmpcharts(res))
        self.assertIn("điểm", html)
        self.assertTrue(res["score"]["errors"])
        self.assertIn(res["score"]["errors"][0]["advice"].split(":")[0][:10], html)

    def _tmpcharts(self, res):
        ctx, ref, res2 = self._result("ready")
        d = tempfile.mkdtemp()
        charts.dashboard(ctx, res2, CFG, os.path.join(d, "dashboard.png"))
        charts.angles_timeline(ctx, ref, res2, CFG, os.path.join(d, "angles_timeline.png"))
        charts.quality_chart(res2, CFG, os.path.join(d, "quality.png"))
        return d

    def test_small_reference_warning_present(self):
        _, _, res = self._result("ready")
        self.assertTrue(any("chỉ dựng từ 1 video" in w for w in res["warnings"]))

    def test_serve_view_mismatch_warning(self):
        ctx, ref, res = self._result("serve")
        ref["meta"]["view_yaw_deg"] = {"coach_serve_01": 80.0}                    # chuẩn quay ngang, video này quay chính diện
        self.assertTrue(any("Góc quay lệch chuẩn" in w for w in make_warnings(res, ctx, ref, CFG)))


class CliTests(unittest.TestCase):
    def test_version_and_help(self):
        self.assertIn("badminton3d", run_cli("--version").stdout)
        h = run_cli("--help").stdout
        for cmd in ("run", "analyze", "build-reference", "coverage", "evaluate", "config"):
            self.assertIn(cmd, h)

    def test_config_dump(self):
        self.assertIn("visibility_min", run_cli("config").stdout)

    def test_missing_input_folder_exit_code_1(self):
        r = run_cli("run", "--input", "/khong/ton/tai")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Lỗi", r.stderr)

    def test_build_reference_rejects_wrong_exercise(self):
        r = run_cli("build-reference", "ready", "Input/coach_serve_02.mp4")
        self.assertEqual(r.returncode, 1)

    def test_missing_reference_message(self):
        with tempfile.TemporaryDirectory() as d:
            from badminton3d.reference import load_reference
            with self.assertRaisesRegex(FileNotFoundError, "build-reference"):
                load_reference(os.path.join(d, "ready.json"))


if __name__ == "__main__":
    unittest.main()
