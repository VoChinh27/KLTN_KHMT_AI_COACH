import unittest

import numpy as np

from badminton3d import render as R


def sc_ready(score, status="HIGH", dev=30.0):
    return {"units": [{"id": "ready_1", "frames": [10, 20], "score": score, "features": [
        {"feature": "knee_mean", "status": status, "dev": dev, "score": 20.0, "level": "PRIMARY", "center": 126.0, "tol": 12.0, "unit": "°"}]}]}


class RenderTests(unittest.TestCase):
    def test_enhance_keeps_shape_and_brightens(self):
        img = np.full((60, 80, 3), 100, np.uint8)
        out = R.enhance_frame(img)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(out.mean(), img.mean())

    def test_vietnamese_text_draws_pixels(self):
        img = np.zeros((60, 300, 3), np.uint8)
        out = R.draw_text_pil(img, "Góc gối: 121° ổ ệ ữ", (4, 4), 20, (255, 255, 255))
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_frame_eval_only_inside_scored_units(self):
        ev = R.build_frame_eval(sc_ready(40.0))
        self.assertIn(15, ev)
        self.assertNotIn(5, ev)
        self.assertNotIn(25, ev)

    def test_pass_threshold_and_colour_sync(self):
        self.assertTrue(R.FrameEval(70.0).passed)
        self.assertFalse(R.FrameEval(69.9).passed)
        _, col_ok = R.verdict_text(R.FrameEval(95.0))
        _, col_bad = R.verdict_text(R.FrameEval(40.0))
        self.assertEqual(col_ok, R.GREEN)
        self.assertEqual(col_bad, R.RED)

    def test_error_message_follows_direction(self):
        fe = R.build_frame_eval(sc_ready(40.0, "HIGH", +30.0))[15]
        self.assertIn("Gối duỗi", R.verdict_text(fe)[0])
        fe = R.build_frame_eval(sc_ready(40.0, "LOW", -30.0))[15]
        self.assertIn("Gối gập", R.verdict_text(fe)[0])

    def test_passing_with_small_error_stays_green_with_tip(self):
        fe = R.build_frame_eval(sc_ready(80.0))[15]
        text, col = R.verdict_text(fe)
        self.assertTrue(text.startswith("ĐẠT"))
        self.assertIn(col, (R.GREEN, R.LIGHT_GREEN))

    def test_status_icon_draws(self):
        for passed in (True, False):
            img = np.zeros((300, 300, 3), np.uint8)
            R.draw_status_icon(img, passed)
            self.assertGreater(int(img[150:, 150:].sum()), 0)


if __name__ == "__main__":
    unittest.main()
