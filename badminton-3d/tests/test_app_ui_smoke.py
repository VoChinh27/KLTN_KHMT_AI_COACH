"""Kiểm thử khói cho app_ui.py bằng Streamlit GIẢ (môi trường kiểm thử không cài streamlit): chạy hết kịch bản giao diện, bắt lỗi cú pháp/luồng/tên hàm.
Không thay thế việc chạy thật `streamlit run app_ui.py`."""
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from badminton3d import sports as S
from badminton3d import ui_backend as B

ROOT = Path(__file__).resolve().parents[1]


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def make_streamlit(radio_pick=None, button_true=(), upload=None):
    st = mock.MagicMock(name="streamlit")
    pick = radio_pick or {}
    calls = {"buttons": [], "errors": [], "successes": [], "videos": 0, "dataframes": 0, "html": 0}
    st.session_state = SessionState()
    st.tabs.side_effect = lambda labels: [mock.MagicMock() for _ in labels]
    def widget_obj():
        o = mock.MagicMock()
        for fn in ("radio", "selectbox", "text_input", "checkbox", "number_input", "slider", "button", "file_uploader", "multiselect"):
            setattr(o, fn, lambda *a, _f=fn, **k: getattr(st, _f)(*a, **k))        # c.radio(...) hoạt động như st.radio(...)
        return o

    st.columns.side_effect = lambda spec, **k: [widget_obj() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def choose(label, options, index=0, **k):
        opts = list(options)
        return pick.get(label, opts[index] if opts else None)

    st.radio.side_effect = choose
    st.selectbox.side_effect = choose
    st.multiselect.side_effect = lambda label, options, **k: []
    st.checkbox.side_effect = lambda label, value=False, **k: value
    st.number_input.side_effect = lambda label, min_value=0, max_value=1, value=0, *a, **k: value
    st.slider.side_effect = lambda label, min_value=0, max_value=1, value=None, *a, **k: value
    st.text_input.side_effect = lambda label, value="", **k: value
    st.file_uploader.side_effect = lambda *a, **k: upload

    def button(label, **k):
        calls["buttons"].append(label)
        return any(t in label for t in button_true)

    st.button.side_effect = button
    st.form_submit_button.return_value = False
    st.error.side_effect = lambda m, *a, **k: calls["errors"].append(str(m))
    st.success.side_effect = lambda m, *a, **k: calls["successes"].append(str(m))
    st.video.side_effect = lambda *a, **k: calls.__setitem__("videos", calls["videos"] + 1)
    st.dataframe.side_effect = lambda *a, **k: calls.__setitem__("dataframes", calls["dataframes"] + 1)
    comp = types.SimpleNamespace(html=lambda *a, **k: calls.__setitem__("html", calls["html"] + 1))
    return st, comp, calls


def run_app(st, comp):
    st.components = types.SimpleNamespace(v1=comp)
    with mock.patch.dict(sys.modules, {"streamlit": st, "streamlit.components": st.components, "streamlit.components.v1": comp}):
        return runpy.run_path(str(ROOT / "app_ui.py"), run_name="__main__")


class AppSmokeTests(unittest.TestCase):
    def test_default_render_all_tabs(self):
        st, comp, calls = make_streamlit()
        run_app(st, comp)
        self.assertEqual(calls["errors"], [])
        st.set_page_config.assert_called_once()
        labels = [c.args[0] for c in st.tabs.call_args_list if c.args]
        flat = " ".join(" ".join(l) for l in labels if isinstance(l, list))
        for name in ("Chấm điểm Video", "Quay & Chấm trực tiếp", "Chuẩn HLV & Cấu hình", "Thống kê & Báo cáo tổng hợp"):
            self.assertIn(name, flat)
        for btn in ("Bắt đầu Phân tích & Chấm điểm", "Mở Camera & Bắt đầu Quay", "Dựng lại bộ chuẩn", "Chạy đánh giá toàn bộ dữ liệu"):
            self.assertTrue(any(btn in b for b in calls["buttons"]), btn)

    def test_analyze_without_upload_gives_friendly_error(self):
        st, comp, calls = make_streamlit(button_true=("Bắt đầu Phân tích",))
        run_app(st, comp)
        self.assertTrue(any("chưa chọn video" in e for e in calls["errors"]), calls["errors"])

    def test_analyze_existing_video_shows_result(self):
        SP = S.get_sport()
        vids = [v for v in B.list_input_videos(SP) if v["analyzed"]]
        if not vids:
            self.skipTest("chưa có video đã chấm")
        v = vids[0]
        st, comp, calls = make_streamlit(radio_pick={"Nguồn video": "Chọn video có sẵn trong Input/"}, button_true=("Bắt đầu Phân tích",))
        st.selectbox.side_effect = lambda label, options, index=0, **k: list(options)[[x["stem"] for x in B.list_input_videos(SP)].index(v["stem"])] if label == "Chọn video" else list(options)[index]
        with mock.patch.object(B, "analyze_video", return_value=B.load_report(SP, v["stem"])) as an:
            run_app(st, comp)
        an.assert_called_once()
        self.assertEqual(calls["errors"], [])
        self.assertGreaterEqual(calls["videos"] + calls["dataframes"], 1)
        self.assertGreaterEqual(calls["html"], 1)                       # báo cáo HTML nhúng
        self.assertIn(v["stem"], " ".join(str(c) for c in st.markdown.call_args_list))

    def test_reference_missing_shows_hint(self):
        vids = B.list_input_videos(S.get_sport())
        if not vids:
            self.skipTest("Input trống")
        st, comp, calls = make_streamlit(radio_pick={"Nguồn video": "Chọn video có sẵn trong Input/"}, button_true=("Bắt đầu Phân tích",))
        with mock.patch.object(B, "analyze_video", side_effect=B.ReferenceMissing("Chưa có chuẩn HLV cho bài 'serve'")):
            run_app(st, comp)
        self.assertTrue(any("Chưa có chuẩn HLV" in e for e in calls["errors"]))

    def test_unexpected_exception_is_wrapped(self):
        vids = B.list_input_videos(S.get_sport())
        if not vids:
            self.skipTest("Input trống")
        st, comp, calls = make_streamlit(radio_pick={"Nguồn video": "Chọn video có sẵn trong Input/"}, button_true=("Bắt đầu Phân tích",))
        with mock.patch.object(B, "analyze_video", side_effect=RuntimeError("boom")):
            run_app(st, comp)
        self.assertTrue(any("Có lỗi khi phân tích video" in e for e in calls["errors"]))

    def test_camera_button_calls_record(self):
        st, comp, calls = make_streamlit(button_true=("Mở Camera",))
        with mock.patch.object(B, "record_live", return_value=(0, "ok", None)) as rec:
            run_app(st, comp)
        rec.assert_called_once()
        self.assertEqual(rec.call_args.args[1], "ready")                  # bài đầu tiên lấy từ sports.yaml, không hard-code trong giao diện
        self.assertEqual(rec.call_args.args[2], "0")

    def test_build_reference_requires_selection(self):
        st, comp, calls = make_streamlit(button_true=("Dựng lại bộ chuẩn (Build Reference)",))
        run_app(st, comp)
        self.assertTrue(any("ít nhất 1 video HLV" in e for e in calls["errors"]))

    def test_sport_selector_is_first_in_sidebar_and_data_driven(self):
        st, comp, calls = make_streamlit()
        run_app(st, comp)
        labels = [c.args[0] for c in st.selectbox.call_args_list if c.args]
        self.assertEqual(labels[0], "Chọn Môn thể thao")                       # ô chọn môn là widget đầu tiên
        opts = st.selectbox.call_args_list[0].args[1]
        self.assertEqual(opts, list(S.list_sports()))
        self.assertTrue(any("Thêm Môn thể thao / Bài tập mới" in b for b in calls["buttons"]))

    def test_app_has_no_hardcoded_badminton_actions(self):
        src = (ROOT / "app_ui.py").read_text(encoding="utf-8")
        import re
        for bad in (r"[\"']ready[\"']", r"[\"']serve[\"']", r"EXERCISE_VN", r"Reference/ready", r"badminton3d\.yaml"):
            self.assertIsNone(re.search(bad, src), bad)

    def test_switching_to_another_sport_loads_its_actions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sp = S.create_sport("tennis", "Quần vợt", "🎾", S.parse_actions("forehand | Thuận tay\nbackhand | Trái tay"), Path(d))
            sports = {**S.list_sports(), "tennis": sp}
            st, comp, calls = make_streamlit(radio_pick={"Chọn Môn thể thao": "tennis"})
            st.selectbox.side_effect = lambda label, options, index=0, **k: "tennis" if label == "Chọn Môn thể thao" else list(options)[index]
            with mock.patch.object(S, "list_sports", return_value=sports):
                run_app(st, comp)
            radios = [c for c in st.radio.call_args_list if c.args and c.args[0] == "Bài tập"]
            self.assertTrue(radios)
            self.assertEqual(radios[0].args[1], ["forehand", "backhand"])         # danh sách bài tập của môn mới, không phải ready/serve
            self.assertEqual(calls["errors"], [])

    def test_add_sport_form_calls_create_sport(self):
        st, comp, calls = make_streamlit(button_true=("Thêm Môn thể thao",))
        st.form_submit_button.side_effect = lambda label, **k: label == "Tạo"
        st.text_input.side_effect = lambda label, value="", **k: {"Mã (chữ thường, không dấu, ví dụ: tennis, chay_bo)": "tennis", "Tên hiển thị": "Quần vợt"}.get(label, value)
        st.text_area.side_effect = lambda label, value="", **k: "forehand | Thuận tay"
        st.radio.side_effect = lambda label, options, index=0, **k: list(options)[0]
        fake = S.Sport("tennis", "Quần vợt", "🎾", None, Path("c.yaml"), Path("i"), Path("r"), Path("o"), {}, Path("."))
        with mock.patch.object(S, "create_sport", return_value=fake) as cs:
            run_app(st, comp)
        cs.assert_called_once()
        self.assertEqual(cs.call_args.args[0], "tennis")
        self.assertEqual(cs.call_args.args[3], [("forehand", "Thuận tay")])
        self.assertEqual(st.session_state.get("pending_sport"), "tennis")
        self.assertIn("Đã tạo môn", st.session_state.get("flash", ""))

    def test_add_sport_validation_error_is_friendly(self):
        st, comp, calls = make_streamlit(button_true=("Thêm Môn thể thao",))
        st.form_submit_button.side_effect = lambda label, **k: label == "Tạo"
        st.text_input.side_effect = lambda label, value="", **k: "Sai Ma"
        st.text_area.side_effect = lambda label, value="", **k: "a1 | A"
        run_app(st, comp)
        self.assertTrue(any("không hợp lệ" in e or "đã tồn tại" in e for e in calls["errors"]), calls["errors"])

    def test_run_all_button(self):
        st, comp, calls = make_streamlit(button_true=("Chạy đánh giá toàn bộ",))
        with mock.patch.object(B, "run_all", return_value=(0, "ok")) as ra:
            run_app(st, comp)
        ra.assert_called_once()
        self.assertTrue(any("Đã chạy xong" in s for s in calls["successes"]))


if __name__ == "__main__":
    unittest.main()
