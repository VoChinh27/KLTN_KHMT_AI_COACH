"""Giao diện chấm điểm tư thế thể thao (Streamlit) — kiến trúc ĐA MÔN THỂ THAO.

Cài đặt:   pip install streamlit            (và pip install -r requirements.txt cho lõi badminton3d)
Chạy:      streamlit run app_ui.py

Giao diện KHÔNG chứa tên bài tập, thư mục hay file cấu hình của môn nào: mọi thứ đọc từ config/sports.yaml (badminton3d/sports.py).
Môn mới (chạy bộ, tennis, gym...) tạo ngay trên giao diện bằng nút «➕ Thêm Môn thể thao / Bài tập mới»; chấm điểm cần gắn engine của môn đó
(docs/MULTISPORT.md). Xử lý nằm ở badminton3d/ui_backend.py (kiểm thử bằng unittest, không cần Streamlit).
"""
from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from badminton3d import __version__
from badminton3d import sports as S
from badminton3d import ui_backend as B

st.set_page_config(page_title="Chấm điểm tư thế thể thao · badminton-3d", page_icon="🏅", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root { --bd-green:#0E9F6E; --bd-dark:#0B3D2E; --bd-blue:#2F80ED; --bd-soft:#E8F5EE; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
.bd-hero { background: linear-gradient(120deg, #0B3D2E 0%, #0E9F6E 60%, #2F80ED 100%); color: #fff; border-radius: 16px; padding: 18px 26px; margin-bottom: 14px; }
.bd-hero h1 { margin: 0; font-size: 1.7rem; letter-spacing: .2px; color:#fff; }
.bd-hero p { margin: 4px 0 0 0; opacity: .92; font-size: .98rem; }
.bd-card { background: var(--bd-soft); border: 1px solid #cfe8da; border-radius: 14px; padding: 14px 18px; }
.bd-badge { display:inline-block; padding: 4px 14px; border-radius: 999px; color:#fff; font-weight:700; font-size: 1.05rem; }
.bd-step { border-left: 4px solid var(--bd-green); padding: 6px 12px; margin: 6px 0; background:#fff; border-radius: 6px; }
div[data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 800; color: var(--bd-dark); }
button[kind="primary"] { background: var(--bd-green); border: none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def rerun() -> None:
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def friendly_error(exc: Exception, where: str) -> None:
    """Hiện lỗi thân thiện; chi tiết kỹ thuật nằm trong khung mở rộng."""
    if isinstance(exc, (B.UIError, S.SportError)):
        st.error(f"⚠️ {exc}")
    else:
        st.error(f"⚠️ Có lỗi khi {where}. Hãy kiểm tra video/cấu hình rồi thử lại.")
        with st.expander("Chi tiết kỹ thuật (gửi cho người phát triển nếu cần)"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def show_log(code: int, log: str, ok_text: str, fail_text: str) -> None:
    (st.success if code == 0 else st.error)(ok_text if code == 0 else fail_text)
    if log:
        with st.expander("Nhật ký chi tiết", expanded=code != 0):
            st.code(log[-6000:])


# ====================================================================== đăng ký môn thể thao
try:
    SPORTS = S.list_sports()
except Exception as exc:  # noqa: BLE001
    st.error(f"⚠️ Không đọc được danh sách môn thể thao (config/sports.yaml): {exc}")
    st.stop()

if st.session_state.get("pending_sport") in SPORTS:                      # vừa tạo môn mới: chọn luôn (phải đặt TRƯỚC khi vẽ ô chọn)
    st.session_state["sport_widget"] = st.session_state.pop("pending_sport")
default_key = S.default_sport_key()
keys = list(SPORTS)
if st.session_state.get("sport_widget") not in keys:
    st.session_state["sport_widget"] = default_key if default_key in keys else keys[0]

with st.sidebar:
    sport_key = st.selectbox("Chọn Môn thể thao", keys, key="sport_widget", format_func=lambda k: f"{SPORTS[k].icon}  {SPORTS[k].name}")
SPORT = SPORTS[sport_key]
SK = SPORT.key                                                           # tiền tố khóa widget: mỗi môn có trạng thái riêng
ACTIONS = list(SPORT.actions)


def action_label(k: str) -> str:
    return SPORT.label(k) + ("" if SPORT.supports(k) else "  (chưa có bộ chấm)")


st.markdown(f"""<div class="bd-hero"><h1>{SPORT.icon} {SPORT.name} · Chấm điểm tư thế</h1>
<p>So từng góc 3D với chuẩn huấn luyện viên · điểm, lỗi và cách sửa · {len(ACTIONS)} bài tập · phiên bản {__version__}</p></div>""", unsafe_allow_html=True)

# ====================================================================== thanh bên
with st.sidebar:
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if st.button("➕ Thêm Môn thể thao / Bài tập mới", key="add_toggle"):
        st.session_state["show_add"] = not st.session_state.get("show_add", False)
    if st.session_state.get("show_add"):
        with st.form("add_sport_form"):
            mode = st.radio("Bạn muốn thêm", ["Môn thể thao mới", f"Bài tập mới cho «{SPORT.name}»"], key="add_mode")
            st.caption("Môn mới: tạo Input/<mã>/, Reference/<mã>/, Output/<mã>/ và config/<mã>.yaml mẫu.")
            a_key = st.text_input("Mã (chữ thường, không dấu, ví dụ: tennis, chay_bo)", value="", key="add_key")
            a_name = st.text_input("Tên hiển thị", value="", key="add_name")
            a_icon = st.text_input("Biểu tượng", value="🏅", key="add_icon")
            a_acts = st.text_area("Bài tập — mỗi dòng: ma_bai | Nhãn hiển thị", value="", height=110, key="add_acts",
                                  help="Ví dụ:\nforehand | Cú đánh thuận tay\nbackhand | Cú đánh trái tay\nVới «Bài tập mới» chỉ cần một dòng.")
            a_hand = st.checkbox("Bài tập cần chọn tay thuận (chỉ áp dụng khi thêm bài tập)", value=False, key="add_hand")
            go = st.form_submit_button("Tạo")
        if go:
            try:
                if mode == "Môn thể thao mới":
                    new = S.create_sport(a_key, a_name, a_icon, S.parse_actions(a_acts))
                    st.session_state["flash"] = f"Đã tạo môn «{new.name}»: Input/{new.key}/, Reference/{new.key}/, Output/{new.key}/ và config/{new.key}.yaml."
                else:
                    pairs = S.parse_actions(a_acts) or S.parse_actions(f"{a_key} | {a_name}")
                    if len(pairs) != 1:
                        raise S.SportError("Mục «Bài tập mới» chỉ thêm 1 bài mỗi lần: nhập một dòng «ma_bai | Nhãn».")
                    new = S.add_action(SPORT.key, pairs[0][0], pairs[0][1], hand=a_hand)
                    st.session_state["flash"] = f"Đã thêm bài tập «{pairs[0][1]}» vào môn «{new.name}»."
                st.session_state["pending_sport"] = new.key
                st.session_state["show_add"] = False
                rerun()
            except Exception as exc:  # noqa: BLE001
                friendly_error(exc, "tạo môn/bài tập")
    st.markdown("---")
    st.markdown("**Trạng thái chuẩn HLV**")
    if not ACTIONS:
        st.caption("Môn này chưa có bài tập nào.")
    for a in ACTIONS:
        st.write(("✅ " if B.reference_exists(SPORT, a) else "❌ ") + SPORT.label(a))
    if SPORT.engine_path is None:
        st.info("Môn này chưa có bộ chấm điểm (engine): bạn quản lý được video, chuẩn và cấu hình. Xem docs/MULTISPORT.md để gắn engine.")
    st.markdown("---")
    st.markdown("**Mẹo quay video tốt**")
    st.markdown("\n".join(f"- {t}" for t in SPORT.tips))
    st.markdown("---")
    st.caption("Lệnh chạy: `streamlit run app_ui.py`")


# ====================================================================== kết quả chấm
def render_result(stem: str, key: str) -> None:
    """Hiển thị kết quả của <Output môn>/<stem>/: điểm, cảnh báo, video overlay, bảng lỗi, biểu đồ, báo cáo HTML."""
    rep = B.load_report(SPORT, stem)
    if rep is None:
        st.info("Chưa có kết quả cho video này. Hãy bấm «Bắt đầu Phân tích & Chấm điểm».")
        return
    s = B.summary(rep)
    files = B.report_files(SPORT, stem)
    st.markdown(f"### 📊 Kết quả: `{stem}`")
    if s["overall"] is None:
        st.warning(s.get("message") or "Video này chưa được chấm điểm (không tìm thấy đoạn/động tác để chấm).")
    else:
        color = B.GRADE_COLOR.get(s["grade"], "#555")
        c1, c2, c3, c4 = st.columns([1.1, 1.2, 1, 1])
        c1.metric("Điểm tổng", f"{s['overall']:.0f}/100")
        c2.markdown(f"<div style='margin-top:1.2rem'><span class='bd-badge' style='background:{color}'>{s['grade']}</span></div>", unsafe_allow_html=True)
        c3.metric("Bài tập", SPORT.short(s["action"]))
        c4.metric("Độ dài video", f"{(s['duration'] or 0):.1f} s")
    if s["warnings"]:
        with st.expander(f"⚠️ Cảnh báo độ tin cậy ({len(s['warnings'])})", expanded=True):
            for w in s["warnings"]:
                st.warning(w)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### 🎬 Video phân tích (khung xương, pha, lỗi từng frame)")
        web = B.web_video(files["overlay"])
        if web is not None:
            st.video(web.read_bytes())
        elif files["overlay"] is not None:
            st.info("Trình duyệt chưa phát được định dạng overlay.mp4 (cần ffmpeg để chuyển sang H.264). Dưới đây là các khung hình chính; bạn vẫn tải được video gốc.")
            frames = B.key_frames(files["overlay"], 6)
            for i in range(0, len(frames), 3):
                cols = st.columns(3)
                for col, fr in zip(cols, frames[i:i + 3]):
                    col.image(fr)
        else:
            st.info("Không có overlay.mp4 (có thể đã tắt xuất video).")
        if files["overlay"] is not None:
            st.download_button("⬇️ Tải overlay.mp4", files["overlay"].read_bytes(), file_name=f"{stem}_overlay.mp4", mime="video/mp4", key=f"dl_ov_{SK}_{key}")
    with right:
        st.markdown("#### 🛠️ Cần sửa gì để tăng điểm")
        rows = B.error_rows(rep)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        elif s["overall"] is not None:
            st.success("🎉 Không có góc nào vượt dung sai. Giữ nguyên kỹ thuật!")
        fm = rep.get("fault_metrics")
        if fm:
            with st.expander("Số đo kiểm tra lỗi riêng"):
                for i, m in enumerate(fm, 1):
                    st.write(f"Lần {i}: " + ", ".join(f"{k} = {'—' if v is None else v}" for k, v in m.items()))

    t1, t2, t3, t4 = st.tabs(["📈 Dashboard", "⏱️ Góc theo thời gian", "🧪 Chất lượng dữ liệu", "📄 Báo cáo HTML đầy đủ"])
    for tab, name, label in ((t1, "dashboard", "Tổng hợp điểm và lỗi"), (t2, "timeline", "Các góc theo thời gian"), (t3, "quality", "Độ phủ và xử lý mất khớp")):
        with tab:
            if files[name] is not None:
                st.image(str(files[name]), caption=label)
            else:
                st.info("Chưa có biểu đồ này.")
    with t4:
        if files["html"] is not None:
            components.html(B.inline_images(files["html"]), height=900, scrolling=True)
            st.download_button("⬇️ Tải report.html", files["html"].read_bytes(), file_name=f"{stem}_report.html", mime="text/html", key=f"dl_html_{SK}_{key}")
        else:
            st.info("Chưa có report.html.")


def hand_picker(action: str, current: str, key: str) -> str:
    """Ô chọn tay thuận chỉ hiện khi bài tập đó cần (khai báo `hand: true` trong sports.yaml)."""
    if not SPORT.actions[action].hand:
        return "auto"
    return st.radio(SPORT.hand_label, list(B.HAND_VN), index=list(B.HAND_VN).index(current) if current in B.HAND_VN else 0, format_func=B.HAND_VN.get, horizontal=True, key=key,
                    help="Khuyến nghị khai báo đúng tay thuận; «Tự động» có thể chọn nhầm khi một tay bị che. Được lưu vào <Input>/<video>.json.")


tab1, tab2, tab3, tab4 = st.tabs(["🎯 Chấm điểm Video", "🎥 Quay & Chấm trực tiếp", "🧩 Chuẩn HLV & Cấu hình", "📚 Thống kê & Báo cáo tổng hợp"])

# ====================================================================== TAB 1
with tab1:
    st.subheader("Chấm điểm video")
    if not ACTIONS:
        st.warning("Môn này chưa có bài tập nào. Dùng «➕ Thêm Môn thể thao / Bài tập mới» ở thanh bên.")
    src = st.radio("Nguồn video", ["Tải video lên", f"Chọn video có sẵn trong {SPORT.input.name}/"], horizontal=True, key=f"{SK}_t1_src")
    existing = B.list_input_videos(SPORT)
    chosen, upload, role, action = None, None, "trainee", (ACTIONS[0] if ACTIONS else None)
    if src == "Tải video lên":
        upload = st.file_uploader("Kéo thả hoặc chọn file video (.mp4, .mov)", type=["mp4", "mov"], key=f"{SK}_t1_upload")
        c1, c2 = st.columns(2)
        if ACTIONS:
            action = c1.radio("Bài tập", ACTIONS, format_func=action_label, key=f"{SK}_t1_ex")
        role = c2.radio("Vai trò trong video", list(B.ROLE_VN), format_func=B.ROLE_VN.get, key=f"{SK}_t1_role")
        if upload is not None:
            st.video(upload.getvalue())
    else:
        if not existing:
            st.info(f"Thư mục {SPORT.input.name}/ chưa có video. Hãy chuyển sang «Tải video lên».")
        else:
            idx = st.selectbox("Chọn video", range(len(existing)), format_func=lambda i: B.video_label(SPORT, existing[i]), key=f"{SK}_t1_pick")
            chosen = existing[idx]
            if chosen["action"] is None:
                st.warning(f"Tên video cần chứa mã bài tập của môn ({', '.join(ACTIONS)}), ví dụ trainee_{ACTIONS[0] if ACTIONS else 'bai'}_01.mp4.")
            else:
                st.caption(f"Bài tập nhận từ tên file: **{SPORT.label(chosen['action'])}** · vai trò: **{B.ROLE_VN.get(chosen['role'], 'không rõ')}**")
                action = chosen["action"]
    cur_hand = B.read_annotation(chosen["stem"], SPORT.input).get("racket_hand", "auto") if chosen else "auto"
    hand = hand_picker(action, cur_hand, f"{SK}_t1_hand_{chosen['stem'] if chosen else 'up'}") if action else "auto"
    force = st.checkbox("Trích lại khung xương từ đầu (bỏ bộ nhớ đệm)", value=False, key=f"{SK}_t1_force")

    if st.button("🚀 Bắt đầu Phân tích & Chấm điểm", type="primary", key=f"{SK}_t1_go"):
        try:
            if src == "Tải video lên":
                if upload is None:
                    raise B.UIError("Bạn chưa chọn video. Hãy tải một file .mp4 hoặc .mov lên.")
                if action is None:
                    raise B.UIError("Môn này chưa có bài tập để chấm.")
                path = B.save_upload(SPORT, upload.getvalue(), upload.name, role, action)
            else:
                if chosen is None:
                    raise B.UIError("Chưa có video nào để chấm.")
                path = chosen["path"]
            with st.spinner("Đang trích khung xương, tách đoạn và chấm điểm… (có thể mất vài chục giây)"):
                B.analyze_video(SPORT, path, hand, force)
            st.session_state[f"{SK}_t1_stem"] = path.stem
            st.success("Đã chấm xong!")
        except Exception as exc:  # noqa: BLE001 - hiển thị thân thiện cho mọi lỗi
            friendly_error(exc, "phân tích video")
    stem = st.session_state.get(f"{SK}_t1_stem") or (chosen["stem"] if chosen and chosen["analyzed"] else None)
    if stem:
        st.markdown("---")
        render_result(stem, "t1")

# ====================================================================== TAB 2
with tab2:
    st.subheader("Quay video trực tiếp rồi chấm ngay")
    st.markdown("<div class='bd-card'>Camera mở trong <b>một cửa sổ riêng trên máy đang chạy giao diện này</b> (cần chạy Streamlit trên chính máy có camera). "
                "Quay xong, kết quả hiện ngay bên dưới.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Nguồn camera", ["Webcam máy tính 0", "Webcam máy tính 1", "Camera điện thoại (nhập URL)"], key=f"{SK}_t2_kind")
    url = ""
    if kind.startswith("Camera điện thoại"):
        url = c1.text_input("Địa chỉ camera", value="http://192.168.1.x:8080/video", key=f"{SK}_t2_url", help="Dùng ứng dụng IP Webcam / DroidCam trên điện thoại, cùng mạng Wi-Fi với máy tính.")
    source = url if url else ("1" if kind.endswith("1") else "0")
    ex2 = c2.radio("Bài tập", ACTIONS, format_func=action_label, key=f"{SK}_t2_ex") if ACTIONS else None
    c3, c4, c5 = st.columns(3)
    role2 = c3.radio("Vai trò", list(B.ROLE_VN), format_func=B.ROLE_VN.get, key=f"{SK}_t2_role")
    hand2 = "auto"
    if ex2 and SPORT.actions[ex2].hand:
        hand2 = c4.radio(SPORT.hand_label, list(B.HAND_VN), format_func=B.HAND_VN.get, key=f"{SK}_t2_hand")
    maxs = c5.slider("Thời gian quay tối đa (giây)", 10, 300, 120, key=f"{SK}_t2_max")
    st.markdown("""<div class='bd-step'><b>Cách thao tác:</b></div>
<div class='bd-step'>1. Bấm nút bên dưới → cửa sổ camera mở ra.</div>
<div class='bd-step'>2. Đứng vào khung hình (thấy toàn thân), bấm nút xanh trên cửa sổ hoặc phím <b>SPACE</b> để <b>bắt đầu quay</b>.</div>
<div class='bd-step'>3. Tập xong bấm <b>SPACE</b> (hoặc nút đỏ) để <b>dừng quay</b>; bấm <b>Q</b> để hủy không lưu.</div>
<div class='bd-step'>4. Hệ thống tự lưu video, chấm điểm và hiện kết quả ở đây.</div>""", unsafe_allow_html=True)
    if st.button("🎥 Mở Camera & Bắt đầu Quay", type="primary", key=f"{SK}_t2_go"):
        try:
            if ex2 is None:
                raise B.UIError("Môn này chưa có bài tập nào để quay.")
            with st.spinner("Camera đang mở… hãy thao tác trong cửa sổ camera (SPACE: quay/dừng). Giao diện sẽ chờ tới khi bạn dừng quay."):
                code, log, stem2 = B.record_live(SPORT, ex2, source, role2, hand2, maxs)
            if stem2 and B.load_report(SPORT, stem2) is not None:
                st.session_state[f"{SK}_t2_stem"] = stem2
                st.success("Đã quay và chấm xong!")
            else:
                show_log(1, log, "", "Chưa quay được đoạn nào hoặc không mở được camera. Kiểm tra camera đã cắm/cấp quyền và thử lại.")
        except Exception as exc:  # noqa: BLE001
            friendly_error(exc, "quay video")
    if st.session_state.get(f"{SK}_t2_stem"):
        st.markdown("---")
        render_result(st.session_state[f"{SK}_t2_stem"], "t2")


# ====================================================================== TAB 3
def quick_settings_form() -> None:
    """Form ngưỡng hay chỉnh: engine mô tả bằng dữ liệu (engine.quick_settings) nên giao diện không hard-code tham số của môn nào."""
    e = None
    try:
        e = SPORT.engine
    except S.SportError:
        pass
    groups = getattr(e, "quick_settings", None) or []
    if not groups:
        return
    changes = {}
    with st.form(f"{SK}_cfg_form"):
        for g in groups:
            st.markdown(f"**{g['title']}**" + (f" — {g['help']}" if g.get("help") else ""))
            cols = st.columns(len(g["items"]))
            for col, it in zip(cols, g["items"]):
                keys_t, cur, kind = tuple(it["keys"]), B.yaml_get(it["keys"], SPORT.config), it["kind"]
                wk = f"{SK}_qs_{'_'.join(it['keys'])}"
                if kind == "bool":
                    new = col.checkbox(it["label"], value=bool(cur), key=wk)
                elif kind == "slider":
                    new = col.slider(it["label"], float(it["min"]), float(it["max"]), float(cur), float(it["step"]), key=wk)
                elif kind == "int_or_null":
                    on = col.checkbox("Bật: " + it["label"], value=cur is not None, key=wk + "_on")
                    val = col.number_input(it["label"], int(it["min"]), int(it["max"]), int(cur if cur is not None else it.get("default", it["min"])), 1, key=wk)
                    new = int(val) if on else None
                else:
                    new = col.number_input(it["label"], float(it["min"]), float(it["max"]), float(cur), float(it["step"]), key=wk)
                if new != cur:
                    changes[keys_t] = new
        saved = st.form_submit_button("💾 Lưu cấu hình nhanh")
    if saved:
        _apply_changes(changes)


def _apply_changes(changes: dict) -> None:
    if not changes:
        st.info("Chưa có thay đổi nào để lưu.")
        return
    B.save_config(changes, SPORT.config)
    st.success(f"Đã lưu {len(changes)} thay đổi vào {SPORT.config.name} (giữ nguyên chú thích; bản sao lưu: {SPORT.config.with_suffix('.yaml.bak').name}). Chấm lại video để áp dụng.")


def all_settings_form() -> None:
    """Trình sửa chung cho MỌI tham số đơn của file cấu hình môn (bool / số / chữ), nhóm theo mục cấp 1."""
    leaves = B.yaml_leaves(SPORT.config)
    if not leaves:
        st.info(f"Chưa có file cấu hình {SPORT.config.name}.")
        return
    changes = {}
    groups = {}
    for lf in leaves:
        groups.setdefault(lf["keys"][0], []).append(lf)
    with st.form(f"{SK}_cfg_all_form"):
        for top, items in groups.items():
            with st.expander(f"{top}  ({len(items)} tham số)"):
                for lf in items:
                    keys_t, cur, label = tuple(lf["keys"]), lf["value"], ".".join(lf["keys"][1:]) or lf["keys"][0]
                    wk = f"{SK}_all_{'_'.join(lf['keys'])}"
                    hint = lf["comment"] or None
                    if isinstance(cur, bool):
                        new = st.checkbox(label, value=cur, key=wk, help=hint)
                    elif isinstance(cur, int):
                        new = int(st.number_input(label, value=cur, step=1, key=wk, help=hint))
                    elif isinstance(cur, float):
                        new = float(st.number_input(label, value=cur, step=0.01 if abs(cur) < 1 else 0.5, format="%.4f", key=wk, help=hint))
                    elif isinstance(cur, str):
                        new = st.text_input(label, value=cur, key=wk, help=hint)
                    else:
                        st.caption(f"{label}: đang tắt (null) — sửa trực tiếp trong file YAML")
                        continue
                    if new != cur:
                        changes[keys_t] = new
        saved = st.form_submit_button("💾 Lưu tất cả tham số")
    if saved:
        _apply_changes(changes)


with tab3:
    if st.session_state.get("t3_msg"):
        st.success(st.session_state.pop("t3_msg"))
    st.subheader("Chuẩn HLV")
    st.caption("Chuẩn = giá trị chuẩn ± dung sai của từng chỉ số, dựng từ 1–3 video HLV tập đúng.")
    if not ACTIONS:
        st.info("Môn này chưa có bài tập nào.")
    else:
        for tab, a in zip(st.tabs([SPORT.label(a) for a in ACTIONS]), ACTIONS):
            with tab:
                try:
                    info, rows = B.reference_rows(SPORT, a)
                except Exception as exc:  # noqa: BLE001
                    friendly_error(exc, "đọc file chuẩn")
                    continue
                if info is None:
                    st.warning(f"Chưa có chuẩn cho bài này ({SPORT.reference_file(a).name}). Hãy dựng chuẩn ở phần bên dưới.")
                else:
                    st.markdown(f"Dựng từ **{info['n_videos']}** video: {', '.join(f'`{n}`' for n in info['built_from']) or '—'} · ngày {info['built_on'] or '—'}")
                    if info["n_videos"] and info["n_videos"] < 3:
                        st.info("Chuẩn mới dựng từ dưới 3 video: dung sai đang dựa vào mức tối thiểu trong cấu hình, điểm mang tính tham khảo.")
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Chưa hiển thị được chi tiết chuẩn (môn này chưa có engine diễn giải file chuẩn).")

        st.markdown("---")
        st.subheader("🛠️ Dựng lại bộ chuẩn (Build Reference)")
        bex = st.radio("Dựng chuẩn cho bài", ACTIONS, format_func=action_label, horizontal=True, key=f"{SK}_t3_ex")
        coaches = B.coach_videos(SPORT, bex)
        picked = st.multiselect(f"Chọn 1–3 video HLV tập đúng trong {SPORT.input.name}/", [v["name"] for v in coaches], key=f"{SK}_t3_pick_{bex}")
        if not coaches:
            st.info(f"{SPORT.input.name}/ chưa có video HLV bài «{SPORT.label(bex)}» (tên dạng coach_{bex}_01.mp4). Bạn có thể tải video HLV mới ở bên dưới.")
        with st.expander("➕ Tải video HLV mới lên"):
            ups = st.file_uploader("Video HLV (.mp4, .mov)", type=["mp4", "mov"], accept_multiple_files=True, key=f"{SK}_t3_up_{bex}")
            uh = hand_picker(bex, "auto", f"{SK}_t3_uh_{bex}")
            if st.button("Lưu video HLV vào thư mục Input", key=f"{SK}_t3_save_{bex}"):
                try:
                    if not ups:
                        raise B.UIError("Bạn chưa chọn video HLV nào.")
                    saved = []
                    for f in ups:
                        p = B.save_upload(SPORT, f.getvalue(), f.name, "coach", bex)
                        if SPORT.actions[bex].hand:
                            B.write_hand(p.stem, uh, SPORT.input)
                        saved.append(p.name)
                    st.success("Đã lưu: " + ", ".join(saved) + ". Hãy chọn chúng ở ô phía trên (làm mới trang nếu chưa thấy).")
                except Exception as exc:  # noqa: BLE001
                    friendly_error(exc, "lưu video HLV")
        if st.button("🛠️ Dựng lại bộ chuẩn (Build Reference)", type="primary", key=f"{SK}_t3_build"):
            try:
                if not picked:
                    raise B.UIError("Hãy chọn ít nhất 1 video HLV (tối đa 3).")
                with st.spinner("Đang trích khung xương các video HLV và dựng chuẩn…"):
                    code, log = B.build_reference(SPORT, bex, [SPORT.input / n for n in picked])
                if code == 0:
                    st.session_state["t3_msg"] = "Đã dựng chuẩn mới! Mọi điểm chấm sau này dùng chuẩn này."
                    rerun()
                show_log(code, log, "", "Dựng chuẩn thất bại (video chưa có đoạn đúng tư thế để lấy chuẩn?). Xem nhật ký.")
            except Exception as exc:  # noqa: BLE001
                friendly_error(exc, "dựng chuẩn")

    st.markdown("---")
    st.subheader(f"⚙️ Cấu hình ngưỡng: {SPORT.config.relative_to(SPORT.root) if SPORT.root in SPORT.config.parents else SPORT.config}")
    try:
        quick_settings_form()
        with st.expander("Tất cả tham số (nâng cao)"):
            all_settings_form()
    except Exception as exc:  # noqa: BLE001
        friendly_error(exc, "đọc/ghi cấu hình")

# ====================================================================== TAB 4
with tab4:
    st.subheader("Thống kê & Báo cáo tổng hợp")
    st.caption("Chạy toàn bộ video của môn (báo cáo từng video + tổng hợp, độ phủ, điểm bỏ-một-video). Có thể mất vài phút.")
    if st.button("🔄 Chạy đánh giá toàn bộ dữ liệu (Run & Evaluate)", type="primary", key=f"{SK}_t4_go"):
        try:
            if not any(B.reference_exists(SPORT, a) for a in ACTIONS):
                raise B.ReferenceMissing("Chưa có chuẩn HLV nào. Hãy dựng chuẩn ở tab «Chuẩn HLV & Cấu hình».")
            if not B.list_input_videos(SPORT):
                raise B.UIError(f"Thư mục {SPORT.input.name}/ chưa có video để chạy.")
            with st.spinner("Đang chạy toàn bộ video và tổng hợp…"):
                code, log = B.run_all(SPORT)
            show_log(code, log, "Đã chạy xong toàn bộ dữ liệu!", "Có lỗi khi chạy. Xem nhật ký.")
        except Exception as exc:  # noqa: BLE001
            friendly_error(exc, "chạy đánh giá")
    idx_html = SPORT.tong_hop / "index.html"
    if idx_html.is_file():
        rows = B.evaluation_table(SPORT)
        if rows:
            st.markdown("#### Điểm các video")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        imgs = B.summary_images(SPORT)
        if imgs:
            st.markdown("#### Biểu đồ tổng hợp")
            g1, g2 = st.columns(2)
            for i, p in enumerate(imgs):
                (g1 if i % 2 == 0 else g2).image(str(p), caption=p.stem.replace("_", " "))
        with st.expander("📄 Báo cáo tổng hợp đầy đủ (index.html)", expanded=True):
            components.html(B.inline_images(idx_html), height=1000, scrolling=True)
    else:
        st.info("Chưa có báo cáo tổng hợp. Bấm «Chạy đánh giá toàn bộ dữ liệu» để tạo.")
