"""Quay video trực tiếp từ camera (webcam / điện thoại làm webcam / video giả lập) rồi chấm điểm ngay khi dừng.

    python -m badminton3d record serve                  # webcam 0; bấm nút trên cửa sổ hoặc phím SPACE để quay / dừng
    python -m badminton3d record ready --source 1       # webcam thứ hai
    python -m badminton3d record serve --source http://192.168.1.5:8080/video      # điện thoại (IP Webcam / DroidCam...)
    python -m badminton3d record serve --source Input/trainee_serve_09.mp4 --no-preview   # giả lập camera bằng file (kiểm thử)

Video lưu vào Input/<vai trò>_<bài>_live<NN>.mp4 (cùng quy ước đặt tên), kèm <video>.json khai báo tay vợt; kết quả y như `analyze`:
Output/<video>/report.html, overlay.mp4, ...
"""
from __future__ import annotations

import json
import os
import re
import time
import webbrowser
from typing import Optional

import cv2
import numpy as np

WINDOW = "badminton3d - quay video"


def next_name(input_dir: str, role: str, exercise: str) -> str:
    """Tên mới không trùng: trainee_serve_live01, live02, ..."""
    os.makedirs(input_dir, exist_ok=True)
    used = [int(m.group(1)) for f in os.listdir(input_dir) if (m := re.match(rf"{role}_{exercise}_live(\d+)\.", f))]
    return f"{role}_{exercise}_live{(max(used) + 1 if used else 1):02d}"


def _open(source):
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(src)
    return cap, isinstance(src, int) or (isinstance(src, str) and src.startswith(("http", "rtsp")))


def _button(frame, recording: bool, t: float):
    h, w = frame.shape[:2]
    bw, bh = int(w * 0.22), int(h * 0.09)
    x0, y0 = w // 2 - bw // 2, h - bh - int(h * 0.03)
    col = (40, 40, 220) if recording else (60, 170, 60)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), col, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 2)
    label = f"DUNG ({t:4.1f}s)" if recording else "BAT DAU QUAY"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * w / 1280, 2)
    cv2.putText(frame, label, (x0 + (bw - tw) // 2, y0 + (bh + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * w / 1280, (255, 255, 255), 2, cv2.LINE_AA)
    if recording and int(t * 2) % 2 == 0:
        cv2.circle(frame, (int(w * 0.04), int(h * 0.06)), int(h * 0.02), (0, 0, 255), -1)
    return (x0, y0, x0 + bw, y0 + bh)


def record(exercise: str, cfg: dict, *, source="0", role: str = "trainee", input_dir: str = "Input", output_dir: str = "Output", reference_dir: str = "Reference",
           racket_hand: Optional[str] = None, width: int = 1280, height: int = 720, fps: float = 30.0, preview: bool = True, max_seconds: float = 180.0,
           open_report: bool = True, analyze: bool = True, force_name: Optional[str] = None) -> Optional[dict]:
    """Quay -> lưu -> chấm. Trả về kết quả (dict như `analyze`) hoặc None nếu không quay gì."""
    from badminton3d.runner import process_video

    cap, is_live = _open(source)
    if not cap.isOpened():
        raise IOError(f"Không mở được nguồn video/camera: {source}")
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    stem = force_name or next_name(input_dir, role, exercise)
    path = os.path.join(input_dir, stem + ".mp4")
    tmp = os.path.join(input_dir, f".{stem}.tmp.mp4")
    state = {"rec": not preview, "stop": False, "quit": False, "btn": (0, 0, 0, 0)}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            b = state["btn"]
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                state["rec"], state["stop"] = (True, False) if not state["rec"] else (False, True)

    if preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, on_mouse)
    writer, n, t0 = None, 0, None
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if state["rec"]:
                t0 = t0 or time.time()
                if writer is None:
                    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (fr.shape[1], fr.shape[0]))
                writer.write(fr)
                n += 1
                if (time.time() - t0 if is_live else n / src_fps) >= max_seconds:
                    break
            elif state["stop"] and writer is not None:
                break
            if preview:
                shown = fr.copy()
                state["btn"] = _button(shown, state["rec"], (time.time() - t0) if (t0 and state["rec"]) else 0.0)
                cv2.putText(shown, f"{exercise.upper()}  |  SPACE: quay/dung   Q: thoat", (10, shown.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * shown.shape[1] / 1280, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, shown)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    state["rec"], state["stop"] = (True, False) if not state["rec"] else (False, True)
                elif k in (ord("q"), 27):
                    state["quit"] = True
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
    elapsed = (time.time() - t0) if (t0 and is_live) else (n / src_fps if n else 0.0)
    if state["quit"] or n < 10:
        if os.path.exists(tmp):
            os.remove(tmp)
        print("Chưa quay được đoạn nào (cần ít nhất vài frame); không lưu." if not state["quit"] else "Đã thoát, không lưu.")
        return None
    real_fps = n / elapsed if (is_live and elapsed > 0.5) else src_fps       # camera thường báo sai FPS: dùng FPS đo thực tế
    if abs(real_fps - src_fps) / src_fps > 0.08:
        _rewrite_with_fps(tmp, path, real_fps)
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    with open(os.path.join(input_dir, stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"racket_hand": racket_hand} if racket_hand in ("left", "right") else {}, fh)
    print(f"Đã lưu {path}  ({n} frame, {n / real_fps:.1f} giây, {real_fps:.1f} FPS). Đang chấm điểm...")
    if not analyze:
        return {"video": stem, "saved": path}
    res, _ = process_video(path, cfg, output_dir, reference_dir, racket_hand)
    rep = os.path.join(output_dir, stem, "report.html")
    sc = res.get("score")
    print(f"[{res['status']}] {stem} -> {rep}")
    if sc and sc.get("overall") is not None:
        print(f"  Điểm: {sc['overall']:.1f}/100 ({sc['grade']})")
        for e in sc["errors"][:3]:
            print(f"  - {e['advice']}")
    elif res.get("message"):
        print("  " + res["message"])
    if open_report and preview:
        webbrowser.open("file://" + os.path.abspath(rep))
    return res


def _rewrite_with_fps(src: str, dst: str, fps: float) -> None:
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(fr)
    cap.release()
    wr.release()
