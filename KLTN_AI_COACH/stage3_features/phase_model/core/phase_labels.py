"""
stage3_features/phase_labels.py
=================================
Mục đích : Đổi qua lại giữa nhãn THEO BIÊN (.phases.json) và nhãn TỪNG FRAME
           (6 lớp) để huấn luyện / đánh giá / giải mã.

6 lớp (thứ tự CỐ ĐỊNH, đồng thời là thứ tự trái-phải của bộ giải mã):
    0 outside_pre   : trước pha đầu tiên có mặt (VD frame 0..539 trong clip mẫu)
    1 vao_cho
    2 static
    3 sprint_start
    4 drive_phase
    5 outside_post  : sau pha cuối cùng có mặt

Frame KHÔNG thuộc lớp nào (khe hở giữa 2 pha, ví dụ chỗ pha vắng mặt) được
gán IGNORE_LABEL = -1 — loại khỏi huấn luyện/đánh giá theo frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.phase_schema import PHASE_ORDER, PhaseWindow, Rep

CLASSES = ("outside_pre",) + PHASE_ORDER + ("outside_post",)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
N_CLASSES = len(CLASSES)
MIDDLE_IDS = tuple(CLASS_TO_ID[p] for p in PHASE_ORDER)  # (1,2,3,4)
IGNORE_LABEL = -1


def rep_to_frame_labels(rep: Rep, total_frames: int) -> np.ndarray:
    """Nhãn từng frame (int8, độ dài total_frames) từ 1 rep."""
    labels = np.full(total_frames, IGNORE_LABEL, dtype=np.int8)
    present = rep.present()
    if not present:
        return labels
    first = present[0][1].start_frame
    last = present[-1][1].end_frame
    labels[:first] = CLASS_TO_ID["outside_pre"]
    labels[last + 1:] = CLASS_TO_ID["outside_post"]
    for name, w in present:
        labels[w.start_frame:w.end_frame + 1] = CLASS_TO_ID[name]
    return labels


def frame_labels_to_rep(labels: np.ndarray) -> Rep:
    """Ngược lại: mỗi lớp giữa phải liền mạch, nếu không báo lỗi."""
    rep = Rep()
    for name in PHASE_ORDER:
        idx = np.where(labels == CLASS_TO_ID[name])[0]
        if len(idx) == 0:
            rep.phases[name] = PhaseWindow.missing()
            continue
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"nhãn '{name}' không liền mạch (frame {idx[0]}..{idx[-1]}, {len(idx)} frame)")
        rep.phases[name] = PhaseWindow(int(idx[0]), int(idx[-1]))
    return rep


def class_frequencies(labels: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Tần suất từng lớp (bỏ IGNORE), có làm trơn — dùng chia prior cho
    xác suất của bộ phân lớp để ra scaled likelihood."""
    valid = labels[labels != IGNORE_LABEL]
    counts = np.bincount(valid, minlength=N_CLASSES).astype(np.float64) + smoothing
    return counts / counts.sum()
