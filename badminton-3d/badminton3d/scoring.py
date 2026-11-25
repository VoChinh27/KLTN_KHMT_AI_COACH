"""CHẤM ĐIỂM angle-by-angle: mỗi góc so với chuẩn HLV; trong dung sai = 100, lệch ra ngoài thì giảm dần.
   Có gom lỗi, xếp theo số điểm bị mất, và tính điểm có thể tăng thêm nếu sửa từng lỗi."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

ADVICE = {  # (đặc trưng, 'high'|'low') -> câu góp ý; {d} = độ lệch so với chuẩn
    ("knee_mean", "high"): "Gối đang duỗi hơn chuẩn khoảng {d}: hạ trọng tâm, gập gối thêm.",
    ("knee_mean", "low"): "Gối gập sâu hơn chuẩn khoảng {d}: nâng người lên một chút để di chuyển nhanh hơn.",
    ("hip_mean", "high"): "Thân đang quá thẳng so với đùi (lệch {d}): gập hông, đổ người ra trước thêm.",
    ("hip_mean", "low"): "Gập hông nhiều hơn chuẩn {d}: đứng cao hơn một chút.",
    ("elbow_mean", "high"): "Khuỷu tay duỗi hơn chuẩn {d}: gập khuỷu, giữ vợt gần thân hơn.",
    ("elbow_mean", "low"): "Khuỷu tay gập hơn chuẩn {d}: duỗi bớt để vợt thoải mái.",
    ("shoulder_mean", "high"): "Tay giơ/dang xa thân hơn chuẩn {d}: thu tay lại gần thân hơn.",
    ("shoulder_mean", "low"): "Tay đang áp sát thân (lệch {d}): đưa vợt ra trước, nâng tay lên.",
    ("elbow_arm", "high"): "Khuỷu tay duỗi hơn chuẩn {d}: giữ khuỷu gập hơn ở bước này.",
    ("elbow_arm", "low"): "Khuỷu tay gập hơn chuẩn {d}: duỗi khuỷu để đẩy vợt êm hơn.",
    ("shoulder_arm", "high"): "Tay nâng/dang cao hơn chuẩn {d}: hạ tay xuống.",
    ("shoulder_arm", "low"): "Tay đang áp sát thân (lệch {d}): đưa tay ra trước hơn.",
    ("trunk_lean", "high"): "Thân đổ ra trước hơn chuẩn {d}: dựng thân lên, giữ thăng bằng.",
    ("trunk_lean", "low"): "Thân còn thẳng hơn chuẩn {d}: đổ nhẹ ra trước.",
    ("stance_ratio", "high"): "Hai chân mở rộng hơn chuẩn: thu hẹp bước chân để bật nhanh hơn.",
    ("stance_ratio", "low"): "Hai chân hẹp hơn chuẩn: mở rộng chân tới bằng hoặc hơn bề rộng vai.",
    ("wrist_height", "high"): "Tay cao hơn chuẩn {d}: hạ tay xuống (điểm chạm cầu phải thấp).",
    ("wrist_height", "low"): "Tay thấp hơn chuẩn {d}: nâng tay lên một chút.",
    ("wrist_reach", "high"): "Tay đưa xa thân hơn chuẩn {d}: thu tay lại gần thân.",
    ("wrist_reach", "low"): "Tay quá sát thân (lệch {d}): đưa vợt ra trước hơn.",
}


def fmt_dev(dev: float, unit: str) -> str:
    a = abs(dev)
    if unit == "°":
        return f"{a:.0f}°"
    if unit == "m":
        return f"{a * 100:.0f} cm"
    return f"{a:.2f}"


def feature_score(x: float, center: float, tol: float, decay: float) -> float:
    dev = abs(x - center)
    return 100.0 if dev <= tol else max(0.0, 100.0 * (1.0 - (dev - tol) / (decay * tol)))


def score_features(stats: Dict[str, dict], ref_feats: Dict[str, dict], cfg: dict) -> dict:
    """Chấm một tập đặc trưng của 1 đơn vị (hoặc 1 pha). Trả về từng góc + điểm tổng của tập."""
    sc, lw, minrel = cfg["scoring"], cfg["scoring"]["level_weights"], cfg["quality"]["min_reliability_to_score"]
    rows, num, den = [], 0.0, 0.0
    for name, r in ref_feats.items():
        st = stats.get(name)
        row = {"feature": name, "label": r["label"], "unit": r["unit"], "level": r["level"], "center": r["center"], "tol": r["tol"]}
        if st is None or st["value"] != st["value"] or st["rel"] < minrel:
            row.update(status="NOT_ASSESSED", score=None, weight=0.0, value=None if st is None or st["value"] != st["value"] else st["value"],
                       rel=0.0 if st is None else st["rel"], coverage=0.0 if st is None else st["coverage"], substituted=0.0 if st is None else st["substituted"])
            rows.append(row)
            continue
        s = feature_score(st["value"], r["center"], r["tol"], sc["decay_factor"])
        w = lw[r["level"]] * st["rel"]
        dev = st["value"] - r["center"]
        row.update(value=st["value"], dev=dev, score=s, weight=w, rel=st["rel"], coverage=st["coverage"], substituted=st["substituted"],
                   status="OK" if abs(dev) <= r["tol"] else ("HIGH" if dev > 0 else "LOW"))
        rows.append(row)
        num, den = num + w * s, den + w
    raw = (num / den) if den > 0 else None
    # Lỗi nghiêm trọng: một góc PRIMARY lệch quá xa chuẩn (điểm góc = 0) thì cả đơn vị không thể được tính là "đạt"
    critical = [r["feature"] for r in rows if r.get("level") == "PRIMARY" and r.get("score") == 0.0]
    cap = sc.get("critical_cap")
    score = min(raw, float(cap)) if (raw is not None and cap is not None and critical) else raw
    return {"score": score, "raw_score": raw, "critical": critical, "weight_sum": den, "features": rows}


def grade(score: Optional[float], cfg: dict) -> str:
    if score is None:
        return "Không đủ dữ liệu"
    for th, name in cfg["scoring"]["grades"]:
        if score >= th:
            return name
    return cfg["scoring"]["grades"][-1][1]


def score_ready(units: List[dict], ref: dict, cfg: dict) -> dict:
    per = []
    for u in units:
        r = score_features(u["features"], ref["features"], cfg)
        per.append({"id": u["id"], "frames": list(u["frames"]), "duration_s": u["duration_s"], **r})
    scored = [p for p in per if p["score"] is not None]
    tot = sum(p["duration_s"] for p in scored)
    overall = sum(p["score"] * p["duration_s"] for p in scored) / tot if tot > 0 else None
    conf = (sum(p["weight_sum"] * p["duration_s"] for p in scored) / tot) if tot > 0 else 0.0
    return {"overall": overall, "grade": grade(overall, cfg), "units": per, "scored_seconds": tot,
            "errors": rank_errors_ready(per, cfg)}


def score_serve(units: List[dict], ref: dict, cfg: dict) -> dict:
    pw = cfg["features"]["serve"]["phase_weights"]
    per = []
    for u in units:
        phs, num, wsum = {}, 0.0, 0.0
        for ph, w in pw.items():
            if ph in u["phases"] and ph in ref["phases"]:
                r = score_features(u["phases"][ph], ref["phases"][ph], cfg)
                r["frames"], r["weight"] = list(u["phase_frames"][ph]), w
                phs[ph] = r
                if r["score"] is not None:
                    num, wsum = num + w * r["score"], wsum + w
        ok = wsum >= cfg["scoring"]["min_phase_weight_covered"]
        rep = (num / wsum) if ok else None
        crit = sorted({f for r in phs.values() for f in r.get("critical", [])})
        cap = cfg["scoring"].get("critical_cap")
        if rep is not None and cap is not None and crit:
            rep = min(rep, float(cap))
        per.append({"id": u["id"], "frames": list(u["frames"]), "duration_s": u["duration_s"], "truncated": u["truncated"], "no_prep": u.get("no_prep", False),
                    "score": rep, "raw_score": (num / wsum) if ok else None, "critical": crit, "phase_weight_covered": wsum, "phases": phs})
    scored = [p for p in per if p["score"] is not None]
    overall = float(np.mean([p["score"] for p in scored])) if scored else None
    return {"overall": overall, "grade": grade(overall, cfg), "units": per, "errors": rank_errors_serve(per, cfg, pw)}


# ------------------------------------------------------------------ gom lỗi, xếp hạng, điểm có thể tăng
def _advice(f: dict) -> str:
    d = "high" if f["dev"] > 0 else "low"
    t = ADVICE.get((f["feature"], d), "Điều chỉnh về gần chuẩn.")
    return t.format(d=fmt_dev(f["dev"], f["unit"]))


def severity(feature_score: Optional[float]) -> str:
    """Mức độ lỗi theo điểm của góc: >= 60 nhẹ · 20–60 vừa · < 20 nặng (0 = lệch quá xa chuẩn)."""
    if feature_score is None:
        return "nhẹ"
    return "nhẹ" if feature_score >= 60 else ("vừa" if feature_score >= 20 else "nặng")


def _error_row(f: dict, lost: float, gain: float, where: str, n: int) -> dict:
    return {"where": where, "feature": f["feature"], "label": f["label"], "unit": f["unit"], "value": f["value"], "center": f["center"], "tol": f["tol"],
            "dev": f["dev"], "direction": "cao hơn chuẩn" if f["dev"] > 0 else "thấp hơn chuẩn", "score": f["score"], "lost_points": lost, "potential_gain": gain,
            "advice": _advice(f), "n_units": n, "severity": severity(f["score"]), "kind": "LỖI CƠ BẢN"}


def rank_errors_ready(per: List[dict], cfg: dict) -> List[dict]:
    scored = [p for p in per if p["score"] is not None]
    tot = sum(p["duration_s"] for p in scored)
    acc: Dict[str, dict] = {}
    for p in scored:
        wsum = p["weight_sum"]
        for f in p["features"]:
            if f["status"] in ("HIGH", "LOW"):
                lost = f["weight"] * (100 - f["score"]) / wsum * p["duration_s"] / tot
                a = acc.setdefault(f["feature"], {"f": f, "lost": 0.0, "n": 0, "dev": 0.0, "val": 0.0, "dur": 0.0})
                a["lost"] += lost
                a["n"] += 1
                a["dev"] += f["dev"] * p["duration_s"]
                a["val"] += f["value"] * p["duration_s"]
                a["dur"] += p["duration_s"]
    out = []
    for name, a in acc.items():
        f = dict(a["f"])
        f["dev"], f["value"] = a["dev"] / a["dur"], a["val"] / a["dur"]
        f["score"] = f["score"]
        out.append(_error_row(f, a["lost"], a["lost"], "Tư thế sẵn sàng", a["n"]))
    return sorted(out, key=lambda e: -e["lost_points"])


def rank_errors_serve(per: List[dict], cfg: dict, pw: dict) -> List[dict]:
    scored = [p for p in per if p["score"] is not None]
    n_units = max(1, len(scored))
    acc: Dict[tuple, dict] = {}
    labels = cfg["features"]["serve"]["phase_labels"]
    for p in scored:
        wcov = p["phase_weight_covered"]
        for ph, r in p["phases"].items():
            if r["score"] is None:
                continue
            for f in r["features"]:
                if f["status"] in ("HIGH", "LOW"):
                    lost = (pw[ph] / wcov) * f["weight"] * (100 - f["score"]) / r["weight_sum"] / n_units
                    a = acc.setdefault((ph, f["feature"]), {"f": f, "lost": 0.0, "n": 0, "dev": 0.0, "val": 0.0})
                    a["lost"] += lost
                    a["n"] += 1
                    a["dev"] += f["dev"]
                    a["val"] += f["value"]
    out = []
    for (ph, name), a in acc.items():
        f = dict(a["f"])
        f["dev"], f["value"] = a["dev"] / a["n"], a["val"] / a["n"]
        out.append(_error_row(f, a["lost"], a["lost"], f"Pha {labels[ph]}", a["n"]))
    return sorted(out, key=lambda e: -e["lost_points"])
