"""Đánh giá OOF (leave-one-person-out) theo GIÂY, kèm chẩn đoán và so sánh cấu hình.

Chế độ chạy
-----------
  (mặc định)      Chạy LOPO cho 1 cấu hình, in bảng theo giây + ma trận nhầm lẫn theo frame,
                  ghi reports/oof_<tag>.json.
  --sweep         Thêm: quét skip_penalty x emission_scale trên xác suất OOF đã lưu (không train lại).
                  Lưu ý: chọn tham số trên chính OOF nên hơi lạc quan; chỉ dùng để chẩn đoán.
  --compare A B.. So sánh các file JSON (A là mốc), có khoảng tin cậy bootstrap GHÉP CẶP theo clip.
  --selfcheck     Kiểm tra phase_features mới giống hệt bản cũ ở cấu hình mặc định.

Điểm chính (headline) = trung bình theo clip của tỉ lệ biên nằm trong ±0.2 s (biên bị bỏ sót tính là sai).
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import argparse, glob, json, time
import numpy as np
import train_phase_model as T
from stage3_features.phase_model.core.phase_dataset import load_all
from stage3_features.phase_model.core.phase_decoder import DurationPriors, PhaseDecoder
from stage3_features.phase_model.eval.phase_metrics import rep_boundaries
from stage3_features.phase_model.core.phase_labels import CLASSES, IGNORE_LABEL, N_CLASSES

TOLS_S = (0.1, 0.2, 0.3, 0.5)
HEADLINE_S = 0.2
SWEEP_SKIP = (2.0, 4.0, 8.0)
SWEEP_SCALE = (0.05, 0.1, 0.25, 0.5, 1.0)


# ----------------------------------------------------------------------------- #
# Chỉ số
# ----------------------------------------------------------------------------- #
def _hit(diff, fps, tol_s):
    return diff is not None and abs(diff) / fps <= tol_s


def clip_scores(per_clip, tol_s):
    """Mỗi clip: tỉ lệ biên (có trong nhãn thật) nằm trong ±tol_s. Biên bị bỏ sót = sai."""
    return np.array([np.mean([_hit(d, r['fps'], tol_s) for d in r['diffs'].values()]) for r in per_clip])


def boundary_table(per_clip, tol_s):
    names = []
    for r in per_clip:
        for b in r['diffs']:
            if b not in names:
                names.append(b)
    out = {}
    for b in names:
        rows = [r for r in per_clip if b in r['diffs']]
        hits = [_hit(r['diffs'][b], r['fps'], tol_s) for r in rows]
        secs = [abs(r['diffs'][b]) / r['fps'] for r in rows if r['diffs'][b] is not None]
        out[b] = {'within': float(np.mean(hits)), 'n': len(rows),
                  'miss': int(sum(r['diffs'][b] is None for r in rows)),
                  'median_abs_s': float(np.median(secs)) if secs else None}
    return out


def frame_tol_score(per_clip, tol_frames=3):
    """Điểm cũ theo frame (để đối chiếu với OOF clip accuracy trước đây)."""
    v = [np.mean([d is not None and abs(d) <= tol_frames for d in r['diffs'].values()]) for r in per_clip]
    return float(np.mean(v))


def boot_ci(x, seed=0, n=4000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- #
# Thu OOF + giải mã
# ----------------------------------------------------------------------------- #
def collect_oof(clips, people, a, cfg, clf_params):
    out = []
    for k, held in enumerate(people, 1):
        T.log(f'fold {k}/{len(people)} held={held}')
        train = [c for c in clips if c.person != held]
        test = [c for c in clips if c.person == held]
        clf, _, _ = T.fit_classifier(train, a.backend, a.seed, False, cfg, 0.0, clf_params)
        priors = DurationPriors.fit(T.durations(train))
        for c in test:
            X, _ = T.clip_features(c, cfg)
            out.append({'clip': c, 'probs': np.asarray(clf.predict_proba(X)), 'priors': priors})
    return out


def decode_all(oof, skip, scale):
    per_clip = []
    for r in oof:
        c = r['clip']
        dec = PhaseDecoder(priors=r['priors'], skip_penalty=skip, emission_scale=scale)
        rep = dec.decode(r['probs'], c.fps).to_rep()
        tb, pb = rep_boundaries(c.rep), rep_boundaries(rep)
        diffs = {b: (None if pb.get(b) is None else int(pb[b] - tb[b])) for b in tb if tb[b] is not None}
        per_clip.append({'stem': c.stem, 'person': c.person, 'fps': float(c.fps), 'diffs': diffs})
    return per_clip


def confusion(oof):
    m = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for r in oof:
        lab = np.asarray(r['clip'].pose_labels())
        ok = lab != IGNORE_LABEL
        np.add.at(m, (lab[ok].astype(int), r['probs'].argmax(1)[ok]), 1)
    return m


# ----------------------------------------------------------------------------- #
# Chế độ: chạy 1 cấu hình
# ----------------------------------------------------------------------------- #
def run_eval(a):
    clips = load_all(a.phases_dir, a.pose_cache_dir, group=a.group)
    clips = [c for c in clips if c.phaseset.marked_by in ('manual', 'auto_reviewed')]
    people = sorted({c.person for c in clips})
    if len(people) < 2:
        raise SystemExit('Cần ít nhất 2 người.')
    cfg = T.make_feature_cfg(a)
    clf_params = T.parse_kv(a.lgbm_params)
    tag = a.tag or 'run'
    T.log(f'[{tag}] {len(clips)} clips/{len(people)} people backend={a.backend} cfg={cfg.to_dict()} params={clf_params}')

    t0 = time.time()
    oof = collect_oof(clips, people, a, cfg, clf_params)
    per_clip = decode_all(oof, a.skip_penalty, a.emission_scale)

    print(f'\n=== [{tag}] Điểm theo giây (biên bị bỏ sót tính là sai; n clip = {len(per_clip)}) ===')
    tables = {}
    for tol in TOLS_S:
        tables[str(tol)] = boundary_table(per_clip, tol)
    names = list(tables[str(TOLS_S[0])].keys())
    print(f'{"biên":<22}' + ''.join(f'{"±%.1fs" % t:>8}' for t in TOLS_S) + f'{"miss":>6}{"median|e|(s)":>14}')
    for b in names:
        row = ''.join(f'{tables[str(t)][b]["within"]:>8.2f}' for t in TOLS_S)
        ref = tables[str(HEADLINE_S)][b]
        med = ref['median_abs_s']
        print(f'{b:<22}{row}{ref["miss"]:>6}{(f"{med:.3f}" if med is not None else "-"):>14}')
    means = {str(t): float(clip_scores(per_clip, t).mean()) for t in TOLS_S}
    print(f'{"TRUNG BÌNH (theo clip)":<22}' + ''.join(f'{means[str(t)]:>8.3f}' for t in TOLS_S))
    mean, lo, hi = boot_ci(clip_scores(per_clip, HEADLINE_S), a.seed)
    f3 = frame_tol_score(per_clip, 3)
    print(f'\nĐIỂM CHÍNH (±{HEADLINE_S}s): {mean:.3f}  CI95=({lo:.3f},{hi:.3f})')
    print(f'Đối chiếu điểm cũ (±3 frame): {f3:.3f}   (LightGBM mặc định trước đây = 0.417)')

    conf = confusion(oof)
    peak = float(np.mean([r['probs'].max(1).mean() for r in oof]))
    short = [CLASSES[i][:9] for i in range(N_CLASSES)]
    print(f'\n=== Ma trận nhầm lẫn theo frame (hàng = nhãn thật, % theo hàng; peak_prob={peak:.3f}) ===')
    print(f'{"":<12}' + ''.join(f'{s:>10}' for s in short) + f'{"n":>9}')
    for i in range(N_CLASSES):
        tot = max(int(conf[i].sum()), 1)
        print(f'{CLASSES[i][:11]:<12}' + ''.join(f'{100 * conf[i, j] / tot:>9.1f}%' for j in range(N_CLASSES)) + f'{tot:>9}')
    print(f'frame accuracy = {np.trace(conf) / max(conf.sum(), 1):.3f}')

    sweep = None
    if a.sweep:
        T.log('sweep decoder trên xác suất OOF đã lưu...')
        sweep = []
        print('\n=== Quét decoder (chẩn đoán; chọn trên chính OOF -> hơi lạc quan) ===')
        print(f'{"skip":>6}{"scale":>8}{"±0.2s":>9}{"±0.3s":>9}{"±3f":>9}')
        for sk in SWEEP_SKIP:
            for sc in SWEEP_SCALE:
                pc = decode_all(oof, sk, sc)
                row = {'skip': sk, 'scale': sc, 'w02': float(clip_scores(pc, 0.2).mean()),
                       'w03': float(clip_scores(pc, 0.3).mean()), 'f3': frame_tol_score(pc, 3)}
                sweep.append(row)
                print(f'{sk:>6.1f}{sc:>8.2f}{row["w02"]:>9.3f}{row["w03"]:>9.3f}{row["f3"]:>9.3f}')

    res = {'tag': tag, 'cfg': cfg.to_dict(), 'clf_params': clf_params, 'backend': a.backend, 'group': a.group,
           'seed': a.seed, 'skip_penalty': a.skip_penalty, 'emission_scale': a.emission_scale,
           'n_clips': len(per_clip), 'headline_tol_s': HEADLINE_S, 'headline': {'mean': mean, 'lo': lo, 'hi': hi},
           'frame3': f3, 'means': means, 'tables': tables, 'peak_prob': peak,
           'confusion': conf.tolist(), 'classes': list(CLASSES), 'per_clip': per_clip, 'sweep': sweep,
           'minutes': (time.time() - t0) / 60}
    out = a.out_dir / f'oof_{tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nĐã ghi {out}  ({res["minutes"]:.1f} phút)')


# ----------------------------------------------------------------------------- #
# Chế độ: so sánh
# ----------------------------------------------------------------------------- #
def run_compare(a):
    files = []
    for pat in a.compare:
        files += sorted(glob.glob(pat)) or [pat]
    runs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in files]
    tol = a.compare_tol
    base = {r['stem']: r for r in runs[0]['per_clip']}
    stems = sorted(base)
    print(f'\n=== So sánh @±{tol}s (mốc = {runs[0]["tag"]}; delta có CI95 bootstrap ghép cặp trên {len(stems)} clip) ===')
    bnames = list(boundary_table(runs[0]['per_clip'], tol).keys())
    print(f'{"tag":<16}{"điểm":>8}{"delta":>9}{"CI95 delta":>20}  ' + ' '.join(f'{b.split(":")[1][:7]:>8}' for b in bnames))
    b_scores = None
    for i, r in enumerate(runs):
        d = {x['stem']: x for x in r['per_clip']}
        pcs = [d[s] for s in stems if s in d]
        sc = clip_scores(pcs, tol)
        tab = boundary_table(pcs, tol)
        cols = ' '.join(f'{tab[b]["within"]:>8.2f}' for b in bnames)
        if i == 0:
            b_scores = sc
            print(f'{r["tag"]:<16}{sc.mean():>8.3f}{"-":>9}{"-":>20}  {cols}')
        else:
            if len(sc) == len(b_scores):
                m, lo, hi = boot_ci(sc - b_scores, a.seed)
                flag = ' *' if (lo > 0 or hi < 0) else ''
                print(f'{r["tag"]:<16}{sc.mean():>8.3f}{m:>+9.3f}{f"({lo:+.3f},{hi:+.3f})":>20}  {cols}{flag}')
            else:
                print(f'{r["tag"]:<16}{sc.mean():>8.3f}   (số clip khác mốc, bỏ qua delta)  {cols}')
    print('\n* = CI95 của delta không chứa 0 (khác biệt đáng tin). Không có dấu * -> chưa phân biệt được với nhiễu.')


# ----------------------------------------------------------------------------- #
# Chế độ: selfcheck
# ----------------------------------------------------------------------------- #
def run_selfcheck(a):
    import importlib
    import stage3_features.phase_model.core.phase_features as new
    clips = load_all(a.phases_dir, a.pose_cache_dir, group=a.group)
    c = clips[0]
    P = T.pose_array(c)
    ok = True
    try:
        old = importlib.import_module(a.old_features)
        x0, n0 = old.build_feature_matrix(P, c.fps, c.front_side_override, is_coach=c.group == 'coach', use_cache=False)
        x1, n1 = new.build_feature_matrix(P, c.fps, c.front_side_override, is_coach=c.group == 'coach', use_cache=False)
        same = n0 == n1 and x0.shape == x1.shape and np.array_equal(x0, x1, equal_nan=True)
        print(f'[1] cấu hình mặc định giống hệt bản cũ ({a.old_features}): {same}  shape={x1.shape}')
        ok &= same
    except ImportError:
        print(f'[1] BỎ QUA: không import được {a.old_features} (chưa sao lưu bản cũ?)')
    cfg = new.FeatureConfig(long_scales_s=(2.0, 4.0), time_features=True, include_torso_scale=False)
    x, nm = new.build_feature_matrix(P, c.fps, c.front_side_override, is_coach=c.group == 'coach', cfg=cfg, use_cache=False)
    good = x.shape[1] == len(nm) == len(new.feature_names(cfg)) and nm == new.feature_names(cfg)
    print(f'[2] cấu hình mở rộng: {x.shape[1]} cột, tên khớp: {good}')
    rt = new.FeatureConfig.from_dict(cfg.to_dict()) == cfg
    print(f'[3] cfg <-> dict (bundle) khứ hồi: {rt}')
    ok &= good and rt
    print('SELFCHECK PASS' if ok else 'SELFCHECK FAIL')
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--phases-dir', type=Path, default=Path('data/phases'))
    ap.add_argument('--pose-cache-dir', type=Path, default=Path('data/pose_cache'))
    ap.add_argument('--out-dir', type=Path, default=Path('reports'))
    ap.add_argument('--group', choices=('all', 'coach', 'trainee'), default='coach')
    ap.add_argument('--backend', choices=('lightgbm', 'logreg'), default='lightgbm')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-penalty', type=float, default=4.0)
    ap.add_argument('--emission-scale', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--long-scales', type=float, nargs='*', default=[])
    ap.add_argument('--time-features', action='store_true')
    ap.add_argument('--no-torso-scale', action='store_true')
    ap.add_argument('--lgbm-params', nargs='*', default=[])
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--compare', nargs='+', metavar='JSON')
    ap.add_argument('--compare-tol', type=float, default=HEADLINE_S)
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--old-features', default='stage3_features.phase_features_old')
    a = ap.parse_args()
    if a.compare:
        run_compare(a)
    elif a.selfcheck:
        run_selfcheck(a)
    else:
        run_eval(a)


if __name__ == '__main__':
    main()
