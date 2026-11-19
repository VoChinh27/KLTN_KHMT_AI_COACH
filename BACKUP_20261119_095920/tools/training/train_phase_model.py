"""Huấn luyện frame classifier, đánh giá LOPO và lưu bundle phiên bản bất biến."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, ast, json, os, re, tempfile, time
from typing import Any, Dict, List, Sequence, Tuple
import joblib
import numpy as np
from core.phase_schema import PHASE_ORDER
from stage3_features.phase_model.core.phase_dataset import load_all
from stage3_features.phase_model.core.phase_features import FeatureConfig, build_feature_matrix
from stage3_features.phase_model.core.phase_classifier import FramePhaseClassifier, boundary_proximity_weights
from stage3_features.phase_model.core.phase_labels import IGNORE_LABEL
from stage3_features.phase_model.core.phase_decoder import DurationPriors, PhaseDecoder
from stage3_features.phase_model.eval.phase_confidence import PlattCalibrator
from stage3_features.phase_model.eval.phase_quality import compute_feature_stats
from stage3_features.phase_model.eval.phase_metrics import boundary_summary, rep_boundaries, per_clip_within_tol, bootstrap_ci
from stage3_features.phase_model.support.phase_common import read_json, write_json_atomic
from stage3_features.phase_model.inference.phase_segmenter import load_registry, write_registry

BUNDLE_VERSION = 1


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def pose_array(clip):
    return np.stack([
        np.concatenate((p.xy, np.asarray(p.conf).reshape(-1, 1)), axis=1)
        for p in clip.poses
    ]).astype(np.float32)


# Cache feature theo clip: khi KHÔNG augment, feature của mỗi clip là cố định,
# nên chỉ cần tính đúng 1 lần cho toàn bộ các fold LOPO.
_FEAT_CACHE: Dict[Tuple[str, Any], Tuple[np.ndarray, List[str]]] = {}


def parse_kv(items):
    """['n_estimators=150', 'num_leaves=7'] -> {'n_estimators': 150, 'num_leaves': 7}"""
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def make_feature_cfg(args):
    return FeatureConfig(long_scales_s=tuple(args.long_scales), time_features=bool(args.time_features),
                         include_torso_scale=not args.no_torso_scale)


def clip_features(clip, cfg):
    key = (clip.stem, cfg)
    if key not in _FEAT_CACHE:
        _FEAT_CACHE[key] = build_feature_matrix(
            pose_array(clip), clip.fps, clip.front_side_override,
            is_coach=clip.group == 'coach', cfg=cfg, use_cache=False)
    return _FEAT_CACHE[key]


def prepare_clip(clip, cfg, augment=False, seed=0, feedback_weight=0.0):
    labels = clip.pose_labels()
    if augment:
        from stage3_features.phase_model.support.phase_augment import augment_clip, AugmentConfig
        poses = pose_array(clip)
        poses, labels = augment_clip(poses, labels, np.random.default_rng(seed),
                                     AugmentConfig(fps=clip.fps), fps=clip.fps)
        X, names = build_feature_matrix(poses, clip.fps, clip.front_side_override,
                                        is_coach=clip.group == 'coach', cfg=cfg, use_cache=False)
    else:
        X, names = clip_features(clip, cfg)
    keep = labels != IGNORE_LABEL
    clip_weight = 1.0 + float(feedback_weight) * np.sqrt(max(0, int(getattr(clip, "review_edit_count", 0))))
    frame_weights = boundary_proximity_weights(labels, clip.fps)
    return X[keep], labels[keep], frame_weights[keep] * clip_weight, names, X


def durations(clips):
    d = {p: [] for p in PHASE_ORDER}
    for c in clips:
        for n, w in c.rep.present():
            if c.fps > 0:
                d[n].append(w.n_frames / c.fps)
    return d


def tune_decoder_parameters(clips, backend, seed, augment, cfg, tolerance,
                            feedback_weight, base_skip, base_scale, tune=True, clf_params=None):
    """Inner leave-one-person-out để chọn decoder params, độc lập outer test."""
    people = sorted({c.person for c in clips})
    defaults = {"skip_penalty": float(base_skip), "emission_scale": float(base_scale), "prior_power": 0.0}
    if not tune:
        return defaults, {"reason": "tune_disabled"}
    if len(people) < 3:
        return defaults, {"reason": "not_enough_inner_people"}
    grid = []
    for skip in sorted({max(0.5, float(base_skip) * 0.5), float(base_skip), float(base_skip) * 1.5}):
        for scale in sorted({max(0.1, float(base_scale) * 0.7), float(base_scale), float(base_scale) * 1.3}):
            grid.append({"skip_penalty": skip, "emission_scale": scale, "prior_power": 0.0})
    scores = {str((g["skip_penalty"], g["emission_scale"])): [] for g in grid}
    for i, held in enumerate(people, 1):
        log(f'    inner {i}/{len(people)} held={held}')
        inner_train = [c for c in clips if c.person != held]
        inner_valid = [c for c in clips if c.person == held]
        if not inner_train:
            continue
        clf, _, _ = fit_classifier(inner_train, backend, seed, augment, cfg, feedback_weight, clf_params)
        prior = DurationPriors.fit(durations(inner_train))
        X_cache = []
        for c in inner_valid:
            X, _ = clip_features(c, cfg)
            X_cache.append((c, clf.predict_proba(X)))
        for params in grid:
            decoder = PhaseDecoder(priors=prior, skip_penalty=params["skip_penalty"],
                                   emission_scale=params["emission_scale"])
            key = str((params["skip_penalty"], params["emission_scale"]))
            for c, probs in X_cache:
                pred = decoder.decode(probs, c.fps).to_rep()
                scores[key].append(float(per_clip_within_tol([c.rep], [pred], tolerance)[0]))
    means = {k: (float(np.nanmean(v)) if v else float('-inf')) for k, v in scores.items()}
    best = max(grid, key=lambda g: means[str((g["skip_penalty"], g["emission_scale"]))])
    return best, {"inner_mean_within_tol": means, "grid": grid}


def fit_classifier(clips, backend, seed, augment, cfg, feedback_weight=0.0, clf_params=None):
    xs = []; ys = []; ws = []; names = None; all_x = []
    for i, c in enumerate(clips):
        X, y, w, n, Xfull = prepare_clip(c, cfg, augment, seed + i, feedback_weight)
        if len(y):
            xs.append(X); ys.append(y); ws.append(w); names = n; all_x.append(Xfull)
    if not xs:
        raise ValueError('Fold không có frame có nhãn để huấn luyện')
    clf = FramePhaseClassifier(backend=backend, params=clf_params, random_state=seed).fit(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), names)
    return clf, names, all_x


def version_number(registry):
    nums = [int(m.group(1)) for key in registry.get('versions', {})
            if (m := re.fullmatch(r'v(\d+)', key))]
    return max(nums, default=0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phases-dir', type=Path, default=Path('data/phases'))
    ap.add_argument('--pose-cache-dir', type=Path, default=Path('data/pose_cache'))
    ap.add_argument('--models-dir', type=Path, default=Path('models/phase_segmenter'))
    ap.add_argument('--group', choices=('all', 'coach', 'trainee'), default='all')
    ap.add_argument('--backend', choices=('lightgbm', 'logreg'), default='lightgbm')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tolerance-frames', type=int, default=3)
    ap.add_argument('--skip-penalty', type=float, default=4.0)
    ap.add_argument('--emission-scale', type=float, default=0.5)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--no-tune', action='store_true',
                    help='Bỏ inner LOPO tune decoder (dùng skip-penalty/emission-scale cố định) -> nhanh hơn rất nhiều')
    ap.add_argument('--long-scales', type=float, nargs='*', default=[],
                    help='thang thời gian dài (giây) cho feature ngữ cảnh, vd: --long-scales 2 4')
    ap.add_argument('--time-features', action='store_true',
                    help='thêm t_sec, t_frac, t_left_sec, cum_motion_frac')
    ap.add_argument('--no-torso-scale', action='store_true',
                    help='bỏ feature abs_log_torso_px (hằng số theo clip, dễ gây nhớ clip)')
    ap.add_argument('--lgbm-params', nargs='*', default=[],
                    help='ghi đè tham số classifier, vd: n_estimators=150 num_leaves=7 min_child_samples=100')
    ap.add_argument('--feedback-log', type=Path)
    ap.add_argument('--feedback-weight', type=float, default=0.5)
    args = ap.parse_args()
    tune = not args.no_tune
    augment = not args.no_augment
    t0 = time.time()

    clips = load_all(args.phases_dir, args.pose_cache_dir, group=args.group)
    clips = [c for c in clips if c.phaseset.marked_by in ('manual', 'auto_reviewed')]
    if args.feedback_log and args.feedback_log.exists():
        feedback = {}
        for line in args.feedback_log.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stem = Path(str(row.get('video', ''))).stem
            feedback[stem] = max(feedback.get(stem, 0), int(row.get('n_boundaries_edited', 0)))
        for c in clips:
            c.review_edit_count = feedback.get(c.stem, 0)
    people = sorted({c.person for c in clips})
    if len(people) < 2:
        raise SystemExit('Cần ít nhất hai người khác nhau để đánh giá LOPO.')
    log(f'loaded {len(clips)} clips / {len(people)} people; backend={args.backend} '
        f'augment={augment} tune={tune} group={args.group}')

    cfg = make_feature_cfg(args); clf_params = parse_kv(args.lgbm_params)
    truth = []; pred = []; scores = []; correct = []; folds = []
    log(f'feature_cfg={cfg.to_dict()} clf_params={clf_params}')
    # Outer leave-one-person-out: calibration/evaluation predictions are OOF.
    for k, held in enumerate(people, 1):
        log(f'outer fold {k}/{len(people)} held={held} (elapsed {(time.time()-t0)/60:.1f} min)')
        train = [c for c in clips if c.person != held]
        test = [c for c in clips if c.person == held]
        fold_params, inner_scores = tune_decoder_parameters(
            train, args.backend, args.seed, augment, cfg, args.tolerance_frames,
            args.feedback_weight, args.skip_penalty, args.emission_scale, tune=tune, clf_params=clf_params)
        clf, names, _ = fit_classifier(train, args.backend, args.seed, augment, cfg, args.feedback_weight, clf_params)
        priors = DurationPriors.fit(durations(train))
        decoder = PhaseDecoder(priors=priors, skip_penalty=fold_params["skip_penalty"],
                               emission_scale=fold_params["emission_scale"])
        for c in test:
            Xfull, feature_names = clip_features(c, cfg)
            result = decoder.decode(clf.predict_proba(Xfull), c.fps, compute_posteriors=True)
            rep = result.to_rep(); truth.append(c.rep); pred.append(rep)
            conf = result.boundary_confidence(tol=args.tolerance_frames)
            tb = rep_boundaries(c.rep); pb = rep_boundaries(rep)
            for bname in tb:
                if tb[bname] is None or pb[bname] is None:
                    continue
                phase = bname.split(':', 1)[1] if bname.startswith('start:') else None
                if phase:
                    score = conf.get(phase, {}).get('start', float('nan'))
                else:
                    last = c.rep.present()[-1][0]
                    score = conf.get(last, {}).get('end', float('nan'))
                if np.isfinite(score):
                    scores.append(float(score))
                    correct.append(float(abs(pb[bname] - tb[bname]) <= args.tolerance_frames))
        folds.append({'held_out_person': held, 'n_train_clips': len(train), 'n_test_clips': len(test),
                      'backend': clf.backend, 'decoder_params': fold_params, 'inner_search': inner_scores})

    metrics = boundary_summary(truth, pred, args.tolerance_frames)
    per = per_clip_within_tol(truth, pred, args.tolerance_frames)
    mean, lo, hi = bootstrap_ci(per, seed=args.seed)
    calibrator = PlattCalibrator().fit(scores, correct) if len(scores) >= 10 and len(set(correct)) > 1 else None

    # Final model is fit on all approved labels after OOF evaluation.
    log('final fit on all clips')
    final_params, final_search = tune_decoder_parameters(
        clips, args.backend, args.seed, augment, cfg, args.tolerance_frames,
        args.feedback_weight, args.skip_penalty, args.emission_scale, tune=tune, clf_params=clf_params)
    final_clf, feature_names, all_X = fit_classifier(clips, args.backend, args.seed, augment, cfg, args.feedback_weight, clf_params)
    prior_obj = DurationPriors.fit(durations(clips))
    duration_bounds = {i + 1: [prior_obj.priors[p].min_s, prior_obj.priors[p].max_s]
                       for i, p in enumerate(PHASE_ORDER)}
    bundle = {
        'bundle_version': BUNDLE_VERSION, 'classifier': final_clf, 'feature_names': feature_names,
        'feature_cfg': cfg.to_dict(), 'duration_priors': duration_bounds,
        'duration_priors_hsmm': prior_obj.to_dict(), 'decoder_params': final_params,
        'calibrator': calibrator.to_dict() if calibrator else None,
        'feature_stats': compute_feature_stats(all_X, feature_names),
        'tolerance_frames': args.tolerance_frames,
        'training_stems': [c.stem for c in clips], 'training_people': people,
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(args.models_dir)
    ver = f'v{version_number(registry)}'
    model_name = f'{ver}.joblib'
    model_path = args.models_dir / model_name
    fd, tmp = tempfile.mkstemp(prefix=model_name, suffix='.tmp', dir=args.models_dir)
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, model_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    meta = {
        'version': ver, 'model': model_name, 'backend': final_clf.backend,
        'n_clips': len(clips), 'n_people': len(people),
        'phase_sources': sorted({c.phaseset.marked_by for c in clips}),
        'oof_metrics': metrics, 'within_tol_clip_mean': mean, 'within_tol_clip_ci95': [lo, hi],
        'folds': folds, 'final_decoder_search': final_search, 'seed': args.seed,
        'tuned_decoder': tune, 'augment': augment, 'group': args.group,
        'feature_cfg': cfg.to_dict(), 'clf_params': clf_params, 'n_features': len(feature_names),
    }
    write_json_atomic(args.models_dir / f'{ver}.meta.json', meta)
    registry.setdefault('versions', {})[ver] = {'model': model_name, 'meta': f'{ver}.meta.json'}
    registry['current'] = ver
    write_registry(args.models_dir, registry)
    print(f'trained={len(clips)} clips/{len(people)} people; model={model_path}; '
          f'OOF clip accuracy={mean:.3f} CI95=({lo:.3f},{hi:.3f}); total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
