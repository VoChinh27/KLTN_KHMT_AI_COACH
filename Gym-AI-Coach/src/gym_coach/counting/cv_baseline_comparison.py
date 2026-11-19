"""Additive user-run inner-CV baselines. Never modify frozen research/inner_cv modules."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

from .models import canonical_hash, STATUS
from .readiness import file_info, write_json
from .research import baselines
from .research.data import read, identity, check_identity, local
from .inner_cv import manifest as cv
from .inner_cv.oof import pair_identity, valid_prediction

VERSION = 'squat-count-cv-baseline-comparison-v1'
MANIFEST_HASH = '8f98e626657b64c038d07a8085d64d452fc2715c6f18ec49c02995ee335352bf'
EPOCH_INDEX = 49
METHODS = ('constant', 'duration')
FORMULAS = dict(constant='median(train_targets)',
    duration='OLS: slope=sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))**2); intercept=mean(y)-slope*mean(x)',
    duration_seconds='source.frame_count/source.fps',
    degenerate='zero duration variance: slope=0, intercept=mean(train_targets)',
    output='max(0, raw_prediction); continuous, no rounding')


def tool_identity():
    return dict(version=VERSION, module=file_info(__file__), formula_module=file_info(baselines.__file__))


def validate_partition(manifest):
    selected, _, folds = cv.partition(manifest['rows'])
    if (len(manifest['rows']) != 33 or selected != manifest['rows'] or folds != manifest['folds']
            or len({r['source']['group_id'] for r in selected}) != 12):
        raise ValueError('CV_PARTITION_MISMATCH: exactly the frozen 33 outer-train clips/12groups required')
    for row in selected:
        source = row['source']
        if (type(source['frame_count']) is not int or source['frame_count'] <= 0
                or type(source['fps']) not in (int, float) or not math.isfinite(source['fps']) or source['fps'] <= 0):
            raise ValueError('INVALID_DURATION_SOURCE: '+source['video_id'])
        duration = source['frame_count']/source['fps']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('INVALID_DURATION: '+source['video_id'])
    return {r['source']['video_id']: r for r in selected}


def paired_records(manifest, records):
    """Require exact one-to-one correspondence; errors never reduce the denominator."""
    rows = validate_partition(manifest)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError('OOF_RECORD_FORMAT')
    ids = [r.get('video_id') for r in records]
    counts = Counter(ids)
    if len(records) != 33 or set(ids) != set(rows) or any(n != 1 for n in counts.values()):
        raise ValueError('OOF_COVERAGE: missing='+str(sorted(set(rows)-set(ids)))+
                         '; unexpected='+str(sorted(set(ids)-set(rows), key=str))+
                         '; duplicates='+str([v for v, n in counts.items() if n != 1]))
    fold_for = {v: f['fold'] for f in manifest['folds'] for v in f['evaluation_ids']}
    indexed = {}
    for record in records:
        video_id = record['video_id']
        expected = pair_identity(rows[video_id])
        if (canonical_hash({k: record.get(k) for k in expected}) != canonical_hash(expected)
                or type(record.get('fold')) is not int or record['fold'] != fold_for[video_id]):
            raise ValueError('OOF_LINKAGE: '+video_id)
        if record.get('status') != 'estimated' or record.get('reason') is not None or not valid_prediction(record.get('prediction')):
            raise ValueError('OOF_UNAVAILABLE_OR_INVALID: '+video_id)
        indexed[video_id] = record
    return [indexed[v] for v in sorted(rows)]


def metrics(manifest, records):
    records = paired_records(manifest, records)
    def summarize(items):
        return dict(clips=len(items), continuous_MAE=math.fsum(abs(r['prediction']-r['target']) for r in items)/len(items))
    groups = {g: summarize([r for r in records if r['group_id'] == g]) for g in sorted({r['group_id'] for r in records})}
    return dict(pooled=summarize(records), groups=groups,
        macro_group_MAE=math.fsum(g['continuous_MAE'] for g in groups.values())/12,
        group_denominator=12,
        folds={str(f): summarize([r for r in records if r['fold'] == f]) for f in range(3)})


def selected_gru(manifest, oof):
    if not isinstance(oof.get('selection'), dict):
        raise ValueError('GRU_SELECTION_OR_SCOPE: no saved selection')
    if (oof.get('status') != 'COMPLETE' or oof.get('manifest_sha256') != canonical_hash(manifest)
            or oof.get('selection', {}).get('epoch') != EPOCH_INDEX
            or oof['selection'].get('epoch_one_based') != 50
            or oof.get('complete_epochs') != 50 or len(oof.get('curve', [])) != 50
            or any(oof.get(k) is not False for k in ('outer_validation_used', 'development_test_used', 'final_refit'))):
        raise ValueError('GRU_SELECTION_OR_SCOPE: require saved epoch index49/50th, complete train-only OOF')
    if [e['epoch'] for e in oof['curve']] != list(range(50)) or not all(e['complete'] for e in oof['curve']):
        raise ValueError('GRU_EPOCH_COVERAGE')
    selected = oof['curve'][EPOCH_INDEX]
    records = paired_records(manifest, selected['records'])
    measured = metrics(manifest, records)
    if (measured['pooled']['continuous_MAE'] != selected['pooled']['continuous_MAE']
            or measured['pooled']['continuous_MAE'] != oof['selection']['pooled_oof_MAE']):
        raise ValueError('GRU_SAVED_METRIC_MISMATCH')
    return records


def verify_inputs(root, manifest_path, oof_path):
    """Read/hash and replay saved OOF reports only. No fitting, model forward or learning."""
    from .inner_cv.runtime import aggregate_checkpoints
    root = Path(root).resolve()
    manifest_path, oof_path = local(root, manifest_path), local(root, oof_path)
    manifest, _ = cv.load(root, manifest_path)
    if canonical_hash(manifest) != MANIFEST_HASH:
        raise ValueError('UNAPPROVED_MANIFEST: frozen input must not be replaced')
    validate_partition(manifest)
    oof = cv.unseal(read(oof_path))
    sources = oof.get('checkpoint_sources', [])
    if (len(sources) != 3 or [s['fold'] for s in sources] != [0, 1, 2]
            or any(s['epoch'] != EPOCH_INDEX for s in sources)):
        raise ValueError('CHECKPOINT_SOURCE_COVERAGE: exactly one final checkpoint per fold')
    checkpoint_paths = [check_identity(root, s) for s in sources]
    replay = aggregate_checkpoints(manifest, checkpoint_paths)
    # Normalize paths only, after byte/size and root containment were verified.
    for item, source in zip(replay['checkpoint_sources'], sources):
        item['path'] = source['path']
    if canonical_hash(replay) != canonical_hash(oof):
        raise ValueError('OOF_CHECKPOINT_MISMATCH: saved report differs from checkpoint history')
    records = selected_gru(manifest, oof)
    receipt = dict(version=VERSION, **STATUS, tool=tool_identity(),
        manifest=identity(root, manifest_path), manifest_sha256=canonical_hash(manifest),
        gru_oof=identity(root, oof_path), gru_oof_sha256=canonical_hash(oof),
        dataset=manifest['dataset'], checkpoint_sources=sources,
        selected_epoch_index=EPOCH_INDEX, selected_epoch_one_based=50,
        gru_metrics=metrics(manifest, records), folds=manifest['folds'],
        parent_cohort=112, comparison_clips=33, comparison_groups=12,
        real_baseline_fits=0, training_runs=0, model_forward_calls=0,
        outer_validation_used=False, development_test_used=False,
        development_test_previously_exposed=True,
        limitation='AI weak references; OOF selected epoch is not independent final evaluation')
    return manifest, oof, receipt


def fit_inner_baselines(manifest, *, user_run=False):
    if not user_run:
        raise PermissionError('Baseline parameter fitting is user-run only; --user-run required')
    rows = validate_partition(manifest)
    folds, predictions = [], {name: [] for name in METHODS}
    for fold in manifest['folds']:
        train = [rows[v] for v in fold['train_ids']]
        evaluation = [rows[v] for v in fold['evaluation_ids']]
        # Both subsets keep original outer role=train. Only the explicit inner train rows are passed to fit.
        parameters = baselines.fit_baselines(train)
        if parameters['training_ids'] != sorted(fold['train_ids']):
            raise ValueError('BASELINE_FIT_SCOPE_MISMATCH')
        if not all(type(parameters[k]) in (int, float) and math.isfinite(parameters[k]) for k in
                   ('constant', 'duration_intercept', 'duration_slope')):
            raise ValueError('NONFINITE_BASELINE_PARAMETERS')
        origin = dict(fold=fold['fold'], train_ids=fold['train_ids'], evaluation_ids=fold['evaluation_ids'],
            train_groups=fold['train_groups'], evaluation_groups=fold['evaluation_groups'],
            train_records=[dict(**pair_identity(r), source=r['source'],
                                duration_seconds=r['source']['frame_count']/r['source']['fps']) for r in train],
            evaluation_records=[dict(**pair_identity(r), source=r['source'],
                                     duration_seconds=r['source']['frame_count']/r['source']['fps']) for r in evaluation],
            parameters=parameters, parameter_sha256=canonical_hash(parameters),
            train_rows_sha256=canonical_hash(train), evaluation_rows_sha256=canonical_hash(evaluation),
            formula_source=file_info(baselines.__file__), formulas=FORMULAS)
        folds.append(origin)
        for name in METHODS:
            values = baselines.predict(parameters, evaluation, 'train', name)
            if set(values) != set(fold['evaluation_ids']):
                raise ValueError('BASELINE_PREDICTION_SCOPE_MISMATCH')
            for r in evaluation:
                predictions[name].append(dict(**pair_identity(r), fold=fold['fold'],
                    prediction=values[r['source']['video_id']], status='estimated', reason=None,
                    parameter_sha256=origin['parameter_sha256'], fold_provenance_sha256=canonical_hash(origin)))
    for name in METHODS:
        predictions[name] = paired_records(manifest, predictions[name])
    return dict(version=VERSION, **STATUS, manifest_sha256=canonical_hash(manifest),
        folds=folds, predictions=predictions, prediction_hashes={n: canonical_hash(v) for n, v in predictions.items()},
        fitting_scope='each fold train_ids only; no full33 fit', parameter_sets=3, methods_per_set=list(METHODS),
        labels='AI weak reference; not independent ground truth', outer_validation_used=False, development_test_used=False)


def compare(manifest, gru_records, baseline_result):
    gru_records = paired_records(manifest, gru_records)
    if baseline_result['manifest_sha256'] != canonical_hash(manifest):
        raise ValueError('BASELINE_MANIFEST_MISMATCH')
    if set(baseline_result['predictions']) != set(METHODS) or len(baseline_result['folds']) != 3:
        raise ValueError('BASELINE_METHOD_OR_FOLD_COVERAGE')
    rows = validate_partition(manifest)
    for origin, fold in zip(baseline_result['folds'], manifest['folds']):
        p = origin['parameters']
        train = [rows[v] for v in fold['train_ids']]
        evaluation = [rows[v] for v in fold['evaluation_ids']]
        def origin_records(items):
            return [dict(**pair_identity(r), source=r['source'],
                         duration_seconds=r['source']['frame_count']/r['source']['fps']) for r in items]
        expected_ids_hash = canonical_hash(sorted((r['source']['video_id'], r['source']['sha256'], r['target']) for r in train))
        if (any(origin[k] != fold[k] for k in ('fold', 'train_ids', 'evaluation_ids', 'train_groups', 'evaluation_groups'))
                or p['training_ids'] != sorted(fold['train_ids']) or p['training_subset_hash'] != expected_ids_hash
                or origin['parameter_sha256'] != canonical_hash(p)
                or origin['train_rows_sha256'] != canonical_hash(train)
                or origin['evaluation_rows_sha256'] != canonical_hash(evaluation)
                or origin['train_records'] != origin_records(train) or origin['evaluation_records'] != origin_records(evaluation)
                or origin['formula_source'] != file_info(baselines.__file__) or origin['formulas'] != FORMULAS):
            raise ValueError('BASELINE_PARAMETER_PROVENANCE: fold '+str(fold['fold']))
    summaries = dict(gru=metrics(manifest, gru_records))
    pairs = {}
    for name in METHODS:
        records = paired_records(manifest, baseline_result['predictions'][name])
        if canonical_hash(records) != baseline_result['prediction_hashes'][name]:
            raise ValueError('BASELINE_PREDICTION_HASH: '+name)
        for r in records:
            origin = baseline_result['folds'][r['fold']]
            if r['parameter_sha256'] != origin['parameter_sha256'] or r['fold_provenance_sha256'] != canonical_hash(origin):
                raise ValueError('BASELINE_PREDICTION_PROVENANCE: '+r['video_id'])
            # Prediction replay uses saved coefficients only; never fits to evaluation labels.
            expected = baselines.predict(origin['parameters'], [rows[r['video_id']]], 'train', name)[r['video_id']]
            if r['prediction'] != expected:
                raise ValueError('BASELINE_PREDICTION_VALUE: '+r['video_id'])
        summaries[name] = metrics(manifest, records)
        paired = []
        for gru, baseline in zip(gru_records, records):
            ge, be = abs(gru['prediction']-gru['target']), abs(baseline['prediction']-baseline['target'])
            paired.append(dict(**pair_identity(rows[gru['video_id']]), fold=gru['fold'],
                gru_prediction=gru['prediction'], baseline_prediction=baseline['prediction'],
                gru_signed_error=gru['prediction']-gru['target'], baseline_signed_error=baseline['prediction']-baseline['target'],
                gru_absolute_error=ge, baseline_absolute_error=be, gru_minus_baseline_absolute_error=ge-be))
        pairs[name] = dict(records=paired, clips=33,
            mean_gru_minus_baseline_absolute_error=math.fsum(r['gru_minus_baseline_absolute_error'] for r in paired)/33,
            delta_interpretation='negative = smaller GRU absolute error on this paired clip; not independent accuracy')
    return dict(version=VERSION, **STATUS, manifest_sha256=canonical_hash(manifest),
        epoch_index=EPOCH_INDEX, epoch_one_based=50, metrics=summaries, paired=pairs,
        prediction_hashes=dict(gru=canonical_hash(gru_records), **baseline_result['prediction_hashes']),
        labels='AI weak reference; not independent ground truth',
        limitation='The same OOF was used to select epoch50. This is descriptive paired comparison, not independent final evaluation.',
        outer_validation_used=False, development_test_used=False, development_test_previously_exposed=True,
        final_refit=False, statistical_significance_test=False)


def report_markdown(result):
    lines = ['# So sánh baseline nội bộ CV với GRU epoch50 (index49)', '',
        'AI weak reference; NOT_VALIDATED, official=false, assessment=false. G5.7 blocked, G6 IN_PROGRESS.',
        'OOF đã dùng chọn epoch; không phải đánh giá độc lập cuối cùng hoặc unseen-subject holdout.',
        'Development-test đã exposed trước đây, không dùng trong phép so sánh này. Không final refit.', '',
        '|Phương pháp|MAE clip (n=33)|Macro-MAE (12 groups)|Fold0 (n=13)|Fold1 (n=13)|Fold2 (n=7)|',
        '|---|---:|---:|---:|---:|---:|']
    for name, value in result['metrics'].items():
        numbers = [value['pooled']['continuous_MAE'], value['macro_group_MAE'],
                   *(value['folds'][str(f)]['continuous_MAE'] for f in range(3))]
        lines.append('|'+name+'|'+'|'.join(f'{v:.9f}' for v in numbers)+'|')
    lines += ['', 'Pooled MAE là tổng33 absolute errors/33; macro là trung bình12 group-MAE, mỗi nhóm trọng số bằng nhau.',
        'Mỗi baseline fit riêng trên20/20/26 train clips rồi predict13/13/7 held-out clips; không fit toàn33.',
        'Duration = frame_count/FPS; OLS có intercept, variance=0 dùng slope0/mean target; clamp prediction≥0.', '',
        '|Nhóm|n|GRU MAE|Constant MAE|Duration MAE|', '|---|---:|---:|---:|---:|']
    for group, value in result['metrics']['gru']['groups'].items():
        lines.append('|'+group+'|'+str(value['clips'])+'|'+'|'.join(f"{result['metrics'][n]['groups'][group]['continuous_MAE']:.9f}" for n in ('gru', *METHODS))+'|')
    lines += ['', '## Sai số ghép cặp cùng video', '',
              'Δ = |GRU−AI target| − |baseline−AI target|. Âm nghĩa GRU sai số nhỏ hơn trên clip đó; không là claim accuracy.']
    for name, pair in result['paired'].items():
        lines += ['', f"### {name}; mean Δ trên33clips = {pair['mean_gru_minus_baseline_absolute_error']:.9f}", '',
            '|Video|Fold|AI target|GRU|Baseline|AE GRU|AE baseline|Δ|', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in pair['records']:
            video_id = r['video_id'].replace('|', '&#124;').replace('\n', ' ')
            lines.append(f"|{video_id}|{r['fold']}|{r['target']}|"+'|'.join(f'{r[k]:.9f}' for k in
                ('gru_prediction', 'baseline_prediction', 'gru_absolute_error', 'baseline_absolute_error', 'gru_minus_baseline_absolute_error'))+'|')
    lines += ['', 'Provenance/hashes/IDs/parameters: `baselines.json`; full metrics và pair linkage: `comparison.json`.',
              'Mọi missing/duplicate/hash/linkage mismatch phải làm command fail; không loại clip để tăng kết quả.', '']
    return '\n'.join(lines)


def run_comparison(root, verification_path, out, *, user_run=False):
    if not user_run:
        raise PermissionError('Real baseline fitting is reserved for user; --user-run required')
    receipt = cv.unseal(read(verification_path))
    if receipt['version'] != VERSION or receipt['tool'] != tool_identity():
        raise ValueError('VERIFICATION_TOOL_MISMATCH')
    manifest_path = check_identity(root, receipt['manifest'])
    oof_path = check_identity(root, receipt['gru_oof'])
    manifest, oof, current = verify_inputs(root, manifest_path, oof_path)
    if canonical_hash(receipt) != canonical_hash(current):
        raise ValueError('VERIFICATION_INPUT_MISMATCH')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'inputs.json', cv.sealed(dict(verification=identity(root, verification_path), evidence=current)))
    baseline_result = fit_inner_baselines(manifest, user_run=True)
    write_json(out/'baselines.json', cv.sealed(baseline_result))
    result = compare(manifest, selected_gru(manifest, oof), baseline_result)
    result['inputs'] = current
    write_json(out/'comparison.json', cv.sealed(result))
    with (out/'REPORT.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(report_markdown(result))
    write_json(out/'inventory.json', dict(excludes_self=True, files={p.name: file_info(p) for p in sorted(out.iterdir()) if p.is_file()}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'compare'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--gru-oof', type=Path)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--user-run', action='store_true')
    args = parser.parse_args(argv)
    start = time.perf_counter()
    command = [sys.executable, '-m', __name__ if __name__ != '__main__' else 'gym_coach.counting.cv_baseline_comparison',
               *(argv if argv is not None else sys.argv[1:])]
    if args.command == 'compare':
        if not args.user_run or not args.verification or args.manifest or args.gru_oof:
            parser.error('compare requires --user-run --verification; manifest/OOF are bound by verification')
        result = run_comparison(args.root, args.verification, args.output_dir, user_run=args.user_run)
        summary = dict(status='COMPLETE', metrics=result['metrics'], baseline_parameter_sets=3,
                       gru_training_runs=0, final_refit=False)
    else:
        if not args.manifest or not args.gru_oof or args.verification or args.user_run:
            parser.error('verify requires --manifest --gru-oof; no fitting options permitted')
        _, _, result = verify_inputs(args.root, args.manifest, args.gru_oof)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir/'verification.json', cv.sealed(result))
        summary = dict(status='VERIFIED_NOT_FITTED', gru_epoch_one_based=50, clips=33,
                       real_baseline_fits=0, gru_training_runs=0)
    write_json(args.output_dir/'receipt.json', dict(command=command, seconds=time.perf_counter()-start, result=summary))
    print(json.dumps(dict(output_dir=str(args.output_dir), **summary), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
