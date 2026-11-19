"""Saved inner-CV paired-error diagnosis. No fitting, forward, training or model changes."""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from gym_coach.counting import cv_baseline_comparison as comparison_tool
from gym_coach.counting.inner_cv import manifest as cv
from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.readiness import file_info, write_json
from gym_coach.counting.research.data import read, check_identity, identity
from gym_coach.counting.research.evaluation import rounded_count

MODELS = ('gru', 'constant', 'duration')


@contextmanager
def no_learning_or_forward():
    import torch
    from gym_coach.counting.research import baselines, experiment, network
    from gym_coach.counting.inner_cv import runtime
    def guard():
        def blocked(*args, **kwargs):
            raise AssertionError('Diagnosis forbids fitting, training, gradients, optimizer updates and model forward')
        return blocked
    with ExitStack() as stack:
        for owner, name in ((baselines, 'fit_baselines'), (experiment, 'train'),
            (experiment, 'run_baselines'), (runtime, 'train'), (comparison_tool, 'fit_inner_baselines'),
            (comparison_tool, 'run_comparison'), (network.CountGRU, 'forward'),
            (torch.Tensor, 'backward'), (torch.autograd, 'backward'), (torch.autograd, 'grad'),
            (torch.optim.Optimizer, 'step'), (torch.optim.AdamW, 'step')):
            stack.enter_context(patch.object(owner, name, guard()))
        yield


def describe(values):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Nonfinite/non-numeric descriptive input')
    return dict(n=len(values), minimum=min(values) if values else None, maximum=max(values) if values else None,
        mean=statistics.mean(values) if values else None, median=statistics.median(values) if values else None,
        population_sd=statistics.pstdev(values) if values else None)


def correlation(a, b):
    if len(a) != len(b): raise ValueError('Unpaired correlation')
    describe(a); describe(b)
    if len(a) < 2 or not statistics.pvariance(a) or not statistics.pvariance(b):
        return dict(n=len(a), r=None, reason='insufficient_n_or_zero_variance')
    return dict(n=len(a), r=statistics.correlation(a, b), reason=None)


def support_stats(row, cache):
    """Check saved all-eight-joint mask/time against saved statistics, no raw pose inference."""
    source = row['source']; n = source['frame_count']; fps = source['fps']
    mask, values = cache['complete'], cache['values']
    if (cache['source'] != source or cache['status'] != 'accepted' or cache['stats'] != row['input_stats']
            or len(mask) != n or len(values) != n or any(type(v) is not bool for v in mask)
            or cache['duration_seconds'] != n/fps):
        raise ValueError('Saved input linkage/duration/mask mismatch: '+source['video_id'])
    runs = []; current_valid = 0; current_gap = 0; longest_gap = 0
    for i, (valid, vector) in enumerate(zip(mask, values)):
        if len(vector) != 58 or not all(type(v) in (float, int) and math.isfinite(v) for v in vector):
            raise ValueError('Invalid saved input channels')
        if vector[-2] != i/fps or vector[-1] != (0.0 if i == 0 else 1/fps):
            raise ValueError('Saved timestamp/dt mismatch')
        usable = [vector[7*j+3] for j in range(8)]
        if valid != all(u == 1.0 for u in usable):
            raise ValueError('Complete mask differs from eight joint usability')
        for j in range(8):
            u, observed, low, missing = vector[7*j+3:7*j+7]
            if (any(v not in (0, 1) for v in (u, observed, low, missing))
                    or observed+low+missing != 1 or u > observed):
                raise ValueError('Invalid saved provenance channels')
        if valid:
            current_valid += 1; current_gap = 0
        else:
            if current_valid: runs.append(current_valid); current_valid = 0
            current_gap += 1; longest_gap = max(longest_gap, current_gap)
    if current_valid: runs.append(current_valid)
    total_complete = sum(mask)
    if (cache['stats']['frames'] != n or cache['stats']['complete_frames'] != total_complete
            or cache['stats']['complete_fraction'] != total_complete/n
            or cache['stats']['longest_incomplete_span'] != longest_gap):
        raise ValueError('Saved support statistics disagree with mask')
    if total_complete/n < .7 or longest_gap > 10:
        raise ValueError('Frozen admission violation; no silent exclusion')
    return dict(frames=n, complete_frames=total_complete, complete_fraction=total_complete/n,
        incomplete_frames=n-total_complete, longest_gap_frames=longest_gap, longest_gap_seconds=longest_gap/fps,
        valid_runs=len(runs), valid_run_lengths=runs, longest_valid_run=max(runs),
        raw_joint_observed=cache['stats']['provenance'].get('observed', 0),
        raw_joint_low_visibility=cache['stats']['provenance'].get('low_visibility', 0),
        raw_joint_missing=cache['stats']['provenance'].get('missing', 0))


def summarize(rows):
    n = len(rows); targets = [r['target'] for r in rows]
    models = {}
    for name in MODELS:
        predictions = [r[name] for r in rows]; signed = [p-t for p, t in zip(predictions, targets)]
        errors = [abs(x) for x in signed]
        models[name] = dict(predictions=describe(predictions), mae=statistics.mean(errors) if n else None,
            signed_error=statistics.mean(signed) if n else None,
            under=sum(e < 0 for e in signed), over=sum(e > 0 for e in signed), exact=sum(e == 0 for e in signed),
            rounded_histogram=dict(sorted(Counter(str(rounded_count(p)) for p in predictions).items())),
            prediction_target_correlation=correlation(predictions, targets),
            prediction_duration_correlation=correlation(predictions, [r['duration_seconds'] for r in rows]))
    deltas = [abs(r['gru']-r['target'])-abs(r['duration']-r['target']) for r in rows]
    return dict(n=n, groups=len({r['group_id'] for r in rows}), targets=describe(targets),
        target_histogram=dict(sorted(Counter(str(x) for x in targets).items())),
        duration=describe([r['duration_seconds'] for r in rows]),
        complete_fraction=describe([r['complete_fraction'] for r in rows]),
        longest_gap_frames=describe([r['longest_gap_frames'] for r in rows]), models=models,
        delta_gru_minus_duration=statistics.mean(deltas) if n else None,
        delta_sum=math.fsum(deltas), duration_wins=sum(d > 0 for d in deltas),
        gru_wins=sum(d < 0 for d in deltas), ties=sum(d == 0 for d in deltas))


def analyze(rows):
    if len(rows) != 33 or len({r['video_id'] for r in rows}) != 33:
        raise ValueError('Require all 33 unique paired videos')
    rows = [dict(r) for r in sorted(rows, key=lambda r: r['video_id'])]
    for row in rows:
        row['gru_minus_duration_AE'] = abs(row['gru']-row['target'])-abs(row['duration']-row['target'])
        row['pooled_MAE_gap_contribution'] = row['gru_minus_duration_AE']/33
        for model in MODELS:
            row[model+'_signed_error'] = row[model]-row['target']
            row[model+'_absolute_error'] = abs(row[model+'_signed_error'])
    grouped = {key: {str(v): summarize([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
               for key in ('fold', 'group_id', 'target')}
    bins = dict(duration={'[0,3)': lambda r: r['duration_seconds'] < 3,
                          '[3,5)': lambda r: 3 <= r['duration_seconds'] < 5,
                          '[5,infinity)': lambda r: r['duration_seconds'] >= 5},
        complete_fraction={'[0.7,0.9)': lambda r: .7 <= r['complete_fraction'] < .9,
                           '[0.9,1)': lambda r: .9 <= r['complete_fraction'] < 1,
                           'exact1': lambda r: r['complete_fraction'] == 1},
        longest_gap={'0': lambda r: r['longest_gap_frames'] == 0,
                     '1..5': lambda r: 1 <= r['longest_gap_frames'] <= 5,
                     '6..10': lambda r: 6 <= r['longest_gap_frames'] <= 10})
    for key, mapping in bins.items():
        grouped[key] = {name: summarize([r for r in rows if rule(r)]) for name, rule in mapping.items()}
        if sum(g['n'] for g in grouped[key].values()) != 33:
            raise ValueError('Descriptive bins lost videos: '+key)
    correlations = {x: {y: correlation([r[x] for r in rows], [r[y] for r in rows]) for y in
        ('target', 'gru_signed_error', 'gru_absolute_error', 'gru_minus_duration_AE')}
        for x in ('duration_seconds', 'complete_fraction', 'longest_gap_frames', 'valid_runs')}
    return dict(**STATUS, scope='fixed 33 outer-train admitted OOF clips; epoch index49',
        overall=summarize(rows), grouped=grouped, correlations=correlations,
        frames=sum(r['frames'] for r in rows), complete_frames=sum(r['complete_frames'] for r in rows),
        complete_frame_fraction_weighted=sum(r['complete_frames'] for r in rows)/sum(r['frames'] for r in rows),
        valid_runs=sum(r['valid_runs'] for r in rows), clips_with_gaps=sum(r['incomplete_frames'] > 0 for r in rows),
        contributor_order=[r['video_id'] for r in sorted(rows, key=lambda r: (-r['gru_minus_duration_AE'], r['video_id']))],
        records=rows, limitations='Descriptive association, not causal attribution or independent evaluation')


def verify_and_join(root, manifest_path, oof_path, comparison_dir):
    man, saved_oof, evidence = comparison_tool.verify_inputs(root, manifest_path, oof_path)
    folder = Path(comparison_dir).resolve()
    for name, expected in read(folder/'inventory.json')['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or file_info(path) != expected:
            raise ValueError('Comparison inventory mismatch: '+name)
    inputs = cv.unseal(read(folder/'inputs.json'))
    verification = cv.unseal(read(check_identity(root, inputs['verification'])))
    if verification != evidence or inputs['evidence'] != evidence:
        raise ValueError('Comparison input/verification linkage mismatch')
    baseline = cv.unseal(read(folder/'baselines.json'))
    saved = cv.unseal(read(folder/'comparison.json'))
    replay = comparison_tool.compare(man, comparison_tool.selected_gru(man, saved_oof), baseline)
    replay['inputs'] = evidence
    if canonical_hash(replay) != canonical_hash(saved):
        raise ValueError('Comparison differs from saved parameter/prediction replay')
    if (folder/'REPORT.md').read_text(encoding='utf-8') != comparison_tool.report_markdown(replay):
        raise ValueError('Human report differs from saved numerical result')
    receipt = read(folder/'receipt.json')['result']
    if receipt['status'] != 'COMPLETE' or receipt['metrics'] != saved['metrics']:
        raise ValueError('Comparison receipt mismatch')
    gru = {r['video_id']: r for r in saved_oof['curve'][49]['records']}
    constants = {r['video_id']: r for r in baseline['predictions']['constant']}
    durations = {r['video_id']: r for r in baseline['predictions']['duration']}
    rows = []
    for row in man['rows']:
        source = row['source']; v = source['video_id']
        cache = read(check_identity(root, row['input_artifact']))
        if cache['config_hash'] != canonical_hash(man['config']):
            raise ValueError('Saved input config mismatch')
        rows.append(dict(video_id=v, group_id=source['group_id'], fold=gru[v]['fold'], target=row['target'],
            duration_seconds=source['frame_count']/source['fps'], fps=source['fps'],
            gru=gru[v]['prediction'], constant=constants[v]['prediction'], duration=durations[v]['prediction'],
            **support_stats(row, cache), source_path=source['path'], source_sha256=source['sha256'],
            label_record_hash=row['label_record_hash'], input_artifact=row['input_artifact'],
            frozen_row_sha256=canonical_hash(row)))
    return rows, dict(input_verification=evidence, comparison_file=identity(root, folder/'comparison.json'),
        baseline_file=identity(root, folder/'baselines.json'), comparison_replay_exact=True,
        cached_inputs_checked=33, inference_runs=0, baseline_fits=0, training_steps=0)


def table(rows):
    lines = ['# Sai số từng video — đủ33, epoch50/index49', '',
        'AI weak reference; OOF đã chọn epoch, không độc lập. AE=absolute error; Δ=AE GRU−AE duration.', '',
        '|Video|Fold|gXX|AI count|s|Complete8|Gap max frames|GRU|Constant|Duration|Signed GRU|AE GRU|AE duration|Δ|Δ/33|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['video_id']}|{r['fold']}|{r['group_id']}|{r['target']}|{r['duration_seconds']:.2f}|"
            f"{r['complete_fraction']:.4f}|{r['longest_gap_frames']}|"+'|'.join(f'{r[k]:.6f}' for k in
                ('gru', 'constant', 'duration', 'gru_signed_error', 'gru_absolute_error', 'duration_absolute_error',
                 'gru_minus_duration_AE', 'pooled_MAE_gap_contribution'))+'|')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'gru-oof', 'comparison-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv); started = time.perf_counter()
    with no_learning_or_forward():
        rows, evidence = verify_and_join(args.root, args.manifest, args.gru_oof, args.comparison_dir)
        result = analyze(rows)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir/'verification.json', cv.sealed(evidence))
        write_json(args.output_dir/'analysis.json', cv.sealed(result))
        write_json(args.output_dir/'per-video-errors.json', cv.sealed(dict(records=result['records'], n=33)))
        with (args.output_dir/'PER_VIDEO_ERRORS.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(table(result['records']))
        write_json(args.output_dir/'receipt.json', dict(command=[sys.executable, *sys.argv],
            script=file_info(__file__), seconds=time.perf_counter()-started,
            rows=33, fit_calls=0, forward_calls=0, training_steps=0, gradients_or_optimizer_steps=0))
    print(json.dumps(dict(output_dir=str(args.output_dir), seconds=time.perf_counter()-started,
                         n=result['overall']['n'], models=result['overall']['models'],
                         delta=result['overall']['delta_gru_minus_duration']), ensure_ascii=False))


if __name__ == '__main__':
    main()
