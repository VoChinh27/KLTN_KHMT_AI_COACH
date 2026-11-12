"""Fixed endpoint comparison against saved controls. No estimation of parameters."""
import math
import statistics

from .. import cv_baseline_comparison as pairing
from ..models import STATUS, canonical_hash
from .manifest import ENDPOINT, VERSION, fold_identity

METHODS = ('ablation', 'gru', 'constant', 'duration')


def describe(values):
    return dict(n=len(values), minimum=min(values), maximum=max(values),
        mean=statistics.mean(values), population_sd=statistics.pstdev(values))


def metrics(manifest, records):
    records = pairing.paired_records(manifest, records)
    result = pairing.metrics(manifest, records)
    result['counts'] = {str(t): dict(clips=len(items),
        continuous_MAE=math.fsum(abs(r['prediction']-r['target']) for r in items)/len(items),
        predictions=describe([r['prediction'] for r in items]))
        for t in sorted({r['target'] for r in records})
        for items in [[r for r in records if r['target'] == t]]}
    result['predictions'] = describe([r['prediction'] for r in records])
    for key, field in (('folds', 'fold'), ('groups', 'group_id')):
        for value in result[key]:
            items = [r for r in records if str(r[field]) == value]
            result[key][value]['predictions'] = describe([r['prediction'] for r in items])
    return result


def compare_reports(manifest, reports, controls):
    if len(reports) != 3 or sorted(r['fold'] for r in reports) != [0, 1, 2]:
        raise ValueError('Exactly one endpoint report per fold required')
    all_records = []
    for report in reports:
        fold = report['fold']
        identity = fold_identity(manifest, fold)
        if (type(report['epoch']) is not int or report['epoch'] != ENDPOINT['epoch_index']
                or report['manifest_sha256'] != canonical_hash(manifest)
                or report['identity_sha256'] != canonical_hash(identity)):
            raise ValueError('Fixed endpoint/report identity mismatch')
        ids = [r['video_id'] for r in report['records']]
        if len(ids) != len(set(ids)) or set(ids) != set(identity['partition']['evaluation_ids']):
            raise ValueError('Per-fold endpoint coverage mismatch')
        for r in report['records']:
            if 'fold' in r and r['fold'] != fold:
                raise ValueError('Record fold mismatch')
            all_records.append(dict(r, fold=fold))
    if set(controls) != {'gru', 'constant', 'duration'}:
        raise ValueError('Require all three saved controls')
    predictions = dict(ablation=pairing.paired_records(manifest, all_records),
        **{k: pairing.paired_records(manifest, v) for k, v in controls.items()})
    rows = []
    for i, a in enumerate(predictions['ablation']):
        row = {k: a[k] for k in ('video_id', 'source_sha256', 'group_id', 'component_id',
                                  'target', 'label_record_hash', 'row_sha256', 'fold')}
        row['predictions'] = {m: predictions[m][i]['prediction'] for m in METHODS}
        row['signed_errors'] = {m: row['predictions'][m]-a['target'] for m in METHODS}
        row['absolute_errors'] = {m: abs(row['signed_errors'][m]) for m in METHODS}
        row['delta_AE_ablation_minus_gru'] = row['absolute_errors']['ablation']-row['absolute_errors']['gru']
        rows.append(row)
    return dict(version=VERSION, **STATUS, status='COMPLETE', endpoint=ENDPOINT,
        manifest_sha256=canonical_hash(manifest), clips=33, groups=12,
        metrics={m: metrics(manifest, predictions[m]) for m in METHODS}, records=rows,
        prediction_hashes={m: canonical_hash(predictions[m]) for m in METHODS},
        delta_definition='AE_ablation - AE_original_GRU; positive means original GRU has smaller error',
        delta_mean=math.fsum(r['delta_AE_ablation_minus_gru'] for r in rows)/33,
        delta_by_fold={str(f): statistics.mean(r['delta_AE_ablation_minus_gru'] for r in rows if r['fold'] == f)
                       for f in range(3)},
        baseline_fits=0, outer_validation_used=False, development_test_used=False, final_refit=False,
        limitation='AI weak reference; adaptive development OOF already viewed/used in design, not independent final evaluation. '
                    'One seed, no significance or equivalence claim. Visibility/time/support remain.')


def report_markdown(result):
    lines = ['# XY ablation — fixed epoch50/index49', '', result['limitation'], '',
        'NOT_VALIDATED; official_eligibility=false; assessment=false; G5.7 blocked; G6 IN_PROGRESS.', '',
        '33 clips; macro-MAE gives equal weight to each of 12 gXX groups. SD is population SD.', '',
        '|Method|n clips|Pooled MAE|Macro 12-group MAE|Prediction min|max|SD|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        m = result['metrics'][name]; d = m['predictions']
        lines.append(f"|{name}|33|{m['pooled']['continuous_MAE']:.9f}|{m['macro_group_MAE']:.9f}|"
                     f"{d['minimum']:.6f}|{d['maximum']:.6f}|{d['population_sd']:.6f}|")
    for dimension in ('folds', 'groups', 'counts'):
        lines.extend(['', '## '+dimension, '', '|Value|n|Ablation MAE|GRU MAE|Constant MAE|Duration MAE|',
                      '|---|---:|---:|---:|---:|---:|'])
        for key, item in result['metrics']['ablation'][dimension].items():
            vals = '|'.join(f"{result['metrics'][m][dimension][key]['continuous_MAE']:.9f}" for m in METHODS)
            lines.append(f"|{key}|{item['clips']}|{vals}|")
    lines.extend(['', '## Paired video errors', '', result['delta_definition'], '',
        '|Video|Fold|Group|AI count|Ablation|GRU|Constant|Duration|AE ablation|AE GRU|Delta|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in result['records']:
        vals = '|'.join(f"{r['predictions'][m]:.6f}" for m in METHODS)
        lines.append(f"|{r['video_id']}|{r['fold']}|{r['group_id']}|{r['target']}|{vals}|"
                     f"{r['absolute_errors']['ablation']:.6f}|{r['absolute_errors']['gru']:.6f}|"
                     f"{r['delta_AE_ablation_minus_gru']:.6f}|")
    return '\n'.join(lines)+'\n'
