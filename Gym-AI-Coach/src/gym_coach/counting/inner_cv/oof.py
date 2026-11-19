"""Pure evaluator. No model, fitting, checkpoint selection across different epochs or outer data."""
from collections import Counter, defaultdict
import math
import statistics

from ..models import canonical_hash, STATUS
from .manifest import VERSION, SELECTION


def pair_identity(row):
    return dict(video_id=row['source']['video_id'], source_sha256=row['source']['sha256'],
                group_id=row['source']['group_id'], component_id=row['component_id'],
                target=row['target'], label_record_hash=row['label_record_hash'],
                row_sha256=canonical_hash(row))


def valid_prediction(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def metric(records):
    errors = [abs(r['prediction']-r['target']) for r in records if r['status'] == 'estimated']
    return dict(expected=len(records), comparable=len(errors),
                continuous_MAE=math.fsum(errors)/len(records) if len(errors) == len(records) and records else None)


def evaluate_epoch(manifest, epoch, reports):
    expected = {r['source']['video_id']: pair_identity(r) for r in manifest['rows']}
    fold_for = {v: f['fold'] for f in manifest['folds'] for v in f['evaluation_ids']}
    candidates = defaultdict(list)
    issues = []
    fold_reports = Counter()
    for report in reports:
        fold = report.get('fold')
        if (type(fold) is not int or fold not in range(3) or report.get('epoch') != epoch
                or report.get('manifest_sha256') != canonical_hash(manifest)):
            issues.append(dict(code='report_identity_mismatch', fold=fold))
            continue
        fold_reports[fold] += 1
        for value in report.get('records', []):
            if not isinstance(value, dict):
                issues.append(dict(code='malformed_prediction_record', fold=fold))
                continue
            video_id = value.get('video_id')
            if video_id not in expected or fold_for[video_id] != fold:
                issues.append(dict(code='unexpected_oof_pair', fold=fold, video_id=video_id))
                continue
            candidates[video_id].append(value)
    for fold in range(3):
        if fold_reports[fold] != 1:
            issues.append(dict(code='missing_fold_report' if not fold_reports[fold] else 'duplicate_fold_report', fold=fold))
    records = []
    for video_id, pair in expected.items():
        values = candidates[video_id]
        result = dict(**pair, fold=fold_for[video_id], prediction=None, status='unavailable', reason='missing_prediction')
        if len(values) > 1:
            result.update(reason='duplicate_prediction')
        elif values:
            value = values[0]
            if canonical_hash({k: value.get(k) for k in pair}) != canonical_hash(pair):
                result.update(reason='oof_pair_linkage_mismatch')
            elif value.get('status') == 'estimated' and valid_prediction(value.get('prediction')) and value.get('reason') is None:
                result.update(prediction=value['prediction'], status='estimated', reason=None)
            elif (value.get('status') in ('failed', 'unavailable') and value.get('prediction') is None
                  and isinstance(value.get('reason'), str) and value['reason']):
                result.update(status=value['status'], reason=value['reason'])
            else:
                result.update(reason='invalid_prediction_contract')
        if result['status'] != 'estimated':
            issues.append(dict(code=result['reason'], video_id=video_id, fold=fold_for[video_id]))
        records.append(result)
    pooled = metric(records)
    complete = not issues and pooled['comparable'] == 33
    # Extra/duplicate reports also invalidate a numerically complete curve.
    if not complete:
        pooled['continuous_MAE'] = None
    groups = {g: metric([r for r in records if r['group_id'] == g]) for g in sorted({r['group_id'] for r in records})}
    folds = {str(f): metric([r for r in records if r['fold'] == f]) for f in range(3)}
    for f in range(3):
        if fold_reports[f] != 1 or any(i.get('fold') == f and 'video_id' not in i for i in issues):
            folds[str(f)]['continuous_MAE'] = None
    return dict(epoch=epoch, complete=complete, pooled=pooled, folds=folds, groups=groups,
        macro_group_MAE=math.fsum(m['continuous_MAE'] for m in groups.values())/12 if complete else None,
        issues=issues, records=records)


def aggregate(manifest, reports):
    by_epoch = defaultdict(list)
    global_issues = []
    for report in reports:
        epoch = report.get('epoch')
        if type(epoch) is not int or epoch not in range(50):
            global_issues.append(dict(code='invalid_epoch', epoch=epoch))
        else:
            by_epoch[epoch].append(report)
    curve = [evaluate_epoch(manifest, e, by_epoch[e]) for e in range(50)]
    complete = not global_issues and all(e['complete'] for e in curve)
    selection = None
    if complete:
        # min returns the earliest equal minimum in this ascending epoch list.
        best = min(curve, key=lambda e: e['pooled']['continuous_MAE'])
        fold_best = {str(f): min(curve, key=lambda e: e['folds'][str(f)]['continuous_MAE'])['epoch'] for f in range(3)}
        epochs = list(fold_best.values())
        selection = dict(epoch=best['epoch'], epoch_one_based=best['epoch']+1,
            pooled_oof_MAE=best['pooled']['continuous_MAE'], index5_MAE=curve[5]['pooled']['continuous_MAE'],
            index5_minus_minimum=curve[5]['pooled']['continuous_MAE']-best['pooled']['continuous_MAE'],
            fold_best_epochs=fold_best, fold_best_range=[min(epochs), max(epochs)],
            fold_best_mean=statistics.mean(epochs), fold_best_population_sd=statistics.pstdev(epochs))
    return dict(version=VERSION, **STATUS, manifest_sha256=canonical_hash(manifest),
        status='COMPLETE' if complete else 'INCOMPLETE_OR_INVALID', selection_rule=SELECTION,
        selection=selection, expected_epochs=50, expected_clips_per_epoch=33, expected_groups=12,
        complete_epochs=sum(e['complete'] for e in curve), global_issues=global_issues, curve=curve,
        outer_validation_used=False, development_test_used=False, final_refit=False,
        limitation='AI weak references; selected OOF minimum is not independent evaluation; development-test already exposed')
