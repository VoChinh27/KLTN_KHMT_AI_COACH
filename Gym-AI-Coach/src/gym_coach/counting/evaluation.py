"""Count/event evaluation only; never executes a detector or fills missing labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .models import (AnnotationBundle, EvaluationPolicy, Prediction, VideoIdentity,
                     STATUS, canonical_hash, overlap)


def checked(model, value):
    # Revalidate nested mutable values at every public intake.
    if isinstance(value, model):
        # JSON serialization can coerce a forged bool in an int slot to 1.
        # Validate native types before serialization, including nested models.
        return model.model_validate(value.model_dump(mode='python', warnings='error'), strict=True)
    else:
        value = json.dumps(value, allow_nan=False)
    return model.model_validate_json(value, strict=True)


def disagreement(bundle):
    bundle = checked(AnnotationBundle, bundle)
    a, b = bundle.pass_a, bundle.pass_b
    if a.status != 'completed' or b.status != 'completed':
        return dict(status='PENDING_TWO_PASSES', differences=None)
    fields = ('subject_description', 'count_status', 'count', 'boundaries_status',
              'cycles', 'partials', 'uncertain')
    da, db = a.model_dump(mode='json'), b.model_dump(mode='json')
    return dict(status='REVIEW_REQUIRED' if any(da[k] != db[k] for k in fields) else 'AGREEMENT_REQUIRES_LOG',
                differences={k: dict(pass_a=da[k], pass_b=db[k]) for k in fields if da[k] != db[k]},
                pass_a_sha256=canonical_hash(a), pass_b_sha256=canonical_hash(b))


def match_cycles(predictions, references, tolerance):
    """Order-preserving assignment; max matches, min summed boundary distance."""
    n, m = len(predictions), len(references)
    dp = [[(0, 0., ()) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            p, r = predictions[i].interval, references[j].interval
            errors = (abs(p.start.timestamp_seconds-r.start.timestamp_seconds),
                      abs(p.end.timestamp_seconds-r.end.timestamp_seconds))
            options = [dp[i+1][j], dp[i][j+1]]
            if max(errors) <= tolerance + 1e-9:
                number, cost, pairs = dp[i+1][j+1]
                options.append((number+1, cost+sum(errors), ((i, j),)+pairs))
            dp[i][j] = min(options, key=lambda state: (-state[0], state[1], state[2]))
    return dp[0][0][2]


def stats(values):
    return dict(n=len(values), signed_error=statistics.mean(values) if values else None,
                mae=statistics.mean(abs(x) for x in values) if values else None)


def ratio(n, d):
    return n/d if d else None


def evaluate(sources, bundles, predictions, *, protocol_sha256, data_kind, policy=None):
    if data_kind not in ('synthetic', 'real_development'):
        raise ValueError('Unknown dataset kind')
    sources = [checked(VideoIdentity, s) for s in sources]
    expected = {s.video_id: s for s in sources}
    if len(expected) != len(sources):
        raise ValueError('Duplicate source video ID')
    policy = checked(EvaluationPolicy, policy or EvaluationPolicy())
    labels, outputs = {}, {}
    for bundle in bundles:
        bundle = checked(AnnotationBundle, bundle)
        key = bundle.pass_a.source.video_id
        if (key in labels or key not in expected or bundle.pass_a.source != expected[key] or
                bundle.pass_a.protocol_sha256 != protocol_sha256 or bundle.data_kind != data_kind):
            raise ValueError('Label duplicate/source/protocol/data kind mismatch')
        labels[key] = bundle
    for prediction in predictions:
        prediction = checked(Prediction, prediction)
        key = prediction.source.video_id
        if key in outputs or key not in expected or prediction.source != expected[key]:
            raise ValueError('Prediction duplicate/source mismatch')
        outputs[key] = prediction
    errors, rows = [], []
    seconds = {k: [] for k in ('start', 'end', 'bottom')}
    frames = {k: [] for k in seconds}
    states, reasons = Counter(), Counter()
    labels_available = adjudicated = event_clips = tp = fp = fn = masked = bottom_unavailable = 0
    for source in sources:
        key = source.video_id
        bundle, prediction = labels.get(key), outputs.get(key)
        final = bundle.final if bundle else None
        state = prediction.status if prediction else 'not_run'
        states[state] += 1
        reasons.update(r.code for r in prediction.reasons) if prediction else reasons.update(['prediction_absent'])
        adjudicated += final is not None
        label_available = final is not None and final.count_status == 'available'
        labels_available += label_available
        error = prediction.count-final.count if label_available and state == 'estimated' else None
        if error is not None:
            errors.append(error)
        row = dict(video_id=key, source_sha256=source.sha256, group_id=source.group_id,
                   label_status=final.count_status if final else 'pending_adjudication',
                   reference_count=final.count if final else None, prediction_status=state,
                   predicted_count=prediction.count if prediction else None,
                   candidate_count=len(prediction.cycles) if state == 'candidates_only' else None,
                   signed_count_error=error, event_status='NOT_EVALUATED', matches=[],
                   excluded_prediction_indices=[], reasons=[r.model_dump(mode='json') for r in prediction.reasons] if prediction else [],
                   disagreement=disagreement(bundle) if bundle else None)
        if (label_available and final.boundaries_status == 'exhaustive' and
                state == 'estimated' and prediction.events_exhaustive):
            event_clips += 1
            masks = [p.interval for p in (*final.partials, *final.uncertain)]
            excluded = [i for i, c in enumerate(prediction.cycles) if any(overlap(c.interval, mask) for mask in masks)]
            indices = [i for i in range(len(prediction.cycles)) if i not in excluded]
            selected = [prediction.cycles[i] for i in indices]
            matches = match_cycles(selected, final.cycles, policy.boundary_tolerance_seconds)
            tp += len(matches); fp += len(selected)-len(matches); fn += len(final.cycles)-len(matches)
            masked += len(excluded)
            row.update(event_status='EVALUATED', excluded_prediction_indices=excluded,
                       false_positives=len(selected)-len(matches), false_negatives=len(final.cycles)-len(matches))
            for i, j in matches:
                p, r = selected[i], final.cycles[j]
                record = dict(prediction_index=indices[i], reference_index=j)
                for component in ('start', 'end', 'bottom'):
                    if component == 'bottom':
                        if p.bottom.status != 'event' or r.bottom.status != 'event':
                            record['bottom_error_seconds'] = None
                            bottom_unavailable += 1
                            continue
                        a, b = p.bottom.support.start, r.bottom.support.start
                    else:
                        a, b = getattr(p.interval, component), getattr(r.interval, component)
                    delta = a.timestamp_seconds-b.timestamp_seconds
                    seconds[component].append(delta)
                    frames[component].append(a.frame_index-b.frame_index)
                    record[component+'_error_seconds'] = delta
                    record[component+'_error_frames'] = a.frame_index-b.frame_index
                row['matches'].append(record)
        rows.append(row)
    total = len(sources)
    return dict(version='squat-count-evaluation-v1', **STATUS, data_kind=data_kind,
        evaluation_status='EVALUATED' if errors or event_clips else 'NOT_EVALUATED',
        accuracy_scope='synthetic software checks' if data_kind == 'synthetic' else 'selected development sources only',
        policy=policy.model_dump(mode='json'), policy_sha256=canonical_hash(policy), protocol_sha256=protocol_sha256,
        denominator=dict(video_paths=total, unique_byte_hashes=len({s.sha256 for s in sources}),
            adjudicated_labels=adjudicated, labels_with_count=labels_available, comparable_count=len(errors),
            event_evaluated_clips=event_clips),
        coverage=dict(prediction=ratio(states['estimated'], total), label=ratio(labels_available, total),
            comparable=ratio(len(errors), total), abstain=ratio(states['abstained'], total),
            failure=ratio(states['failed'], total), not_run=ratio(states['not_run'], total),
            candidates_only=ratio(states['candidates_only'], total)),
        prediction_states=dict(states), prediction_reasons=dict(reasons),
        count_metrics=dict(**stats(errors), exact_count_match=ratio(sum(x == 0 for x in errors), len(errors))),
        event_metrics=dict(true_positive=tp if event_clips else None, false_positive=fp if event_clips else None,
            false_negative=fn if event_clips else None, precision=ratio(tp, tp+fp) if event_clips else None,
            recall=ratio(tp, tp+fn) if event_clips else None, masked_predictions=masked,
            matched_without_point_bottom=bottom_unavailable,
            timing_seconds={k: stats(v) for k, v in seconds.items()},
            timing_frames={k: stats(v) for k, v in frames.items()}),
        rows=rows, denominator_note='Every manifest path retained; duplicate paths counted explicitly, group split isolates byte copies. No abstention converted to zero.')


def main(argv=None):
    from .readiness import file_info, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_bytes())
    annotations = json.loads(args.annotations.read_bytes()); predictions = json.loads(args.predictions.read_bytes())
    expected_hash = file_info(args.manifest)['sha256']
    if (manifest['version'] != 'squat-count-dataset-v1' or
            annotations['version'] != 'squat-count-adjudication-collection-v1' or
            predictions['version'] != 'squat-count-prediction-collection-v1' or
            any(d['dataset_manifest_sha256'] != expected_hash for d in (annotations, predictions))):
        raise ValueError('Collection/manifest version or byte hash mismatch')
    result = evaluate([r['source'] for r in manifest['videos']], annotations['bundles'], predictions['predictions'],
                      data_kind=manifest['data_kind'], protocol_sha256=manifest['protocol_sha256'],
                      policy=manifest['evaluation_policy'])
    out = args.output_dir.resolve()
    if not out.is_relative_to(Path.cwd().resolve()/'outputs'):
        raise ValueError('Output must be a new directory under workspace outputs')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'evaluation.json', result)
    write_json(out/'receipt.json', dict(inputs={str(p): file_info(p) for p in (args.manifest, args.annotations, args.predictions)},
                                       output=file_info(out/'evaluation.json')))
    print(json.dumps(result['denominator']))


if __name__ == '__main__':
    main()
