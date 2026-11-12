"""Reproducible synthetic-only audit of the unchanged squat temporal profile.

Run: python -m gym_coach.reporting.temporal_sensitivity --output NEW_DIRECTORY
No input video/metadata/profile path is accepted. No model or threshold fitting.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from ..segmentation import SquatSegmentationProfile, SquatSegmentationResult, segment_squat
from ..segmentation.models import content_hash, strict_json
from ..segmentation.synthetic_fixtures import SampledFixture, build_fixture, canonical_cases
from .squat_builder import phase_at

EXPECTED_PROFILE_HASH = '5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017'
PROTOCOL = {
    'evaluation_version': 'squat-temporal-sensitivity-v1',
    'scope': 'analytic_synthetic_signal_only',
    'fps': [25., 30., 60.],
    'profile_hash': EXPECTED_PROFILE_HASH,
    'matching': 'monotone_one_to_one_max_cardinality_then_max_total_interval_iou',
    'min_interval_iou': .5,
    'matching_epsilon': 1e-12,
    'bottom_must_be_inside_reference_cycle': True,
    'event_error': 'prediction_minus_analytic_reference; frames use nearest_CFR_ties_up',
    'pause_reference': 'entire arrival..departure is Bottom; midpoint is a separate event proxy',
    'pause_boundary_error_interpretation': 'single predicted bottom event relative to analytic hold bounds; algorithm does not predict two pause boundaries',
    'shared_top_phase': 'Return/Top of earlier cycle',
    'phase_denominator': 'union of full reference cycle frames, including unresolved gaps and missed cycles',
    'boundary_partial_references': 'reported separately, excluded from full-cycle matching and phase denominator',
    'unmatched_policy': 'misses/extras retained; timing errors conditional on matched cycles only',
    'validation_status': 'NOT_VALIDATED',
    'official_eligibility': False,
    'assessment_enabled': False,
    'official_gate': 'BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _iou(reference, candidate):
    intersection = max(0., min(reference.end_seconds, candidate.end_seconds) -
                       max(reference.start_seconds, candidate.start_seconds))
    union = max(reference.end_seconds, candidate.end_seconds) - min(reference.start_seconds, candidate.start_seconds)
    return intersection / union if union else 0.


def match_cycles(references, candidates):
    """Order-preserving 1:1 matching. No event-timing tolerance hides bad phases."""
    @lru_cache(None)
    def solve(i, j):
        if i == len(references) or j == len(candidates):
            return 0, 0., ()
        choices = [solve(i + 1, j), solve(i, j + 1)]
        ref, pred = references[i], candidates[j]
        overlap = _iou(ref, pred)
        if (overlap + PROTOCOL['matching_epsilon'] >= PROTOCOL['min_interval_iou']
                and ref.start_seconds <= pred.bottom_seconds <= ref.end_seconds):
            n, total, pairs = solve(i + 1, j + 1)
            choices.append((n + 1, total + overlap, ((i, j), *pairs)))
        return max(choices, key=lambda item: (item[0], item[1]))
    return solve(0, 0)[2]


def _safe_reference(reference, samples):
    selected = samples[reference.start_frame:reference.end_frame + 1]
    reasons = []
    if any(s.value is None for s in selected):
        reasons.append('reference_crosses_unresolved_gap')
    if len({s.side_source for s in selected}) > 1:
        reasons.append('reference_crosses_explicit_side_switch')
    return not reasons, reasons


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _errors(values, unit='seconds'):
    return {'n': len(values), f'signed_bias_{unit}': _ratio(sum(values), len(values)),
            f'mae_{unit}': _ratio(sum(abs(v) for v in values), len(values)),
            f'max_abs_{unit}': max((abs(v) for v in values), default=None)}


def evaluate_fixture(fixture: SampledFixture, result: SquatSegmentationResult) -> dict:
    """Compare independently generated references against frozen output only."""
    if fixture != build_fixture(fixture.definition, fixture.signal.fps):
        raise ValueError('Fixture/reference modified after analytic construction')
    result = SquatSegmentationResult.model_validate_json(strict_json(result), strict=True)
    if result.profile_hash != EXPECTED_PROFILE_HASH or result.input != fixture.signal:
        raise ValueError('Audit requires the unchanged profile and exact synthetic input')
    references = tuple(r for r in fixture.references if not r.boundary_partial)
    candidates = result.repetitions
    pairs = match_cycles(references, candidates)
    matched_refs = {i for i, _ in pairs}
    matched_predictions = {j for _, j in pairs}
    matches = []
    for i, j in pairs:
        ref, pred = references[i], candidates[j]
        events = {}
        for key, seconds, frame, actual_seconds, actual_frame in (
            ('top_start', ref.start_seconds, ref.start_frame, pred.start_seconds, pred.start_frame),
            ('bottom_midpoint_proxy', ref.bottom_midpoint_seconds, ref.bottom_midpoint_frame, pred.bottom_seconds, pred.bottom_frame),
            ('top_end', ref.end_seconds, ref.end_frame, pred.end_seconds, pred.end_frame),
        ):
            events[key] = {'signed_seconds': actual_seconds - seconds, 'signed_frames': actual_frame - frame}
        matches.append({'reference_index': ref.reference_index, 'candidate_index': j + 1,
            'interval_iou': _iou(ref, pred), 'event_errors': events,
            'bottom_arrival_error_seconds': pred.bottom_seconds - ref.bottom_arrival_seconds,
            'bottom_departure_error_seconds': pred.bottom_seconds - ref.bottom_departure_seconds,
            'descent_duration_error_seconds': pred.descent_seconds - (ref.bottom_arrival_seconds - ref.start_seconds),
            'ascent_duration_error_seconds': pred.ascent_seconds - (ref.end_seconds - ref.bottom_departure_seconds)})

    missed = []
    safety = [_safe_reference(ref, fixture.signal.samples) for ref in references]
    for i, ref in enumerate(references):
        if i not in matched_refs:
            safe, reasons = safety[i]
            missed.append({'reference_index': ref.reference_index, 'source_safe': safe,
                'reasons': reasons or ['no_matching_candidate_under_frozen_profile'],
                'segmentation_flag_codes': sorted({f.code.value for f in result.flags})})

    predicted_phases = tuple(phase_at(result, sample.frame_index, fixture.signal.fps)[0]
                             for sample in fixture.signal.samples)
    confusion = Counter()
    support, correct, available = Counter(), Counter(), Counter()
    for truth, pred, included in zip(fixture.reference_phases, predicted_phases, fixture.full_reference_mask):
        if not included:
            continue
        support[truth] += 1
        available[truth] += pred != 'Unavailable'
        correct[truth] += truth == pred
        confusion[truth, pred] += 1
    per_phase = {label: {'reference_frames': n, 'available_frames': available[label],
                        'correct_frames': correct[label], 'availability_coverage': available[label] / n,
                        'recall': correct[label] / n} for label, n in sorted(support.items())}
    total = sum(support.values())
    available_total, correct_total = sum(available.values()), sum(correct.values())
    safe_count = sum(safe for safe, _ in safety)
    return {
        'evaluation_version': PROTOCOL['evaluation_version'], 'protocol_hash': json_hash(PROTOCOL),
        'case_id': fixture.definition.case_id, 'fps': fixture.signal.fps,
        'fixture_definition_hash': content_hash(fixture.definition), 'fixture_payload_hash': json_hash(fixture.payload()),
        'input_hash': fixture.signal.content_hash(), 'segmentation_hash': result.content_hash(),
        'profile_hash': result.profile_hash, 'validation_status': 'NOT_VALIDATED',
        'official_eligibility': False, 'assessment_enabled': False,
        'official_gate': PROTOCOL['official_gate'], 'execution_status': 'executed',
        'reference_cycle_count': len(references), 'all_reference_cycles': len(fixture.references),
        'boundary_partial_references': [r.reference_index for r in fixture.references if r.boundary_partial],
        'source_safe_full_references': safe_count, 'source_invalid_full_references': len(references) - safe_count,
        'candidate_count': len(candidates), 'matched_count': len(pairs),
        'cycle_detection_coverage': _ratio(len(pairs), len(references)),
        'source_safe_cycle_coverage': _ratio(sum(safety[i][0] for i, _ in pairs), safe_count),
        'signed_count_error': len(candidates) - len(references),
        'matches': matches, 'missed_references': missed,
        'unexpected_candidates': [j + 1 for j in range(len(candidates)) if j not in matched_predictions],
        'timing_errors': {key: _errors([m['event_errors'][key]['signed_seconds'] for m in matches])
                          for key in ('top_start', 'bottom_midpoint_proxy', 'top_end')},
        'timing_frame_errors': {key: _errors([m['event_errors'][key]['signed_frames'] for m in matches], 'frames')
                                for key in ('top_start', 'bottom_midpoint_proxy', 'top_end')},
        'phase': {'reference_frames': total, 'available_frames': available_total, 'correct_frames': correct_total,
                  'availability_coverage': _ratio(available_total, total),
                  'accuracy_all_reference_frames': _ratio(correct_total, total),
                  'accuracy_available_frames_only': _ratio(correct_total, available_total),
                  'per_phase': per_phase,
                  'confusion': [{'reference': a, 'prediction': b, 'frames': n} for (a, b), n in sorted(confusion.items())]},
        'predicted_phases': list(predicted_phases),
        'finite_input_coverage': _ratio(sum(s.value is not None for s in fixture.signal.samples), len(fixture.signal.samples)),
    }


def summarize(rows):
    executed = [r for r in rows if r['execution_status'] == 'executed']
    full = sum(r['reference_cycle_count'] for r in executed)
    matched = sum(r['matched_count'] for r in executed)
    phase_n = sum(r['phase']['reference_frames'] for r in executed)
    phase_available = sum(r['phase']['available_frames'] for r in executed)
    phase_correct = sum(r['phase']['correct_frames'] for r in executed)
    return {'evaluation_version': PROTOCOL['evaluation_version'], 'protocol_hash': json_hash(PROTOCOL),
        'validation_status': 'NOT_VALIDATED', 'official_eligibility': False, 'assessment_enabled': False,
        'official_gate': PROTOCOL['official_gate'], 'profile_hash': EXPECTED_PROFILE_HASH,
        'conditions': len(rows), 'execution_successes': len(executed), 'execution_failures': len(rows) - len(executed),
        'reference_cycles_in_executed_conditions': full,
        'matched_cycles': matched, 'cycle_detection_coverage': _ratio(matched, full),
        'missed_cycles': sum(len(r['missed_references']) for r in executed),
        'extra_candidates': sum(len(r['unexpected_candidates']) for r in executed),
        'boundary_partial_cycles': sum(len(r['boundary_partial_references']) for r in executed),
        'phase_reference_frames': phase_n, 'phase_available_frames': phase_available,
        'phase_correct_frames': phase_correct, 'phase_availability_coverage': _ratio(phase_available, phase_n),
        'phase_accuracy_all_reference_frames': _ratio(phase_correct, phase_n),
        'timing_errors_matched_only': {key: _errors([m['event_errors'][key]['signed_seconds']
            for r in executed for m in r['matches']]) for key in ('top_start', 'bottom_midpoint_proxy', 'top_end')},
        'aggregation_note': 'Fixed diagnostic matrix; FPS variants are repeated signals, not independent athlete samples',
    }


def _write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _byte_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_audit(output: Path, *, command=()):
    profile = SquatSegmentationProfile()
    if profile.content_hash() != EXPECTED_PROFILE_HASH:
        raise ValueError('Profile drift: audit refuses to tune or accept modified S3 settings')
    output.mkdir(parents=True, exist_ok=False)
    definitions = canonical_cases()
    _write_json(output / 'protocol.json', PROTOCOL)
    _write_json(output / 'fixture_definitions.json', [d.model_dump(mode='json') for d in definitions])
    _write_json(output / 'profile.json', profile.model_dump(mode='json'))
    rows = []
    for definition in definitions:
        for fps in PROTOCOL['fps']:
            fixture = build_fixture(definition, fps)
            directory = output / f'{definition.case_id}-{fps:g}fps'
            directory.mkdir()
            # Persist analytic ground truth BEFORE observing segmentation output.
            _write_json(directory / 'fixture.json', fixture.payload())
            try:
                result = segment_squat(fixture.signal)
                row = evaluate_fixture(fixture, result)
                _write_json(directory / 'segmentation.json', result.model_dump(mode='json'))
            except (ValueError, TypeError, RuntimeError) as exc:
                # Never treat a failed execution as a successful zero-count case.
                row = {'case_id': definition.case_id, 'fps': fps, 'execution_status': 'failed',
                       'error_type': type(exc).__name__, 'error': str(exc),
                       'profile_hash': EXPECTED_PROFILE_HASH, 'official_eligibility': False,
                       'assessment_enabled': False, 'validation_status': 'NOT_VALIDATED'}
            _write_json(directory / 'evaluation.json', row)
            rows.append(row)
    summary = summarize(rows)
    _write_json(output / 'summary.json', summary)
    _write_json(output / 'results.json', rows)
    lines = ['# Synthetic sensitivity — NOT_VALIDATED', '',
             'Official eligibility=false; assessment=false; G5.7 blocked. No real data.', '',
             '| Case | FPS | Matched/full refs | Candidates | Phase coverage | Phase agreement (all ref frames) | Bottom proxy MAE (s, matched only) | Missed ref IDs |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['execution_status'] != 'executed':
            lines.append(f"|{r['case_id']}|{r['fps']:g}|EXECUTION FAILED|||||{r['error_type']}|")
            continue
        lines.append(f"|{r['case_id']}|{r['fps']:g}|{r['matched_count']}/{r['reference_cycle_count']}|"
                     f"{r['candidate_count']}|{r['phase']['availability_coverage']:.4f}|"
                     f"{r['phase']['accuracy_all_reference_frames']:.4f}|"
                     f"{r['timing_errors']['bottom_midpoint_proxy']['mae_seconds']}|"
                     f"{','.join(str(v['reference_index']) for v in r['missed_references']) or 'none'}|")
    with (output / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    package = Path(__file__).parents[1]
    _write_json(output / 'manifest.json', {
        'artifact_version': 'squat-temporal-sensitivity-artifacts-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': list(command),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        **{name: importlib.metadata.version(name) for name in ('ai-coach-gym', 'pydantic', 'pytest')}},
        'protocol_hash': json_hash(PROTOCOL), 'profile_hash': profile.content_hash(),
        'source_files': {p.relative_to(package).as_posix(): _byte_hash(p) for p in (
            package / 'segmentation/models.py', package / 'segmentation/squat.py',
            package / 'segmentation/synthetic_fixtures.py', package / 'reporting/squat_builder.py', Path(__file__))},
        'files': {p.relative_to(output).as_posix(): _byte_hash(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'scope': 'synthetic_only_no_dataset_reads_or_inference',
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing output refused')
    args = parser.parse_args(argv)
    summary = run_audit(args.output, command=[sys.executable, '-m', __spec__.name,
        *(argv if argv is not None else sys.argv[1:])])
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0 if summary['execution_failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
