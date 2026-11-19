"""Analytic references, honest missed-case accounting and frozen-profile audit."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.segmentation import SquatSegmentationProfile, segment_squat
from gym_coach.segmentation.synthetic_fixtures import (
    CycleSpec, MissingInterval, SyntheticCase, build_fixture, canonical_cases,
)
from gym_coach.reporting.temporal_sensitivity import (
    EXPECTED_PROFILE_HASH, PROTOCOL, evaluate_fixture, json_hash, match_cycles,
    run_audit, summarize,
)

CASES = canonical_cases()


@pytest.mark.parametrize('fps', [25., 30., 60.])
@pytest.mark.parametrize('definition', CASES, ids=lambda case: case.case_id)
def test_fixed_matrix_executes_without_changing_profile_or_input(definition, fps):
    fixture = build_fixture(definition, fps)
    before = fixture.signal.model_dump_json()
    result = segment_squat(fixture.signal)
    row = evaluate_fixture(fixture, result)
    assert fixture.signal.model_dump_json() == before
    assert result.profile == SquatSegmentationProfile()
    assert result.profile_hash == EXPECTED_PROFILE_HASH
    assert result.profile.validation_status == row['validation_status'] == 'NOT_VALIDATED'
    assert result.official_eligibility is row['official_eligibility'] is False
    assert result.assessment_enabled is row['assessment_enabled'] is False
    assert row['matched_count'] + len(row['missed_references']) == row['reference_cycle_count']
    assert row['matched_count'] + len(row['unexpected_candidates']) == row['candidate_count']
    assert row['timing_errors']['bottom_midpoint_proxy']['n'] == row['matched_count']
    assert 0 <= row['phase']['availability_coverage'] <= 1
    assert 0 <= row['phase']['accuracy_all_reference_frames'] <= row['phase']['availability_coverage']
    for rep in result.repetitions:
        selected = fixture.signal.samples[rep.start_frame:rep.end_frame + 1]
        assert all(s.value is not None for s in selected)
        assert len({s.side_source for s in selected}) == 1
    assert all(v is None for s, v in zip(fixture.signal.samples, result.smoothed_values) if s.value is None)
    assert fixture.payload() == json.loads(json.dumps(fixture.payload(), allow_nan=False))


def test_reference_generator_is_independent_of_segmentation_and_reporting(monkeypatch):
    import gym_coach.segmentation.squat as algorithm
    import gym_coach.reporting.squat_builder as reporting
    def forbidden(*a, **k):
        raise AssertionError('Ground truth must not query predictions')
    monkeypatch.setattr(algorithm, 'segment_squat', forbidden)
    monkeypatch.setattr(reporting, 'phase_at', forbidden)
    definition = SyntheticCase(case_id='known_pause', description='Hand specified analytic truth',
        cycles=(CycleSpec(descent_seconds=1., bottom_pause_seconds=.6, ascent_seconds=1.),))
    fixture = build_fixture(definition, 25.)
    ref, = fixture.references
    assert (ref.start_seconds, ref.bottom_arrival_seconds, ref.bottom_departure_seconds, ref.end_seconds) == (1., 2., 2.6, 3.6)
    assert (ref.start_frame, ref.bottom_midpoint_frame, ref.end_frame) == (25, 58, 90)
    assert fixture.signal.samples[25].value == 170.
    assert fixture.signal.samples[55].value == 110.
    assert fixture.reference_phases[25] == 'Start/Top'
    assert fixture.reference_phases[40] == 'Descent'
    assert all(fixture.reference_phases[i] == 'Bottom' for i in range(50, 66))
    assert fixture.reference_phases[80] == 'Ascent'
    assert fixture.reference_phases[90] == 'Return/Top'
    assert sum(fixture.full_reference_mask) == 66


def test_source_definition_hash_and_sampling_are_reproducible():
    definition = CASES[1]
    a, b = build_fixture(definition, 30.), build_fixture(definition, 30.)
    other_fps = build_fixture(definition, 60.)
    assert a == b and json_hash(a.payload()) == json_hash(b.payload())
    assert a.signal.source_hash == other_fps.signal.source_hash
    assert a.signal.content_hash() != other_fps.signal.content_hash()


@pytest.mark.parametrize('fps', [25., 30., 60.])
def test_unseen_cycles_remain_misses_not_timing_successes(fps):
    case = next(c for c in CASES if c.case_id == 'small_excursion')
    fixture = build_fixture(case, fps)
    row = evaluate_fixture(fixture, segment_squat(fixture.signal))
    assert row['reference_cycle_count'] == 3 and row['candidate_count'] == row['matched_count'] == 0
    assert len(row['missed_references']) == 3 and row['cycle_detection_coverage'] == 0
    assert row['timing_errors']['bottom_midpoint_proxy']['mae_seconds'] is None
    assert row['phase']['availability_coverage'] == row['phase']['accuracy_all_reference_frames'] == 0
    assert row['phase']['accuracy_available_frames_only'] is None


@pytest.mark.parametrize('fps', [25., 30., 60.])
def test_sub_spacing_cycles_are_reported_as_unmatched_at_every_fps(fps):
    fixture = build_fixture(next(c for c in CASES if c.case_id == 'fast_cycles'), fps)
    row = evaluate_fixture(fixture, segment_squat(fixture.signal))
    assert row['candidate_count'] == 0 and row['reference_cycle_count'] == 3
    assert [miss['reference_index'] for miss in row['missed_references']] == [1, 2, 3]
    assert row['timing_errors']['top_start']['mae_seconds'] is None
    assert row['phase']['availability_coverage'] == 0


def test_variable_tempo_can_match_cycles_while_exact_bottom_phases_disagree():
    fixture = build_fixture(next(c for c in CASES if c.case_id == 'variable_tempo'), 30.)
    row = evaluate_fixture(fixture, segment_squat(fixture.signal))
    assert row['matched_count'] == row['reference_cycle_count'] == 4
    assert row['phase']['per_phase']['Bottom']['reference_frames'] == 4
    assert row['phase']['per_phase']['Bottom']['correct_frames'] == 1
    assert row['timing_frame_errors']['top_start']['max_abs_frames'] == 2


def test_bottom_hold_coverage_and_bottom_phase_recall_are_different_metrics():
    fixture = build_fixture(CASES[1], 30.)
    row = evaluate_fixture(fixture, segment_squat(fixture.signal))
    assert row['cycle_detection_coverage'] == row['phase']['availability_coverage'] == 1
    assert row['phase']['per_phase']['Bottom']['recall'] < .1
    assert row['phase']['accuracy_all_reference_frames'] < 1
    assert any(item['reference'] == 'Bottom' and item['prediction'] in ('Descent', 'Ascent')
               for item in row['phase']['confusion'])
    assert any(abs(m['descent_duration_error_seconds']) > .4 for m in row['matches'])


def test_gap_and_side_change_references_remain_in_main_denominator():
    for name, reason in (('missing_bottom', 'reference_crosses_unresolved_gap'),
                         ('explicit_side_switch', 'reference_crosses_explicit_side_switch')):
        fixture = build_fixture(next(c for c in CASES if c.case_id == name), 30.)
        row = evaluate_fixture(fixture, segment_squat(fixture.signal))
        assert row['reference_cycle_count'] == 3 and row['source_invalid_full_references'] == 1
        assert any(reason in miss['reasons'] for miss in row['missed_references'])
        assert row['cycle_detection_coverage'] < 1


def test_partial_reference_denominator_and_exact_frame_identity():
    fixture = build_fixture(CASES[-1], 30.)
    row = evaluate_fixture(fixture, segment_squat(fixture.signal))
    assert row['all_reference_cycles'] == 3
    assert row['boundary_partial_references'] == [1, 3]
    assert row['reference_cycle_count'] == row['matched_count'] == 1
    assert row['matches'][0]['reference_index'] == 2
    assert row['signed_count_error'] == 0


def test_matching_is_one_to_one_not_nearest_reuse():
    fixture = build_fixture(CASES[0], 30.)
    result = segment_squat(fixture.signal)
    assert match_cycles(fixture.references, result.repetitions[:1]) == ((0, 0),)
    assert match_cycles(fixture.references, ()) == ()
    assert match_cycles((), result.repetitions) == ()


def test_summary_weights_counts_and_reports_unavailable_timing():
    rows = []
    for name in ('steady_three', 'small_excursion'):
        fixture = build_fixture(next(c for c in CASES if c.case_id == name), 30.)
        rows.append(evaluate_fixture(fixture, segment_squat(fixture.signal)))
    summary = summarize(rows)
    assert summary['conditions'] == summary['execution_successes'] == 2
    assert summary['matched_cycles'] == summary['missed_cycles'] == 3
    assert summary['cycle_detection_coverage'] == .5
    assert summary['phase_availability_coverage'] == .5
    assert summary['timing_errors_matched_only']['top_start']['n'] == 3
    failed = summarize([{'execution_status': 'failed'}])
    assert failed['execution_failures'] == 1 and failed['cycle_detection_coverage'] is None


@pytest.mark.parametrize('fps', [True, 0., -1., float('nan'), float('inf'), 241.])
def test_invalid_sampling_refused(fps):
    with pytest.raises(ValueError):
        build_fixture(CASES[0], fps)


def test_invalid_definitions_and_mismatched_results_refused():
    with pytest.raises(ValidationError):
        CycleSpec(descent_seconds=float('nan'), ascent_seconds=1.)
    with pytest.raises(ValidationError):
        MissingInterval(start_seconds=2., end_seconds=1.)
    fixture = build_fixture(CASES[0], 30.)
    other = build_fixture(CASES[1], 30.)
    with pytest.raises(ValueError):
        evaluate_fixture(fixture, segment_squat(other.signal))
    with pytest.raises(ValueError):
        build_fixture(CASES[0].model_copy(update={'side_switch_seconds': float('nan')}), 30.)


def test_audit_never_overwrites_existing_output(tmp_path):
    sentinel = tmp_path / 'sentinel.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        run_audit(tmp_path)
    assert sentinel.read_text() == 'keep' and list(tmp_path.iterdir()) == [sentinel]


def test_profile_drift_is_refused_before_any_output(tmp_path, monkeypatch):
    import gym_coach.reporting.temporal_sensitivity as audit
    monkeypatch.setattr(audit, 'EXPECTED_PROFILE_HASH', '0' * 64)
    with pytest.raises(ValueError, match='Profile drift'):
        run_audit(tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_full_audit_does_not_import_backends_or_read_dataset(tmp_path):
    code = r'''
import importlib.abc,sys,io,builtins,json,hashlib
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
def guarded(fn):
 def open_file(path,*args,**kwargs):
  if not isinstance(path,int) and ('Data' in Path(path).parts or Path(path).suffix.lower() in {'.avi','.mp4'}):raise AssertionError('Forbidden real source: '+str(path))
  return fn(path,*args,**kwargs)
 return open_file
builtins.open=guarded(builtins.open);io.open=guarded(io.open)
from gym_coach.reporting.temporal_sensitivity import run_audit,EXPECTED_PROFILE_HASH
from gym_coach.segmentation.synthetic_fixtures import SyntheticCase,build_fixture
from gym_coach.segmentation import SquatSegmentationResult
output=Path(sys.argv[1]);s=run_audit(output)
assert s['conditions']==s['execution_successes']==30
assert s['execution_failures']==0 and s['profile_hash']==EXPECTED_PROFILE_HASH
m=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
for name,digest in m['files'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest
for path in output.glob('*/fixture.json'):
 payload=json.loads(path.read_text(encoding='utf-8'))
 definition=SyntheticCase.model_validate_json(json.dumps(payload['definition']),strict=True)
 fixture=build_fixture(definition,payload['input']['fps'])
 assert fixture.payload()==payload
 r=SquatSegmentationResult.model_validate_json((path.parent/'segmentation.json').read_text(encoding='utf-8'),strict=True)
 assert r.input==fixture.signal and not r.official_eligibility and not r.assessment_enabled
print('synthetic-audit-verified')
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path / 'audit')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'synthetic-audit-verified'
