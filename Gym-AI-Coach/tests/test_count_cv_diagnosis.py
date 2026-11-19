"""Saved-data statistics only. No real or synthetic parameter fitting in these tests."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cv_diagnosis', Path(__file__).parents[1]/'scripts/diagnose_squat_count_inner_cv.py')
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.fixture(autouse=True)
def no_learning():
    with diagnosis.no_learning_or_forward():
        yield


def cache_fixture():
    mask = [True]*20; mask[6:8] = [False, False]
    source = dict(video_id='synthetic', frame_count=20, fps=25.)
    values = []
    for i, valid in enumerate(mask):
        joints = [0., 0., .9, 1., 1., 0., 0.]*8
        if not valid: joints[:7] = [0., 0., .2, 0., 0., 1., 0.]
        values.append(joints+[i/25, 0. if i == 0 else 1/25])
    stats = dict(frames=20, complete_frames=18, complete_fraction=.9, longest_incomplete_span=2,
                 provenance=dict(observed=158, low_visibility=2))
    row = dict(source=source, input_stats=copy.deepcopy(stats))
    cache = dict(source=source, stats=stats, complete=mask, values=values, status='accepted', duration_seconds=.8)
    return row, cache


def records():
    return [dict(video_id=f'synthetic{i:02d}', group_id=f'g{i%12+1:02d}', fold=i%3, target=i%6,
        gru=i%6+1, constant=2, duration=i%6+.25, duration_seconds=1+i/10,
        frames=20, complete_frames=20 if i%2 else 19, complete_fraction=1 if i%2 else .95,
        longest_gap_frames=0 if i%2 else 1, valid_runs=1 if i%2 else 2,
        incomplete_frames=0 if i%2 else 1) for i in range(33)]


def test_exact_paired_denominator_contributions_and_no_mutation():
    rows = records(); before = copy.deepcopy(rows); result = diagnosis.analyze(rows)
    assert rows == before
    assert result['overall']['n'] == 33 and result['overall']['duration_wins'] == 33
    assert result['overall']['delta_gru_minus_duration'] == .75
    assert sum(r['pooled_MAE_gap_contribution'] for r in result['records']) == pytest.approx(.75)
    assert result['overall']['models']['gru']['signed_error'] == 1
    assert result['overall']['models']['gru']['over'] == 33
    assert result['overall']['models']['gru']['under'] == 0
    assert result['overall']['models']['duration']['mae'] == .25


@pytest.mark.parametrize('mutation', ['missing', 'duplicate'])
def test_missing_duplicate_rejected(mutation):
    rows = records()
    if mutation == 'missing': rows.pop()
    else: rows[-1] = rows[0]
    with pytest.raises(ValueError, match='33 unique'): diagnosis.analyze(rows)


def test_all_groupings_sum_to_33_and_empty_bins_null():
    result = diagnosis.analyze(records())
    for groups in result['grouped'].values():
        assert sum(g['n'] for g in groups.values()) == 33
    empty = result['grouped']['duration']['[5,infinity)']
    assert empty['n'] == 0 and empty['models']['gru']['mae'] is None
    assert empty['delta_gru_minus_duration'] is None


def test_frame_weighted_support_not_silent_clip_drop():
    rows = records(); rows[0].update(frames=100, complete_frames=75, complete_fraction=.75)
    result = diagnosis.analyze(rows)
    assert result['frames'] == 740 and result['complete_frames'] == 699
    assert result['complete_frame_fraction_weighted'] == 699/740
    assert result['clips_with_gaps'] == 17


def test_direction_and_exact_zero_not_rounded_direction():
    rows = records()[:3]
    rows[0].update(gru=rows[0]['target']+.01)
    rows[1].update(gru=rows[1]['target']-.01)
    rows[2].update(gru=rows[2]['target'])
    value = diagnosis.summarize(rows)['models']['gru']
    assert (value['under'], value['over'], value['exact']) == (1, 1, 1)


def test_support_mask_time_runs_and_saved_stats():
    row, cache = cache_fixture(); result = diagnosis.support_stats(row, cache)
    assert result['complete_frames'] == 18 and result['complete_fraction'] == .9
    assert result['valid_run_lengths'] == [6, 12] and result['longest_gap_frames'] == 2
    assert result['longest_gap_seconds'] == .08 and result['raw_joint_low_visibility'] == 2


@pytest.mark.parametrize('mutation', ['stats', 'time', 'mask', 'provenance', 'nonfinite', 'source', 'duration'])
def test_input_integrity_rejects_mismatch(mutation):
    row, cache = cache_fixture()
    if mutation == 'stats': cache['stats']['complete_frames'] = 19
    elif mutation == 'time': cache['values'][2][-2] = .99
    elif mutation == 'mask': cache['complete'][2] = False
    elif mutation == 'provenance': cache['values'][2][4] = 0.
    elif mutation == 'nonfinite': cache['values'][2][0] = float('nan')
    elif mutation == 'source': cache['source'] = dict(cache['source'], video_id='other')
    else: cache['duration_seconds'] = 999
    with pytest.raises(ValueError): diagnosis.support_stats(row, cache)


def test_correlation_and_dispersion_noncausal_math():
    assert diagnosis.correlation([1, 2, 3], [2, 4, 6])['r'] == pytest.approx(1)
    assert diagnosis.correlation([1], [2])['r'] is None
    assert diagnosis.correlation([1, 1], [2, 3])['r'] is None
    with pytest.raises(ValueError): diagnosis.correlation([1, 2], [3])
    assert diagnosis.describe([0, 2])['population_sd'] == 1


def test_table_keeps_all_video_ids_and_sign_convention():
    result = diagnosis.analyze(records()); text = diagnosis.table(result['records'])
    assert all(r['video_id'] in text for r in result['records'])
    assert 'Δ=AE GRU−AE duration' in text


def test_no_training_path_called_static():
    source = Path(diagnosis.__file__).read_text(encoding='utf-8')
    assert '.fit_baselines(' not in source and '.backward(' not in source and '.step(' not in source
    assert 'comparison_tool.compare(' in source and 'no_learning_or_forward()' in source
