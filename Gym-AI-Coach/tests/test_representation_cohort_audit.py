"""Audit safety and evidence linkage, with no training or model forward."""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cohort_audit', Path(__file__).parents[1] / 'scripts/audit_squat_representations.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture(autouse=True)
def guard():
    with d.helper().previous_tool().no_learning_or_forward():
        yield


def sample(mask=(True, True, False, True, True)):
    source = dict(video_id='v', sha256='hash', frame_count=len(mask), fps=25., width=320, height=240)
    rows = []
    for i, complete in enumerate(mask):
        row = []
        for j in range(8):
            row.extend([i*.01+j*.001, j*.001, .99, int(complete), 1, 0, 0])
        rows.append(row+[i/25, 0 if i == 0 else 1/25])
    return dict(source=source, complete=list(mask), values=rows, duration_seconds=len(mask)/25), source


def test_ranking_keeps_gaps_singletons_and_source_unchanged():
    c, s = sample((True, False, True, True, False, True))
    before = copy.deepcopy(c)
    r = d.rank_windows(c, s)
    assert c == before
    assert [(x['previous_frame'], x['frame_index']) for x in r['edges']] == [(2, 3)]
    assert all(x['status'] == 'suspected_candidate_only' for x in r['edges'])
    assert d.rank_windows(*sample((True, False, True)))['selected_edges'] == []


def test_tie_break_and_pixel_crossing_are_not_confirmation():
    c, s = sample((True,)*4)
    for row in c['values']:
        row[:56] = c['values'][0][:56]
    assert d.rank_windows(c, s)['selected_edges'] == [1]
    c['values'][1][0], c['values'][1][7] = c['values'][1][7], c['values'][1][0]
    r = d.rank_windows(c, s)
    assert max(x['flip_count'] for x in r['edges']) == 1
    assert all(x['status'] == 'suspected_candidate_only' for x in r['edges'])
    assert r == d.rank_windows(c, s)


@pytest.mark.parametrize('case', ['source', 'frames', 'duration', 'timestamp', 'dt', 'mask', 'shape', 'nonfinite'])
def test_fail_closed_source_frame_timeline_and_mask(case):
    c, s = sample(); s = copy.deepcopy(s)
    if case == 'source': s['sha256'] = 'other'
    if case == 'frames': c['values'].pop()
    if case == 'duration': c['duration_seconds'] += .04
    if case == 'timestamp': c['values'][1][56] += .04
    if case == 'dt': c['values'][1][57] = 0
    if case == 'mask': c['complete'][0] = False
    if case == 'shape': c['values'][0].pop()
    if case == 'nonfinite': c['values'][0][0] = float('nan')
    with pytest.raises(ValueError): d.rank_windows(c, s)


def test_overlay_inverse_height_units_and_timestamp():
    c, s = sample()
    r = c['values'][0]; r[0:2] = [-.5, .25]
    assert d.pixels(r, 0, s) == (40., 180.)
    assert d.timestamp(4, s) == .16
    r[3] = 0; r[0] = float('nan')
    assert d.pixels(r, 0, s) is None
    for i in (-1, 5, 1.5, True):
        with pytest.raises(ValueError): d.timestamp(i, s)


def review():
    _, s = sample((True, True))
    original = dict(path='original.jpg', sha256='image1', frame_indices=[0, 1], timestamps_seconds=[0., .04], actually_viewed=True, kind='original_contact_sheet')
    overlay = dict(original, path='overlay.jpg', sha256='image2', kind='raw_pose_overlay')
    records = [dict(video_id='v', source_sha256='hash', independent_ground_truth=False, human_reviewed=False, count_status='unverified', notes='AI image review only', viewed_evidence=[original, overlay])]
    issue = dict(issue_id='suspected', video_id='v', source_sha256='hash', status='suspected', type='joint_drift', frame_indices=[1], timestamps_seconds=[.04], detection='rank', verification_evidence=['overlay.jpg'], verification_method='image_pose_comparison', known_impact='unknown', proposed_handling='future version only', risks='uncertain', previous_issue=None)
    inv = {('v', e['path']):copy.deepcopy(e) for e in records[0]['viewed_evidence']}
    return records, {'v':s}, [issue], inv


def test_review_never_promotes_or_claims_independent_count():
    args = review(); before = copy.deepcopy(args)
    q = d.validate_review(*args)
    assert args == before and q['confirmed'] == 0 and q['suspected'] == 1
    assert q['independent_count_labels'] == q['AI_semantics_consistent'] == 0


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'source', 'human', 'independent', 'unviewed', 'frame', 'time', 'hash', 'path', 'coverage', 'duplicate_issue', 'issue_source', 'issue_time', 'numeric_confirm', 'missing_overlay'])
def test_bad_review_or_ledger_rejected(case):
    records, expected, issues, inventory = review()
    r = records[0]; e = r['viewed_evidence'][0]; x = issues[0]
    if case == 'missing': records.clear()
    if case == 'duplicate': records.append(copy.deepcopy(r))
    if case == 'source': r['source_sha256'] = 'other'
    if case == 'human': r['human_reviewed'] = True
    if case == 'independent': r['independent_ground_truth'] = True
    if case == 'unviewed': e['actually_viewed'] = False
    if case == 'frame': e['frame_indices'] = [0, 0]
    if case == 'time': e['timestamps_seconds'] = [0, .08]
    if case == 'hash': e['sha256'] = 'other'
    if case == 'path': e['path'] = 'unknown.jpg'
    if case == 'coverage': r['viewed_evidence'].pop(0)
    if case == 'duplicate_issue': issues.append(copy.deepcopy(x))
    if case == 'issue_source': x['source_sha256'] = 'other'
    if case == 'issue_time': x['timestamps_seconds'] = [.08]
    if case == 'numeric_confirm': x.update(status='confirmed', verification_method='numeric_ranking')
    if case == 'missing_overlay': x.update(status='confirmed', verification_evidence=['unseen.jpg'])
    with pytest.raises(ValueError): d.validate_review(records, expected, issues, inventory)


def test_source_preservation_detects_mutation_and_wrong_hash(tmp_path):
    p = tmp_path/'input.json'; p.write_text('original')
    s = d.helper().Sources(tmp_path); s.add(p)
    assert s.preserve()['status'] == 'UNCHANGED'
    with pytest.raises(ValueError): s.add(p, {'sha256':'wrong'})
    p.write_text('mutated')
    with pytest.raises(ValueError): s.preserve()


def test_audit_has_no_learning_or_forward_calls():
    tree = ast.parse(Path(d.__file__).read_text(encoding='utf8'))
    forbidden = {'train', 'backward', 'step', 'fit', 'forward', 'predict', 'fit_baselines', 'restore_state'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]
