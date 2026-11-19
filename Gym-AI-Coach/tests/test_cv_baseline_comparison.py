"""Synthetic baseline formula tests only; all real fitting and GRU learning prohibited."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from gym_coach.counting import cv_baseline_comparison as tool
from gym_coach.counting.inner_cv import manifest as cv, oof, runtime
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json, file_info
from gym_coach.counting.research import baselines, experiment
from gym_coach.counting.research.contract import Config
from gym_coach.counting.research.data import identity


@pytest.fixture(autouse=True)
def synthetic_fitting_only(monkeypatch):
    original = baselines.fit_baselines
    def guarded(rows):
        assert rows and all(r['source']['video_id'].startswith('fixture_') and
                            r['source']['path'].startswith('synthetic/') for r in rows), 'REAL fitting forbidden'
        return original(rows)
    monkeypatch.setattr(baselines, 'fit_baselines', guarded)
    import torch
    def guard():
        def blocked(*args, **kwargs): raise AssertionError('GRU training/backward/optimizer forbidden')
        return blocked
    for owner, name in ((torch.Tensor, 'backward'), (torch.autograd, 'backward'),
                        (torch.autograd, 'grad'), (torch.optim.Optimizer, 'step'),
                        (torch.optim.AdamW, 'step'), (experiment, 'train'), (runtime, 'train')):
        monkeypatch.setattr(owner, name, guard())


def manifest():
    rows = []
    for g, n in enumerate([3, 5, 1, 4, 1, 3, 1, 2, 2, 4, 4, 3], 1):
        for c in range(n):
            vid = f'fixture_g{g:02d}_c{c:02d}'
            duration = c+1
            rows.append(dict(source=dict(video_id=vid, path='synthetic/'+vid, group_id=f'g{g:02d}',
                sha256=hashlib.sha256(vid.encode()).hexdigest(), fps=25.0, frame_count=duration*25),
                component_id=f'group{g}', target=2*duration+1, role='train', input_status='accepted',
                supervised_eligible=True, label_record_hash=hashlib.sha256(('label'+vid).encode()).hexdigest()))
    selected, _, folds = cv.partition(rows)
    return dict(version=cv.VERSION, rows=selected, folds=folds, config=Config().model_dump(mode='json'),
                core_identity={'synthetic': True}, cv_code={})


def gru(man, errors=None):
    errors = errors or (lambda row, fold: fold+1.)
    rows = {r['source']['video_id']: r for r in man['rows']}
    return [dict(**oof.pair_identity(rows[v]), fold=f['fold'], prediction=rows[v]['target']+errors(rows[v], f['fold']),
                 status='estimated', reason=None) for f in man['folds'] for v in f['evaluation_ids']]


def saved_oof(man):
    reports = []
    for epoch in range(50):
        records = gru(man, lambda row, fold: (49-epoch)/8)
        for f in range(3):
            reports.append(dict(epoch=epoch, fold=f, manifest_sha256=canonical_hash(man),
                                records=[r for r in records if r['fold'] == f]))
    result = oof.aggregate(man, reports)
    result['checkpoint_sources'] = [dict(path=f'fixture-checkpoint{f}.pt',sha256='a'*64,size_bytes=0,fold=f,epoch=49) for f in range(3)]
    return result


def test_exact_formula_linear_clamping_and_even_median():
    man = manifest(); rows = copy.deepcopy(man['rows'][:2])
    for r, x, y in zip(rows, [1, 3], [0, 4]):
        r['source'].update(frame_count=x*25, fps=25.); r['target'] = y
    p = baselines.fit_baselines(rows)
    assert p['constant'] == 2 and p['duration_slope'] == 2 and p['duration_intercept'] == -2
    assert not p['duration_degenerate']
    probe = copy.deepcopy(rows[0]); probe['source']['frame_count'] = 5
    assert list(baselines.predict(p, [probe], 'train', 'duration').values()) == [0.]


def test_zero_variance_and_fractional_median_mean():
    rows = copy.deepcopy(manifest()['rows'][:4])
    for r, y in zip(rows, [0, 1, 4, 8]):
        r['target'] = y; r['source'].update(frame_count=100, fps=25.)
    p = baselines.fit_baselines(rows)
    assert p['constant'] == 2.5 and p['duration_slope'] == 0
    assert p['duration_intercept'] == 3.25 and p['duration_degenerate'] is True


def test_folds_fit_only_exact_train_ids_spy_and_do_not_mutate(monkeypatch):
    man = manifest(); before = copy.deepcopy(man); calls = []
    original = baselines.fit_baselines
    def spy(rows):
        calls.append([r['source']['video_id'] for r in rows]); return original(rows)
    monkeypatch.setattr(baselines, 'fit_baselines', spy)
    result = tool.fit_inner_baselines(man, user_run=True)
    assert man == before and calls == [f['train_ids'] for f in man['folds']]
    for fold in result['folds']:
        assert not set(fold['train_ids']) & set(fold['evaluation_ids'])
        assert len(fold['train_groups']) == 8 and len(fold['evaluation_groups']) == 4
        assert fold['parameters']['duration_slope'] == 2 and fold['parameters']['duration_intercept'] == 1
    assert len(result['predictions']['constant']) == len(result['predictions']['duration']) == 33


def test_heldout_targets_cannot_affect_own_fold_parameters():
    man = manifest(); before = tool.fit_inner_baselines(man, user_run=True)
    heldout = set(man['folds'][0]['evaluation_ids'])
    for r in man['rows']:
        if r['source']['video_id'] in heldout: r['target'] += 1000
    after = tool.fit_inner_baselines(man, user_run=True)
    assert before['folds'][0]['parameters'] == after['folds'][0]['parameters']


@pytest.mark.parametrize('key', ['sha256', 'component_id'])
def test_leakage_refused_before_fit(key, monkeypatch):
    man = manifest(); index = {r['source']['video_id']: r for r in man['rows']}; f = man['folds'][0]
    a, b = index[f['train_ids'][0]], index[f['evaluation_ids'][0]]
    if key == 'sha256': b['source'][key] = a['source'][key]
    else: b[key] = a[key]
    monkeypatch.setattr(baselines, 'fit_baselines', lambda rows: pytest.fail('fit must not occur'))
    with pytest.raises(ValueError, match='leakage'): tool.fit_inner_baselines(man, user_run=True)


@pytest.mark.parametrize('mutation', ['membership', 'outer_validation', 'development_test', 'group'])
def test_partition_reselection_or_outer_role_refused(mutation):
    man = manifest()
    if mutation == 'membership': man['folds'][0]['train_ids'].append(man['folds'][0]['evaluation_ids'][0])
    elif mutation == 'group': man['rows'][0]['source']['group_id'] = 'g99'
    else: man['rows'][0]['role'] = mutation
    with pytest.raises(ValueError): tool.fit_inner_baselines(man, user_run=True)


@pytest.mark.parametrize('field,value', [('fps', 0), ('fps', float('nan')), ('fps', True), ('frame_count', 0), ('frame_count', None)])
def test_duration_source_invalid(field, value):
    man = manifest(); man['rows'][0]['source'][field] = value
    with pytest.raises(ValueError, match='DURATION'): tool.validate_partition(man)


def test_metrics_and_paired_errors_exact_same_clips():
    man = manifest(); b = tool.fit_inner_baselines(man, user_run=True)
    result = tool.compare(man, gru(man), b)
    assert result['metrics']['duration']['pooled']['continuous_MAE'] == 0
    expected = sum((f['fold']+1)*len(f['evaluation_ids']) for f in man['folds'])/33
    assert result['metrics']['gru']['pooled'] == dict(clips=33, continuous_MAE=expected)
    assert result['metrics']['gru']['macro_group_MAE'] == 2 and expected != 2
    assert result['paired']['duration']['mean_gru_minus_baseline_absolute_error'] == expected
    for pair in result['paired']['duration']['records']:
        assert pair['gru_minus_baseline_absolute_error'] == pair['fold']+1
    assert [result['metrics']['gru']['folds'][str(f)]['clips'] for f in range(3)] == [len(f['evaluation_ids']) for f in man['folds']]
    reversed_gru = list(reversed(gru(man)))
    assert tool.compare(man, reversed_gru, b) == result


def test_negative_delta_convention():
    man = manifest(); b = tool.fit_inner_baselines(man, user_run=True)
    result = tool.compare(man, gru(man, lambda row, fold: 0), b)
    assert result['paired']['constant']['mean_gru_minus_baseline_absolute_error'] < 0
    assert all(r['gru_minus_baseline_absolute_error'] <= 0 for r in result['paired']['constant']['records'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'extra', 'source', 'label', 'target', 'fold', 'failed', 'null', 'nan', 'negative'])
def test_bad_prediction_pair_fails_no_denominator_shrink(mutation):
    man = manifest(); records = gru(man)
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records[-1] = copy.deepcopy(records[0])
    elif mutation == 'extra': records.append(dict(records[0], video_id='outer_video'))
    elif mutation == 'source': records[0]['source_sha256'] = '0'*64
    elif mutation == 'label': records[0]['label_record_hash'] = '0'*64
    elif mutation == 'target': records[0]['target'] += 1
    elif mutation == 'fold': records[0]['fold'] = 99
    elif mutation == 'failed': records[0].update(status='failed', prediction=None, reason='fixture_failure')
    elif mutation == 'null': records[0]['prediction'] = None
    elif mutation == 'nan': records[0]['prediction'] = float('nan')
    else: records[0]['prediction'] = -1
    with pytest.raises(ValueError, match='OOF_'): tool.metrics(man, records)


@pytest.mark.parametrize('mutation', ['parameters', 'full33_ids', 'train_record', 'prediction', 'prediction_origin', 'baseline_manifest'])
def test_parameter_and_prediction_hash_linkage(mutation):
    man = manifest(); b = tool.fit_inner_baselines(man, user_run=True)
    if mutation == 'parameters': b['folds'][0]['parameters']['constant'] += 1
    elif mutation == 'full33_ids': b['folds'][0]['parameters']['training_ids'] = [r['source']['video_id'] for r in man['rows']]
    elif mutation == 'train_record': b['folds'][0]['train_records'][0]['duration_seconds'] += 1
    elif mutation == 'prediction': b['predictions']['constant'][0]['prediction'] += 1
    elif mutation == 'prediction_origin': b['predictions']['constant'][0]['parameter_sha256'] = '0'*64
    else: b['manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='BASELINE_'): tool.compare(man, gru(man), b)


@pytest.mark.parametrize('mutation', ['epoch', 'metric', 'missing_epoch', 'manifest', 'outer', 'no_selection'])
def test_saved_gru_selection_not_changed_or_silently_accepted(mutation):
    man = manifest(); saved = saved_oof(man)
    if mutation == 'epoch': saved['selection'].update(epoch=48, epoch_one_based=49)
    elif mutation == 'metric': saved['selection']['pooled_oof_MAE'] += 1
    elif mutation == 'missing_epoch': saved['curve'].pop()
    elif mutation == 'manifest': saved['manifest_sha256'] = '0'*64
    elif mutation == 'outer': saved['development_test_used'] = True
    else: saved['selection'] = None
    with pytest.raises(ValueError, match='GRU_'): tool.selected_gru(man, saved)


def test_verify_replays_saved_reports_without_fit(tmp_path, monkeypatch):
    man = manifest(); saved = saved_oof(man)
    for source in saved['checkpoint_sources']:
        p = tmp_path/source['path']; p.write_bytes(b'synthetic-not-a-trained-checkpoint')
        source.update(file_info(p))
    path = tmp_path/'oof.json'; write_json(path, cv.sealed(saved))
    mp = tmp_path/'manifest.json'; write_json(mp, cv.sealed(man))
    monkeypatch.setattr(cv, 'load', lambda root, p: (man, {}))
    monkeypatch.setattr(tool, 'MANIFEST_HASH', canonical_hash(man))
    man['dataset'] = identity(tmp_path, mp)
    # Adding dataset changes hash: generate a newly coherent synthetic OOF fixture.
    updated = saved_oof(man); updated['checkpoint_sources'] = saved['checkpoint_sources']
    path.write_text(json.dumps(cv.sealed(updated)), encoding='utf-8')
    monkeypatch.setattr(tool, 'MANIFEST_HASH', canonical_hash(man))
    monkeypatch.setattr(runtime, 'aggregate_checkpoints', lambda m, ps: copy.deepcopy(updated))
    monkeypatch.setattr(baselines, 'fit_baselines', lambda rows: pytest.fail('verify cannot fit'))
    _, _, receipt = tool.verify_inputs(tmp_path, mp, path)
    assert receipt['real_baseline_fits'] == 0 and receipt['selected_epoch_one_based'] == 50
    assert receipt['comparison_clips'] == 33
    bad = copy.deepcopy(updated); bad['selection']['pooled_oof_MAE'] = 99
    path.write_text(json.dumps(cv.sealed(bad)), encoding='utf-8')
    with pytest.raises(ValueError, match='OOF_CHECKPOINT_MISMATCH'): tool.verify_inputs(tmp_path, mp, path)


def test_compare_command_guard_precedes_io(tmp_path):
    with pytest.raises(PermissionError): tool.run_comparison(None, None, tmp_path/'never')
    with pytest.raises(PermissionError): tool.fit_inner_baselines(None)
    with pytest.raises(SystemExit): tool.main(['compare', '--output-dir', str(tmp_path/'never')])
    assert not (tmp_path/'never').exists()


def test_synthetic_comparison_output_roundtrip_and_no_overwrite(tmp_path, monkeypatch):
    man = manifest(); saved = saved_oof(man)
    mp = tmp_path/'manifest.json'; op = tmp_path/'oof.json'
    write_json(mp, cv.sealed(man)); write_json(op, cv.sealed(saved))
    receipt = dict(version=tool.VERSION, tool=tool.tool_identity(), manifest=identity(tmp_path, mp), gru_oof=identity(tmp_path, op))
    verification = tmp_path/'verification.json'; write_json(verification, cv.sealed(receipt))
    monkeypatch.setattr(tool, 'verify_inputs', lambda root, m, o: (man, saved, receipt))
    out = tmp_path/'new'
    result = tool.run_comparison(tmp_path, verification, out, user_run=True)
    assert cv.unseal(json.loads((out/'comparison.json').read_text(encoding='utf-8'))) == result
    assert (out/'REPORT.md').exists() and (out/'inventory.json').exists()
    text = (out/'REPORT.md').read_text(encoding='utf-8')
    assert 'index49' in text and 'n=33' in text and 'không phải đánh giá độc lập' in text
    with pytest.raises(FileExistsError): tool.run_comparison(tmp_path, verification, out, user_run=True)
    op.write_text('{}')
    with pytest.raises(ValueError, match='changed'): tool.run_comparison(tmp_path, verification, tmp_path/'never', user_run=True)


def test_import_light():
    subprocess.run([sys.executable, '-c', "import sys; import gym_coach.counting.cv_baseline_comparison; assert not {'torch','numpy','cv2','mediapipe'} & set(sys.modules)"], check=True, capture_output=True)
