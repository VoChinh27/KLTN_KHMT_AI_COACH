"""All tests forbid learning, including on synthetic fixtures. No model fits here."""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from gym_coach.counting import cv_baseline_comparison as pairing
from gym_coach.counting.inner_cv import manifest as cv, runtime as original_rt
from gym_coach.counting.inner_cv.oof import pair_identity, metric
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json
from gym_coach.counting.research.data import identity
from gym_coach.counting.research.contract import Config
from gym_coach.counting.research.experiment import save_checkpoint, restore_state
from gym_coach.counting.xy_ablation import transform as tr, manifest as m, runtime as rt, evaluation as ev, cli
from gym_coach.counting.xy_ablation.safety import no_learning


@pytest.fixture(autouse=True)
def forbid_all_learning():
    with no_learning():
        yield


def base_manifest():
    groups = sorted((f'g{i:02d}' for i in range(1, 13)),
                    key=lambda g: hashlib.sha256((cv.VERSION+'|'+g).encode()).hexdigest())
    sizes = {0: [4, 3, 3, 3], 1: [4, 3, 3, 3], 2: [2, 2, 2, 1]}
    rows = []
    for i, g in enumerate(groups):
        for j in range(sizes[i % 3][i // 3]):
            v = f'synthetic_{g}_c{j:02d}'
            rows.append(dict(source=dict(video_id=v, group_id=g, sha256=hashlib.sha256(v.encode()).hexdigest(),
                fps=25, frame_count=100), component_id='component-'+g, target=(i+j) % 6,
                label_record_hash=hashlib.sha256(('AI-'+v).encode()).hexdigest(),
                role='train', input_status='accepted', supervised_eligible=True))
    selected, _, folds = cv.partition(rows)
    return dict(rows=selected, folds=folds, config=Config().model_dump(mode='json'),
                core_identity={'synthetic_fixture': True}, cv_code={})


def manifest():
    return m.build(base_manifest(), {'synthetic_fixture': True})


def sample():
    return dict(video_id='synthetic', values=[[float(i+j+1) for i in range(58)] for j in range(5)],
                complete=[True, False, True, True, True], duration_seconds=.2, target=2)


class MemoryDataset:
    def __init__(self, rows, samples):
        self.rows, self.samples = rows, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def endpoint_reports(man, *, epoch=49, error=.25):
    lookup = {r['source']['video_id']: r for r in man['rows']}
    reports = []
    for f in man['folds']:
        ident = m.fold_identity(man, f['fold'])
        records = [dict(**pair_identity(lookup[v]), prediction=lookup[v]['target']+error,
                        status='estimated', reason=None) for v in f['evaluation_ids']]
        reports.append(dict(epoch=epoch, fold=f['fold'], manifest_sha256=canonical_hash(man),
            identity_sha256=canonical_hash(ident), model_sha256='synthetic', records=records,
            metric=metric(records), training_steps=0, training_loss=None))
    return reports


def controls(man):
    return {name: [dict(r, fold=report['fold']) for report in endpoint_reports(man, error=err)
                   for r in report['records']]
            for name, err in [('gru', .5), ('constant', .75), ('duration', 1.)]}


def fixture_checkpoint(tmp_path, man, fold=0, epoch=1):
    import torch
    model, optimizer = original_rt.fresh_components(Config())
    # Explicit synthetic slots for restore-only testing; no optimizer update.
    parameter = next(model.parameters())
    optimizer.state[parameter] = dict(step=torch.tensor(3.), exp_avg=torch.ones_like(parameter),
                                    exp_avg_sq=torch.ones_like(parameter)*2)
    ident = m.fold_identity(man, fold)
    history = []
    for e in range(epoch+1):
        report = endpoint_reports(man, epoch=e)[fold]
        report['model_sha256'] = original_rt.state_hash(model.state_dict())
        history.append(cv.sealed(report))
    payload = rt.checkpoint_payload(ident, model, optimizer, epoch, 0, history, fixture=True)
    path = tmp_path/f'synthetic-no-learning-{fold}.pt'
    save_checkpoint(path, payload)
    return path, payload


def test_only_exact_sixteen_xy_channels_change_no_source_mutation():
    original = sample(); before = copy.deepcopy(original)
    result = tr.zero_xy(original)
    assert tr.XY_INDICES == tuple(j*7+k for j in range(8) for k in (0, 1))
    assert len(set(tr.XY_INDICES)) == 16 and len(result['values'][0]) == 58
    for a, b in zip(original['values'], result['values']):
        assert [i for i in range(58) if a[i] != b[i]] == list(tr.XY_INDICES)
        assert all(b[i] == 0.0 for i in tr.XY_INDICES)
    assert original == before
    assert {k: v for k, v in result.items() if k != 'values'} == {k: v for k, v in before.items() if k != 'values'}
    assert result['complete'] is not original['complete']
    result['values'][0][2] = -99; result['complete'][0] = False
    assert original == before


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), True, '2'])
def test_transform_rejects_bad_channels_even_when_xy_would_be_zeroed(bad):
    s = sample(); s['values'][0][0] = bad
    with pytest.raises(ValueError): tr.zero_xy(s)


@pytest.mark.parametrize('change', ['empty', 'short', 'long', 'mask_length', 'mask_type'])
def test_transform_rejects_shapes(change):
    s = sample()
    if change == 'empty': s['values'] = []
    if change == 'short': s['values'][0].pop()
    if change == 'long': s['values'][0].append(1.)
    if change == 'mask_length': s['complete'].pop()
    if change == 'mask_type': s['complete'][0] = 1
    with pytest.raises(ValueError): tr.zero_xy(s)


def test_same_transform_train_evaluation_mask_and_metadata_preserved(monkeypatch):
    s = sample(); source = MemoryDataset([{'source': 'synthetic'}], [s])
    monkeypatch.setattr(original_rt, 'subset', lambda root, data, ids: source)
    train = rt.subset(None, None, ['train'])
    evaluation = rt.subset(None, None, ['eval'])
    assert train[0] == evaluation[0] == tr.zero_xy(s)
    assert train[0]['complete'][1] is False and train[0]['complete'][0] is True
    assert train.rows is source.rows and len(train) == 1
    assert tr.zero_xy(tr.zero_xy(s)) == tr.zero_xy(s)


def test_forward_shape_finite_weights_unchanged_and_initialization_independent():
    import torch
    from gym_coach.counting.research.network import collate
    a, oa = original_rt.fresh_components(Config())
    b, ob = original_rt.fresh_components(Config())
    before = original_rt.state_hash(a.state_dict())
    assert before == original_rt.state_hash(b.state_dict())
    assert oa is not ob and not oa.state and not ob.state
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters()))
    batched = collate([tr.zero_xy(sample())])
    assert batched['x'].shape == (1, 5, 58)
    assert batched['complete'].tolist() == [sample()['complete']]
    assert torch.count_nonzero(batched['x'][..., list(tr.XY_INDICES)]) == 0
    with torch.inference_mode():
        output = a(batched)
    assert output.shape == (1,) and torch.isfinite(output).all() and not output.requires_grad
    assert all(p.grad is None for p in a.parameters())
    assert original_rt.state_hash(a.state_dict()) == before
    assert sum(p.numel() for p in a.parameters()) == 3666


def test_fold_isolation_counts_determinism_and_frozen_config():
    base = base_manifest(); man = manifest()
    assert man['folds'] == base['folds'] and man['rows'] == base['rows'] and man['config'] == base['config']
    assert man['endpoint'] == {'epoch_index': 49, 'epoch_one_based': 50, 'selection': 'fixed_endpoint_no_minimum_search'}
    assert [(len(f['train_ids']), len(f['evaluation_ids'])) for f in man['folds']] == [(20,13),(20,13),(26,7)]
    assert sorted(v for f in man['folds'] for v in f['evaluation_ids']) == [r['source']['video_id'] for r in man['rows']]
    assert man == manifest()
    for f in man['folds']:
        assert len(f['train_groups']) == 8 and len(f['evaluation_groups']) == 4
        assert not set(f['train_ids']) & set(f['evaluation_ids'])
        assert not set(f['train_groups']) & set(f['evaluation_groups'])


@pytest.mark.parametrize('key', ['group_id', 'sha256', 'component_id'])
def test_manifest_rejects_leakage(key):
    base = base_manifest(); left = base['folds'][0]['train_ids'][0]; right = base['folds'][0]['evaluation_ids'][0]
    rows = {r['source']['video_id']: r for r in base['rows']}
    a, b = (rows[v] if key == 'component_id' else rows[v]['source'] for v in (left, right))
    b[key] = a[key]
    with pytest.raises(ValueError): m.build(base, {})


def test_training_loop_and_batch_order_identical_to_frozen_loop_without_execution():
    # Source comparison guards the entire parameter-update loop, not a smoke fit.
    original_file = Path(original_rt.__file__)
    def batch_loop(source):
        tree = ast.parse(source)
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        loop = next(n for n in train.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
        stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'records' for t in n.targets))
        return [ast.dump(n, include_attributes=False) for n in loop.body[:stop]]
    assert batch_loop(original_file.read_text(encoding='utf-8')) == batch_loop(inspect.getsource(rt))
    source = inspect.getsource(rt.train)
    assert source.index('if not user_run') < source.index('import torch')
    assert 'frozen_runtime.fresh_components(config)' in source
    assert "evaluation = subset(root, data, identity['partition']['evaluation_ids'])" in source
    assert "train_set = subset(root, data, identity['partition']['train_ids'])" in source


def test_resume_restores_model_optimizer_rng_without_learning(tmp_path):
    import torch
    man = manifest(); path, payload = fixture_checkpoint(tmp_path, man)
    saved = rt.load_checkpoint(path, man, 0, allow_fixture=True)
    model, optimizer = original_rt.fresh_components(Config())
    assert restore_state(saved, model, optimizer) == (2, 0)
    assert original_rt.state_hash(model.state_dict()) == payload['model_sha256']
    assert torch.equal(torch.get_rng_state(), payload['torch_rng'])
    assert len(optimizer.state) == 1 and float(next(iter(optimizer.state.values()))['step']) == 3
    with pytest.raises(ValueError, match='user-trained'): rt.load_checkpoint(path, man, 0)
    with pytest.raises(ValueError, match='identity'): rt.load_checkpoint(path, man, 1, allow_fixture=True)
    with pytest.raises(ValueError): rt.endpoint_reports(man, [path, path, path])


@pytest.mark.parametrize('key', ['transform', 'config_sha256', 'code', 'controls', 'endpoint'])
def test_checkpoint_rejects_other_transform_config_controls_or_code(tmp_path, key):
    man = manifest(); path, _ = fixture_checkpoint(tmp_path, man)
    changed = copy.deepcopy(man)
    changed[key] = {'changed': True} if key != 'config_sha256' else 'bad'
    with pytest.raises(ValueError, match='identity'): rt.load_checkpoint(path, changed, 0, allow_fixture=True)


@pytest.mark.parametrize('key', ['version', 'model_sha256', 'history_sha256', 'epoch', 'training_steps'])
def test_checkpoint_tampering_and_old_family_rejected(tmp_path, key):
    man = manifest(); _, payload = fixture_checkpoint(tmp_path, man)
    payload[key] = 50 if key in ('epoch', 'training_steps') else 'squat-count-inner-cv-checkpoint-v1'
    path = tmp_path/'tampered.pt'; save_checkpoint(path, payload)
    with pytest.raises(ValueError): rt.load_checkpoint(path, man, 0, allow_fixture=True)


def test_checkpoint_no_overwrite_byte_hash_and_history_pairing(tmp_path):
    man = manifest(); path, payload = fixture_checkpoint(tmp_path, man)
    with pytest.raises(FileExistsError): save_checkpoint(path, payload)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='byte hash'): rt.load_checkpoint(path, man, 0, allow_fixture=True)
    report = cv.unseal(payload['epoch_reports'][-1]); report['records'][0]['source_sha256'] = 'wrong'
    payload['epoch_reports'][-1] = cv.sealed(report); payload['history_sha256'] = canonical_hash(payload['epoch_reports'])
    other = tmp_path/'bad-link.pt'; save_checkpoint(other, payload)
    with pytest.raises(ValueError, match='linkage'): rt.load_checkpoint(other, man, 0, allow_fixture=True)


def test_comparison_metrics_pairing_delta_sign_range_sd_and_serialization():
    man = manifest(); result = ev.compare_reports(man, endpoint_reports(man), controls(man))
    assert result['status'] == 'COMPLETE' and result['clips'] == 33 and result['groups'] == 12
    assert result['delta_mean'] == -.25
    for method, error in [('ablation', .25), ('gru', .5), ('constant', .75), ('duration', 1.)]:
        mtr = result['metrics'][method]
        assert mtr['pooled'] == {'clips': 33, 'continuous_MAE': error}
        assert mtr['macro_group_MAE'] == error
        for key in ('folds', 'groups', 'counts'):
            assert sum(v['clips'] for v in mtr[key].values()) == 33
            assert all(v['continuous_MAE'] == error for v in mtr[key].values())
        vals = [r['predictions'][method] for r in result['records']]
        assert mtr['predictions']['population_sd'] == statistics.pstdev(vals)
        assert mtr['predictions']['minimum'] == min(vals) and mtr['predictions']['maximum'] == max(vals)
    assert cv.unseal(json.loads(json.dumps(cv.sealed(result)))) == result
    assert result['official_eligibility'] is False and result['assessment'] is False
    assert result['validation_status'] == 'NOT_VALIDATED'
    assert '## Paired video errors' in ev.report_markdown(result)


def test_weighted_pooled_vs_equal_group_denominator_and_positive_delta():
    man = manifest(); reports = endpoint_reports(man)
    for report in reports:
        for r in report['records']:
            r['prediction'] = r['target']+int(r['group_id'][1:])/2
    result = ev.compare_reports(man, reports, controls(man))
    errors = [r['absolute_errors']['ablation'] for r in result['records']]
    assert result['metrics']['ablation']['pooled']['continuous_MAE'] == math.fsum(errors)/33
    assert result['metrics']['ablation']['macro_group_MAE'] == sum(i/2 for i in range(1,13))/12
    assert result['delta_mean'] > 0
    assert result['delta_mean'] == pytest.approx(math.fsum(errors)/33-.5)


@pytest.mark.parametrize('change', ['missing_fold', 'duplicate_fold', 'epoch48', 'epoch50', 'mixed_epochs',
    'missing_clip', 'duplicate_clip', 'source', 'target', 'group', 'wrong_fold', 'null', 'nonfinite',
    'missing_control', 'control_hash', 'control_duplicate', 'manifest_hash'])
def test_comparison_fails_closed_no_reduced_denominator(change):
    man = manifest(); reports = endpoint_reports(man); ctrl = controls(man)
    first = reports[0]['records'][0]
    if change == 'missing_fold': reports.pop()
    if change == 'duplicate_fold': reports[1] = copy.deepcopy(reports[0])
    if change == 'epoch48':
        for r in reports: r['epoch'] = 48
    if change == 'epoch50':
        for r in reports: r['epoch'] = 50
    if change == 'mixed_epochs': reports[1]['epoch'] = 5
    if change == 'missing_clip': reports[0]['records'].pop()
    if change == 'duplicate_clip': reports[0]['records'][-1] = copy.deepcopy(first)
    if change == 'source': first['source_sha256'] = 'bad'
    if change == 'target': first['target'] += 1
    if change == 'group': first['group_id'] = 'g99'
    if change == 'wrong_fold': first['fold'] = 1
    if change == 'null': first.update(prediction=None, status='failed', reason='synthetic_failure')
    if change == 'nonfinite': first['prediction'] = float('nan')
    if change == 'missing_control': ctrl['duration'].pop()
    if change == 'control_hash': ctrl['gru'][0]['row_sha256'] = 'bad'
    if change == 'control_duplicate': ctrl['constant'][-1] = copy.deepcopy(ctrl['constant'][0])
    if change == 'manifest_hash': reports[0]['manifest_sha256'] = 'bad'
    with pytest.raises(ValueError): ev.compare_reports(man, reports, ctrl)


def test_cli_and_runtime_refuse_training_without_user_run(tmp_path):
    with pytest.raises(PermissionError): rt.train(None, None, None, None, tmp_path/'training')
    with pytest.raises(SystemExit) as error:
        cli.main(['train', '--fold', '0', '--manifest', 'none.json', '--output-dir', str(tmp_path/'new')])
    assert error.value.code == 2 and not (tmp_path/'new').exists()


def test_cli_existing_output_untouched(tmp_path):
    dest = tmp_path/'existing'; dest.mkdir(); (dest/'keep').write_text('keep')
    with pytest.raises(SystemExit):
        cli.main(['verify', '--root', str(tmp_path), '--manifest', 'missing.json', '--output-dir', str(dest)])
    assert list(dest.iterdir()) == [dest/'keep'] and (dest/'keep').read_text() == 'keep'


def test_cli_failure_receipt_no_complete_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'environment', lambda: {'synthetic_test': True})
    def bad(*args): raise ValueError('synthetic input hash mismatch')
    monkeypatch.setattr(m, 'load', bad)
    out = tmp_path/'new'
    assert cli.main(['aggregate', '--root', str(tmp_path), '--manifest', 'bad.json', '--checkpoints',
        'a.pt', 'b.pt', 'c.pt', '--output-dir', str(out)]) == 2
    failure = json.loads((out/'failure.json').read_text())
    assert failure['expected_clips'] == 33 and failure['comparison_complete'] is False
    assert not (out/'comparison.json').exists()


def test_light_import_and_no_fit_in_aggregate_source():
    command = ('import sys; import gym_coach.counting.xy_ablation.transform; '
               'import gym_coach.counting.xy_ablation.manifest; '
               'assert not any(m in sys.modules for m in ("torch","numpy","mediapipe","cv2"))')
    subprocess.run([sys.executable, '-c', command], check=True)
    tree = ast.parse(inspect.getsource(ev))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                   n.func.attr in ('fit_baselines', 'backward', 'step', 'train') for n in ast.walk(tree))


@pytest.mark.parametrize('change', ['none', 'transform', 'code', 'config', 'endpoint', 'input_bytes'])
def test_manifest_reload_verifies_transform_config_code_endpoint_and_input_bytes(tmp_path, monkeypatch, change):
    base = base_manifest()
    write_json(tmp_path/'dataset.json', cv.sealed({'rows': base['rows']}))
    base['dataset'] = identity(tmp_path, tmp_path/'dataset.json')
    for name in ('original.json', 'oof.json', 'baseline.json'):
        write_json(tmp_path/name, {'synthetic': True})
    links = dict(original_manifest=identity(tmp_path, tmp_path/'original.json'),
                 gru_oof=identity(tmp_path, tmp_path/'oof.json'), comparison_dir='synthetic',
                 comparison_files={'baseline.json': identity(tmp_path, tmp_path/'baseline.json')})
    frozen = m.build(base, links)
    ctrl = controls(frozen)
    monkeypatch.setattr(m, 'verify_controls', lambda *args: (base, links, ctrl))
    if change in ('transform', 'code', 'config', 'endpoint'): frozen[change] = {'changed': True}
    write_json(tmp_path/'manifest.json', cv.sealed(frozen))
    if change == 'input_bytes': (tmp_path/'baseline.json').write_text('changed')
    if change == 'none':
        loaded, data, stored_controls = m.load(tmp_path, tmp_path/'manifest.json')
        assert loaded == frozen and data['rows'] == base['rows'] and stored_controls == ctrl
    else:
        with pytest.raises(ValueError): m.load(tmp_path, tmp_path/'manifest.json')


@pytest.mark.parametrize('case', ['complete', 'wrong_epoch', 'duplicate', 'missing'])
def test_endpoint_checkpoint_coverage_no_best_epoch_substitution(tmp_path, monkeypatch, case):
    man = manifest()
    paths = [fixture_checkpoint(tmp_path, man, fold=f, epoch=48 if case == 'wrong_epoch' and f == 1 else 49)[0]
             for f in range(3)]
    loader = rt.load_checkpoint
    # Test-only fixture dispatch: production CLI has no allow-fixture switch.
    monkeypatch.setattr(rt, 'load_checkpoint', lambda path, payload, fold:
                        loader(path, payload, fold, allow_fixture=True))
    if case == 'duplicate': paths[1] = paths[0]
    if case == 'missing': paths.pop()
    if case == 'complete':
        reports, sources = rt.endpoint_reports(man, list(reversed(paths)))
        assert [s['fold'] for s in sources] == [0, 1, 2]
        assert all(r['epoch'] == 49 for r in reports)
        assert sum(len(r['records']) for r in reports) == 33
    else:
        with pytest.raises(ValueError): rt.endpoint_reports(man, paths)


def test_saved_sampler_order_verified_without_parameter_learning():
    import torch
    man = manifest(); ident = m.fold_identity(man, 0)
    report = endpoint_reports(man)[0]
    report['training_steps'] = 50*math.ceil(len(ident['partition']['train_ids'])/8)
    report['training_loss'] = .25  # synthetic report scalar, not a fitted loss
    ids = ident['partition']['train_ids']
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(Config().seed+49)).tolist()
    report['batch_order_sha256'] = canonical_hash([ids[i] for i in order])
    rt.validate_report(report, ident, man, 49)
    report['batch_order_sha256'] = 'changed'
    with pytest.raises(ValueError, match='sampler'): rt.validate_report(report, ident, man, 49)


def test_cli_aggregate_outputs_saved_controls_hashes_and_pairs(tmp_path, monkeypatch):
    man = manifest(); ctrl = controls(man); reports = endpoint_reports(man)
    monkeypatch.setattr(cli, 'environment', lambda: {'synthetic_test': True})
    monkeypatch.setattr(m, 'load', lambda *args: (man, {}, ctrl))
    paths = []
    for i in range(3):
        path = tmp_path/f'synthetic-{i}.json'; write_json(path, {'synthetic_fixture_only': True}); paths.append(path)
    sources = [dict(path=str(p), fold=i, epoch=49) for i, p in enumerate(paths)]
    monkeypatch.setattr(rt, 'endpoint_reports', lambda *args: (reports, sources))
    out = tmp_path/'comparison-fixture'
    assert cli.main(['aggregate', '--root', str(tmp_path), '--manifest', 'mock.json', '--checkpoints',
        *map(str, paths), '--output-dir', str(out)]) == 0
    result = cv.unseal(json.loads((out/'comparison.json').read_text()))
    assert result['status'] == 'COMPLETE' and result['clips'] == 33 and result['baseline_fits'] == 0
    assert len(cv.unseal(json.loads((out/'per-video-errors.json').read_text()))) == 33
    assert result['checkpoint_sources'][0]['path'] == 'synthetic-0.json'
    assert (out/'inventory.json').is_file() and (out/'REPORT.md').is_file()
