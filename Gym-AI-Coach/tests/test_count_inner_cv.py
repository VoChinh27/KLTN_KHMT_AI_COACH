"""No learning, including smoke training: all gradients/steps/fitting are forbidden."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from gym_coach.counting.inner_cv import manifest as m, oof, runtime as rt
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json, file_info
from gym_coach.counting.research.contract import Config, INPUT_SIZE
from gym_coach.counting.research.data import identity
from gym_coach.counting.research.experiment import save_checkpoint, restore_state


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import torch
    from gym_coach.counting.research import baselines, experiment
    def guard():
        # Distinct sentinels: torch's lazy trace registry distinguishes grad/backward.
        def blocked(*args, **kwargs):
            raise AssertionError('Training, backward, optimizer step and fitting forbidden')
        return blocked
    for owner, name in ((torch.Tensor, 'backward'), (torch.autograd, 'backward'),
                        (torch.autograd, 'grad'), (torch.optim.Optimizer, 'step'),
                        (torch.optim.AdamW, 'step'), (baselines, 'fit_baselines'),
                        (experiment, 'train'), (experiment, 'run_baselines')):
        monkeypatch.setattr(owner, name, guard())


def rows():
    result = []
    for g, count in enumerate([3, 5, 1, 4, 1, 3, 1, 2, 2, 4, 4, 3], start=1):
        for c in range(count):
            video_id = f'fixture_g{g:02d}_c{c:02d}'
            result.append(dict(source=dict(video_id=video_id, group_id=f'g{g:02d}',
                sha256=hashlib.sha256(video_id.encode()).hexdigest()), component_id=f'component-{g}',
                role='train', input_status='accepted', supervised_eligible=True, target=c,
                label_record_hash=hashlib.sha256(('label-'+video_id).encode()).hexdigest()))
    return result


def frozen():
    selected, order, folds = m.partition(rows())
    return dict(version=m.VERSION, rows=selected, folds=folds, config=Config().model_dump(mode='json'),
                core_identity={'synthetic_fixture': True}, cv_code={})


def reports(manifest, error=None):
    error = error or (lambda e, f, row: abs(e-7)/10)
    result = []
    by_id = {r['source']['video_id']: r for r in manifest['rows']}
    for epoch in range(50):
        for fold in manifest['folds']:
            records = []
            for v in fold['evaluation_ids']:
                row = by_id[v]
                records.append(dict(**oof.pair_identity(row), prediction=row['target']+error(epoch, fold['fold'], row),
                                    status='estimated', reason=None))
            result.append(dict(epoch=epoch, fold=fold['fold'], manifest_sha256=canonical_hash(manifest), records=records))
    return result


def fixture_checkpoint(tmp_path, manifest, fold=0, epoch=1):
    import torch
    model, optimizer = rt.fresh_components(Config())
    # Synthetic optimizer slots only exercise serialization/restore. No update was performed.
    param = next(model.parameters())
    optimizer.state[param] = dict(step=torch.tensor(3.), exp_avg=torch.ones_like(param), exp_avg_sq=torch.ones_like(param)*2)
    ident = m.fold_identity(manifest, fold)
    history = []
    for report in reports(manifest):
        if report['fold'] == fold and report['epoch'] <= epoch:
            history.append(m.sealed(dict(**report, identity_sha256=canonical_hash(ident),
                                         model_sha256=rt.state_hash(model.state_dict()))))
    payload = rt.checkpoint_payload(ident, model, optimizer, epoch, 0, history, fixture=True)
    path = tmp_path/f'fixture-not-trained-fold{fold}.pt'
    save_checkpoint(path, payload)
    return path, payload


def test_exact_deterministic_group_assignment_coverage_and_order_independence():
    selected, ordered, folds = m.partition(rows())
    expected = sorted({r['source']['group_id'] for r in rows()},
        key=lambda g: hashlib.sha256(('squat-count-inner-cv-v1|'+g).encode()).hexdigest())
    assert ordered == expected
    assert m.partition(list(reversed(rows()))) == (selected, ordered, folds)
    assert len(selected) == 33
    assert sorted(v for f in folds for v in f['evaluation_ids']) == [r['source']['video_id'] for r in selected]
    for f in folds:
        assert f['evaluation_groups'] == sorted(expected[f['fold']::3])
        assert len(f['train_groups']) == 8 and len(f['evaluation_groups']) == 4
        assert not set(f['train_ids']) & set(f['evaluation_ids'])
        assert len(f['train_ids'])+len(f['evaluation_ids']) == 33


def test_assignment_ignores_labels_and_quality():
    modified = rows()
    for row in modified:
        row.update(target=999, quality={'invented': 'not used by fold assignment'})
    assert m.partition(modified)[1:] == m.partition(rows())[1:]


@pytest.mark.parametrize('mutation', ['remove', 'duplicate', 'null', 'rejected', 'fewer_groups'])
def test_bad_cohort_rejected(mutation):
    data = rows()
    if mutation == 'remove': data.pop()
    elif mutation == 'duplicate': data.append(data[0])
    elif mutation == 'null': data[0]['target'] = None
    elif mutation == 'rejected': data[0]['input_status'] = 'rejected'
    else: data[0]['source']['group_id'] = 'g02'; data[1]['source']['group_id'] = 'g02'; data[2]['source']['group_id'] = 'g02'
    with pytest.raises(ValueError): m.partition(data)


@pytest.mark.parametrize('key', ['sha256', 'component_id'])
def test_inner_leakage_blocks_instead_of_reassigning(key):
    data = rows()
    _, _, folds = m.partition(data)
    a = next(r for r in data if r['source']['video_id'] == folds[0]['train_ids'][0])
    b = next(r for r in data if r['source']['video_id'] == folds[0]['evaluation_ids'][0])
    if key == 'sha256': b['source'][key] = a['source'][key]
    else: b[key] = a[key]
    with pytest.raises(ValueError, match='Inner leakage'): m.partition(data)


@pytest.mark.parametrize('key', ['group_id', 'sha256', 'component_id'])
def test_outer_leakage_checked_even_excluded_rows(key):
    data = rows()
    outer = copy.deepcopy(data[0])
    outer.update(role='validation', supervised_eligible=False, component_id='other')
    outer['source'].update(video_id='outer', group_id='g90', sha256='f'*64)
    if key == 'component_id': outer[key] = data[0][key]
    else: outer['source'][key] = data[0]['source'][key]
    with pytest.raises(ValueError, match='Outer leakage'): m.partition(data+[outer])


def test_excludes_all_outer_rows_and_preserves_input():
    data = rows()
    for i, role in enumerate(('validation', 'development_test')):
        outer = copy.deepcopy(data[0]); outer.update(role=role, component_id=f'outer{i}')
        outer['source'].update(video_id=f'outer{i}', group_id=f'g9{i}', sha256=str(i)*64)
        data.append(outer)
    original = copy.deepcopy(data)
    selected, _, folds = m.partition(data)
    assert len(selected) == 33 and data == original
    assert all(r['role'] == 'train' for r in selected)


def test_manifest_roundtrip_and_code_tamper(tmp_path, monkeypatch):
    data = dict(rows=rows(), config=Config().model_dump(mode='json'))
    dataset = tmp_path/'dataset.json'; write_json(dataset, data)
    monkeypatch.setattr(m, 'LOCKED_DATASET_SHA256', file_info(dataset)['sha256'])
    monkeypatch.setattr(m, 'load_dataset', lambda root, path: data)
    manifest = m.prepare(tmp_path, dataset, tmp_path/'new')
    path = tmp_path/'new/manifest.json'
    assert m.load(tmp_path, path)[0] == manifest
    with pytest.raises(FileExistsError): m.prepare(tmp_path, dataset, tmp_path/'new')
    monkeypatch.setattr(m, 'code_identity', lambda: {'changed': True})
    with pytest.raises(ValueError, match='mismatch'): m.load(tmp_path, path)


def test_manifest_byte_tamper(tmp_path, monkeypatch):
    data = dict(rows=rows(), config=Config().model_dump(mode='json'))
    dataset = tmp_path/'dataset.json'; write_json(dataset, data)
    monkeypatch.setattr(m, 'LOCKED_DATASET_SHA256', file_info(dataset)['sha256'])
    monkeypatch.setattr(m, 'load_dataset', lambda root, path: data)
    m.prepare(tmp_path, dataset, tmp_path/'new')
    dataset.write_text('{}')
    with pytest.raises(ValueError): m.load(tmp_path, tmp_path/'new/manifest.json')


def test_independent_initialization_same_seed_no_shared_storage():
    import torch
    a, oa = rt.fresh_components(Config()); b, ob = rt.fresh_components(Config())
    assert a is not b and oa is not ob and oa.state == ob.state == {}
    assert rt.state_hash(a.state_dict()) == rt.state_hash(b.state_dict())
    assert all(x.data_ptr() != y.data_ptr() and torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert oa.param_groups[0]['lr'] == .001 and oa.param_groups[0]['weight_decay'] == .0001
    assert sum(p.numel() for p in a.parameters()) == 3666


def test_resume_restores_only_identity_rng_slots_and_history(tmp_path):
    import torch
    manifest = frozen(); path, original = fixture_checkpoint(tmp_path, manifest)
    loaded = rt.load_checkpoint(path, m.fold_identity(manifest, 0), allow_fixture=True)
    model, optimizer = rt.fresh_components(Config())
    epoch, steps = restore_state(loaded, model, optimizer)
    assert (epoch, steps) == (2, 0)
    assert rt.state_hash(model.state_dict()) == original['model_sha256']
    assert torch.equal(torch.get_rng_state(), original['torch_rng'])
    assert len(optimizer.state) == 1 and float(next(iter(optimizer.state.values()))['step']) == 3
    assert loaded['epoch_reports'] == original['epoch_reports']
    with pytest.raises(ValueError, match='user-trained'): rt.load_checkpoint(path, m.fold_identity(manifest, 0))
    with pytest.raises(ValueError, match='identity'): rt.load_checkpoint(path, m.fold_identity(manifest, 1), allow_fixture=True)


@pytest.mark.parametrize('key', ['model_sha256', 'history_sha256', 'epoch', 'training_steps', 'version'])
def test_checkpoint_resume_tamper_rejected(tmp_path, key):
    manifest = frozen(); _, payload = fixture_checkpoint(tmp_path, manifest)
    payload[key] = 3 if key in ('epoch', 'training_steps') else 'wrong'
    path = tmp_path/'tampered-fixture.pt'; save_checkpoint(path, payload)
    with pytest.raises(ValueError): rt.load_checkpoint(path, m.fold_identity(manifest, 0), allow_fixture=True)


def test_checkpoint_hash_and_no_overwrite(tmp_path):
    manifest = frozen(); path, payload = fixture_checkpoint(tmp_path, manifest)
    with pytest.raises(FileExistsError): save_checkpoint(path, payload)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'): rt.load_checkpoint(path, m.fold_identity(manifest, 0), allow_fixture=True)


def test_aggregate_rejects_fixture_checkpoint_and_missing_file(tmp_path):
    manifest = frozen(); path, _ = fixture_checkpoint(tmp_path, manifest)
    result = rt.aggregate_checkpoints(manifest, [path, tmp_path/'not-found.pt'])
    assert result['selection'] is None and result['complete_epochs'] == 0
    assert len(result['global_issues']) == 2
    assert result['curve'][0]['pooled'] == dict(expected=33, comparable=0, continuous_MAE=None)


def test_cached_train_subset_and_forward_only(tmp_path):
    import torch
    manifest = frozen(); data = dict(rows=manifest['rows'])
    for row in data['rows']:
        path = tmp_path/(row['source']['video_id']+'.json')
        write_json(path, dict(values=[[0.]*INPUT_SIZE for _ in range(5)], complete=[True]*5, duration_seconds=.2))
        row['input_artifact'] = identity(tmp_path, path)
    result = rt.forward_only(tmp_path, data, manifest)
    assert len(result['checks']) == 6 and result['parameters_unchanged']
    assert result['training_steps'] == result['baseline_fits'] == 0 and not result['checkpoint_created']
    with pytest.raises(ValueError): rt.subset(tmp_path, data, ['unknown_outer_video'])
    with pytest.raises(ValueError): rt.subset(tmp_path, data, [data['rows'][0]['source']['video_id']]*2)
    model, _ = rt.fresh_components(Config())
    dataset = rt.subset(tmp_path, data, manifest['folds'][0]['evaluation_ids'])
    before = rt.state_hash(model.state_dict())
    predicted = rt.predict(model, dataset)
    assert len(predicted) == len(dataset) and all(p['status'] == 'estimated' for p in predicted)
    assert rt.state_hash(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    (tmp_path/(dataset.rows[0]['source']['video_id']+'.json')).write_text('{}')
    failed = rt.predict(model, dataset)
    assert failed[0]['prediction'] is None and failed[0]['reason'] == 'prediction_operational_error'


def test_oof_pooled_not_unweighted_fold_mean_and_macro():
    manifest = frozen(); values = reports(manifest, lambda e, f, r: f+1)
    result = oof.aggregate(manifest, values); first = result['curve'][0]
    expected = sum((f['fold']+1)*len(f['evaluation_ids']) for f in manifest['folds'])/33
    assert first['pooled'] == dict(expected=33, comparable=33, continuous_MAE=expected)
    assert expected != 2 and first['macro_group_MAE'] == 2
    assert len(first['groups']) == 12
    assert result['selection']['epoch'] == 0  # earliest exact tie over all 50
    assert result['selection']['index5_minus_minimum'] == 0


def test_one_common_epoch_not_fold_best_stitching():
    manifest = frozen(); result = oof.aggregate(manifest, reports(manifest, lambda e, f, r: abs(e-[2, 7, 9][f])))
    selected = result['selection']
    assert selected['fold_best_epochs'] == {'0': 2, '1': 7, '2': 9}
    assert selected['fold_best_range'] == [2, 9] and selected['fold_best_population_sd'] > 0
    assert selected['pooled_oof_MAE'] > 0  # stitched fold-best would falsely be 0
    assert selected['pooled_oof_MAE'] == result['curve'][selected['epoch']]['pooled']['continuous_MAE']
    assert selected['index5_MAE'] == result['curve'][5]['pooled']['continuous_MAE']


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'missing_prediction'), ('duplicate', 'duplicate_prediction'),
    ('target', 'oof_pair_linkage_mismatch'), ('source', 'oof_pair_linkage_mismatch'),
    ('failed', 'fixture_backend_failure'), ('nan', 'invalid_prediction_contract'),
    ('bool', 'invalid_prediction_contract'), ('null', 'invalid_prediction_contract'),
    ('negative', 'invalid_prediction_contract'), ('extra', 'unexpected_oof_pair'),
    ('wrong_fold', 'unexpected_oof_pair'), ('duplicate_report', 'duplicate_fold_report')])
def test_oof_invalid_or_missing_never_shrinks_denominator(mutation, reason):
    manifest = frozen(); values = reports(manifest); record = values[0]['records'][0]
    if mutation == 'missing': values[0]['records'].pop(0)
    elif mutation == 'duplicate': values[0]['records'].append(copy.deepcopy(record))
    elif mutation == 'target': record['target'] += 1
    elif mutation == 'source': record['source_sha256'] = '0'*64
    elif mutation == 'failed': record.update(status='failed', prediction=None, reason='fixture_backend_failure')
    elif mutation == 'nan': record['prediction'] = float('nan')
    elif mutation == 'bool': record['prediction'] = True
    elif mutation == 'null': record['prediction'] = None
    elif mutation == 'negative': record['prediction'] = -1
    elif mutation == 'extra': values[0]['records'].append(dict(record, video_id='outer_validation_video'))
    elif mutation == 'wrong_fold': values[1]['records'].append(values[0]['records'].pop(0))
    else: values.append(copy.deepcopy(values[0]))
    result = oof.aggregate(manifest, values)
    assert result['status'] == 'INCOMPLETE_OR_INVALID' and result['selection'] is None
    assert result['complete_epochs'] == 49
    first = result['curve'][0]
    assert first['pooled']['expected'] == 33 and first['pooled']['continuous_MAE'] is None
    assert first['macro_group_MAE'] is None and len(first['records']) == 33
    assert reason in [i['code'] for i in first['issues']]


@pytest.mark.parametrize('mutation', ['missing_epoch', 'invalid_epoch', 'report_hash', 'wrong_epoch'])
def test_curve_incomplete_blocks_selection_even_with_some_complete_epochs(mutation):
    manifest = frozen(); values = reports(manifest)
    if mutation == 'missing_epoch': values = values[3:]
    elif mutation == 'invalid_epoch': values[0]['epoch'] = 50
    elif mutation == 'report_hash': values[0]['manifest_sha256'] = '0'*64
    else: values[0]['epoch'] = 1
    result = oof.aggregate(manifest, values)
    assert result['selection'] is None and len(result['curve']) == 50


def test_serialization_and_sealed_metadata():
    manifest = frozen(); result = oof.aggregate(manifest, reports(manifest))
    assert m.unseal(json.loads(json.dumps(m.sealed(result), allow_nan=False))) == result
    envelope = m.sealed(result); envelope['payload']['selection']['epoch'] = 49
    with pytest.raises(ValueError, match='hash'): m.unseal(envelope)


def test_train_refuses_without_explicit_user_run_before_any_io(tmp_path):
    with pytest.raises(PermissionError): rt.train(None, None, None, None, tmp_path/'never')
    from gym_coach.counting.inner_cv.cli import main
    with pytest.raises(SystemExit): main(['train', '--output-dir', str(tmp_path/'never')])
    assert not (tmp_path/'never').exists()


def test_user_loop_uses_frozen_loss_steps_and_only_inner_heldout_static():
    source = Path(rt.__file__).read_text()
    tree = ast.parse(source)
    train = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train')
    assert isinstance(train.body[0], ast.If)  # authorization guard before model/IO
    text = ast.unparse(train)
    for fragment in ('config.epochs', 'config.batch_size', 'config.seed + epoch', 'count_loss(model(batch), batch)',
                     'clip_grad_norm_(model.parameters(), 1.0', 'fresh_components(config)', 'predict(model, evaluation)'):
        assert fragment in text
    assert 'fit_baselines' not in text and 'best_model' not in text
    assert 'development_test' not in [n.value for n in ast.walk(train) if isinstance(n, ast.Constant) and n.value == 'development_test']


def test_old_checkpoint_reader_is_unchanged_and_new_version_rejected(tmp_path):
    from gym_coach.counting.research.experiment import load_checkpoint
    manifest = frozen(); path, _ = fixture_checkpoint(tmp_path, manifest)
    with pytest.raises(ValueError, match='identity'): load_checkpoint(path, m.fold_identity(manifest, 0))


def test_public_import_does_not_load_heavy_backends():
    code = "import sys; import gym_coach.counting.inner_cv.manifest; import gym_coach.counting.inner_cv.oof; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def test_other_dataset_blocked_before_loading(tmp_path, monkeypatch):
    path = tmp_path/'other.json'; write_json(path, dict(rows=rows()))
    def forbidden(*args): raise AssertionError('must not load another dataset')
    monkeypatch.setattr(m, 'load_dataset', forbidden)
    with pytest.raises(ValueError, match='approved locked'): m.prepare(tmp_path, path, tmp_path/'never')
    assert not (tmp_path/'never').exists()


def test_self_contained_history_aggregation_no_training(tmp_path, monkeypatch):
    manifest = frozen()
    paths = [fixture_checkpoint(tmp_path, manifest, fold=f, epoch=49)[0] for f in range(3)]
    original = rt.load_checkpoint
    # Test-only explicit fixture reader. CLI has no switch accepting synthetic checkpoints.
    monkeypatch.setattr(rt, 'load_checkpoint', lambda p, i: original(p, i, allow_fixture=True))
    result = rt.aggregate_checkpoints(manifest, paths)
    assert result['status'] == 'COMPLETE' and result['complete_epochs'] == 50
    assert result['selection']['epoch'] == 7 and result['selection']['pooled_oof_MAE'] == 0
    assert len(result['checkpoint_sources']) == 3
    duplicate = rt.aggregate_checkpoints(manifest, paths+[paths[0]])
    assert duplicate['selection'] is None and duplicate['complete_epochs'] == 0


def test_incomplete_checkpoint_history_remains_incomplete(tmp_path, monkeypatch):
    manifest = frozen()
    paths = [fixture_checkpoint(tmp_path, manifest, fold=f, epoch=1)[0] for f in range(3)]
    original = rt.load_checkpoint
    monkeypatch.setattr(rt, 'load_checkpoint', lambda p, i: original(p, i, allow_fixture=True))
    result = rt.aggregate_checkpoints(manifest, paths)
    assert result['complete_epochs'] == 2 and result['selection'] is None
    assert len(result['curve']) == 50 and result['curve'][2]['pooled']['expected'] == 33


def test_corrupt_checkpoint_container_reported(tmp_path):
    path = tmp_path/'bad.pt'; path.write_bytes(b'invalid container')
    write_json(Path(str(path)+'.sha256.json'), file_info(path))
    result = rt.aggregate_checkpoints(frozen(), [path])
    assert result['selection'] is None and result['global_issues'][0]['code'] == 'checkpoint_unavailable_or_invalid'


def test_cli_missing_checkpoints_writes_explicit_nonzero_receipt(tmp_path, monkeypatch):
    from gym_coach.counting.inner_cv import cli
    monkeypatch.setattr(cli, 'environment', lambda: dict(synthetic_fixture=True))
    monkeypatch.setattr(m, 'load', lambda root, path: (frozen(), {'rows': rows()}))
    out = tmp_path/'output'
    code = cli.main(['aggregate', '--manifest', 'fixture-only.json', '--output-dir', str(out)])
    assert code == 2
    value = m.unseal(json.loads((out/'oof.json').read_text()))
    assert value['status'] == 'INCOMPLETE_OR_INVALID' and value['selection'] is None
    assert value['curve'][0]['pooled']['comparable'] == 0


def test_cli_resume_guard_uses_new_output_and_fold_identity_static():
    tree = ast.parse(Path(rt.__file__).read_text())
    train = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train')
    text = ast.unparse(train)
    assert 'load_checkpoint(resume, identity)' in text
    assert 'exist_ok=False' in text and 'restore_state(previous, model, optimizer)' in text
    assert "previous['epoch_reports'][:]" in text
