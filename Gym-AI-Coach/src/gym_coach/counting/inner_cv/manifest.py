"""Freeze train-only folds without inspecting labels, quality or predictions for assignment."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..models import canonical_hash, STATUS
from ..readiness import file_info, write_json
from ..research.contract import Config
from ..research.data import load_dataset, read, identity, check_identity
from ..research.experiment import experiment_identity

VERSION = 'squat-count-inner-cv-v1'
SELECTION = 'pooled_oof_continuous_clip_MAE_earliest_minimum_all50'
LOCKED_DATASET_SHA256 = '97f3619c18e95e9d7a1384596368e9b5fd5c4628a5d3be693b9549c09037502d'


def code_identity():
    return {p.name: file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def sealed(payload):
    return dict(payload=payload, sha256=canonical_hash(payload))


def unseal(envelope):
    if canonical_hash(envelope['payload']) != envelope['sha256']:
        raise ValueError('Inner CV content hash mismatch')
    return envelope['payload']


def partition(rows):
    """Pure assignment; exact cohort and all outer boundaries are checked, never repaired."""
    if len({r['source']['video_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate video ID')
    selected = sorted((r for r in rows if r['role'] == 'train' and r['supervised_eligible']),
                      key=lambda r: r['source']['video_id'])
    groups = {r['source']['group_id'] for r in selected}
    if len(selected) != 33 or len(groups) != 12:
        raise ValueError('Exactly 33 supervised outer-train clips / 12 groups required')
    if any(r['input_status'] != 'accepted' or type(r['target']) is not int or r['target'] < 0
           for r in selected):
        raise ValueError('Invalid supervised admission/target')
    ordered = sorted(groups, key=lambda g: (hashlib.sha256((VERSION+'|'+g).encode()).hexdigest(), g))
    assignment = {g: i % 3 for i, g in enumerate(ordered)}
    # Full-cohort outer leakage checks, including rows excluded by admission.
    for key in ('group_id', 'sha256', 'component_id'):
        seen = {}
        for r in rows:
            value = r[key] if key == 'component_id' else r['source'][key]
            if not value:
                raise ValueError('Missing linkage: '+key)
            if value in seen and seen[value] != r['role']:
                raise ValueError('Outer leakage: '+key)
            seen[value] = r['role']
    folds = []
    for i in range(3):
        train = [r for r in selected if assignment[r['source']['group_id']] != i]
        evaluation = [r for r in selected if assignment[r['source']['group_id']] == i]
        for key in ('group_id', 'sha256', 'component_id'):
            def values(items):
                return {r[key] if key == 'component_id' else r['source'][key] for r in items}
            if values(train) & values(evaluation):
                raise ValueError('Inner leakage: '+key+'; assignment is not rebalanced')
        folds.append(dict(fold=i, train_ids=[r['source']['video_id'] for r in train],
            evaluation_ids=[r['source']['video_id'] for r in evaluation],
            train_groups=sorted({r['source']['group_id'] for r in train}),
            evaluation_groups=sorted({r['source']['group_id'] for r in evaluation})))
    return selected, ordered, folds


def build(root, dataset_path, data):
    selected, ordered, folds = partition(data['rows'])
    config = Config.model_validate_json(json.dumps(data['config']))
    return dict(version=VERSION, **STATUS,
        dataset=identity(root, dataset_path), config=config.model_dump(mode='json'),
        core_identity=experiment_identity(dataset_path, config), cv_code=code_identity(),
        assignment_rule='SHA256(squat-count-inner-cv-v1|gXX), ascending digest, round-robin 3',
        group_order=[dict(group=g, sha256=hashlib.sha256((VERSION+'|'+g).encode()).hexdigest()) for g in ordered],
        folds=folds, rows=selected, parent_cohort=len(data['rows']),
        parent_roles=dict(Counter(r['role'] for r in data['rows'])),
        excluded_from_inner_cv=len(data['rows'])-len(selected),
        selection=SELECTION, epochs=50, oof_clips=33, oof_groups=12,
        outer_validation_used=False, development_test_used=False, development_test_previously_exposed=True,
        target_kind='AI_weak_video_complete_count_not_independent_ground_truth',
        scope='selection stability diagnostic; no final refit; OOF selection is not independent evaluation')


def prepare(root, dataset_path, out):
    if file_info(dataset_path)['sha256'] != LOCKED_DATASET_SHA256:
        raise ValueError('Only the approved locked count-only dataset is supported')
    data = load_dataset(root, dataset_path)
    payload = build(root, dataset_path, data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'manifest.json', sealed(payload))
    return payload


def load(root, path):
    payload = unseal(read(path))
    dataset_path = check_identity(root, payload['dataset'])
    if payload['dataset']['sha256'] != LOCKED_DATASET_SHA256:
        raise ValueError('Dataset differs from the approved locked count-only dataset')
    data = load_dataset(root, dataset_path)
    if payload != build(root, dataset_path, data):
        raise ValueError('Frozen CV manifest/dataset/code/environment mismatch; no implicit migration')
    return payload, data


def fold_identity(manifest, fold):
    if type(fold) is not int or fold not in range(3):
        raise ValueError('Fold must be 0, 1 or 2')
    return dict(version=VERSION, manifest_sha256=canonical_hash(manifest), fold=fold,
                partition=manifest['folds'][fold], core_identity=manifest['core_identity'],
                cv_code=manifest['cv_code'], config_hash=canonical_hash(manifest['config']))
