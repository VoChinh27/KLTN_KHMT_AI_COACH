"""Saved evidence replay only; no decoding, inference, training or fitting."""
import importlib.util
from pathlib import Path
from ..inner_cv.manifest import unseal
from ..models import canonical_hash
from ..research.data import local, read, identity


def diagnosis_tool(root):
    path = local(root, 'scripts/diagnose_squat_cv500_fold0.py')
    spec = importlib.util.spec_from_file_location('cv500_seed_source_diagnosis', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def verify(root, parent_manifest, comparison, diagnosis_report, review_path):
    tool = diagnosis_tool(root)
    sources = tool.Sources(root)
    sources.add('scripts/diagnose_squat_cv500_fold0.py')
    report = sources.add(diagnosis_report)
    folder = report.parent
    inventory = sources.json(folder/'artifact-inventory.json')
    for name, expected in inventory['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Diagnosis inventory escapes run')
        sources.add(path, expected)
    receipt = sources.json(folder/'analysis-v1/receipt.json')
    command = receipt['command']
    previous_review = command[command.index('--review')+1]
    parent, data, saved, replay = tool.verify(root, parent_manifest, comparison, previous_review, sources)
    from ..research.experiment import experiment_identity
    from ..research.contract import Config as InputConfig
    from ..inner_cv import manifest as original_cv
    from ..inner_cv_500 import manifest as parent_cv
    if (experiment_identity(local(root,parent['dataset']['path']),InputConfig()) != parent['core_identity']
            or original_cv.code_identity() != parent['cv_code'] or parent_cv.code_identity() != parent['code']):
        raise ValueError('Frozen code/environment identity changed')
    review = sources.json(review_path)
    analysis = sources.json(folder/'analysis-v1/analysis.json')
    if (local(root, review['run']) != folder or not review['prediction_replay']
            or review['heldout_clips'] != 13 or review['train_clips'] != 20
            or review['continuous_playback'] is not False):
        raise ValueError('Fold0 review identity/status mismatch')
    # The recorded diagnosis includes sampled visuals, not independent count validation.
    measured = saved['endpoints']['499']['results']['cv500']
    if review['heldout_metrics'] != analysis['held_out']['overall']:
        raise ValueError('Fold0 diagnosis/review mismatch')
    if review['heldout_metrics']['models']['gru500']['MAE'] != measured['folds']['0']['held_out']['MAE']:
        raise ValueError('Fold0 review/actual metric mismatch')
    for item in sources.json(folder/'analysis-v1/preservation.json')['sources']:
        sources.link(item)
    for row in parent['rows']:
        sources.link(row['input_artifact'])
        for link in row['source_links'] + row['label_links']:
            sources.link(link)
    links = dict(parent_manifest=identity(root,parent_manifest), comparison=identity(root,local(root,comparison)/'result.json'),
        diagnosis_report=identity(root,report), review=identity(root,review_path),
        source_verifier=identity(root,local(root,'scripts/diagnose_squat_cv500_fold0.py')),
        diagnostic_tool=parent['controls']['diagnostic_tool'],
        baseline=parent['controls']['verified_lineage']['baseline'])
    evidence = dict(replay=replay, old_metrics={method:q['aggregate'] for method,q in saved['endpoints']['499']['results'].items()},
        fold0=measured['folds']['0'], source_files=sources.preserve(),
        full_playback=False, new_model_forwards=0, training_steps=0, baseline_fits=0)
    return parent, data, saved, links, evidence
