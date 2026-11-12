"""Bind approved frozen inputs and a separate ablation code/config identity."""
from pathlib import Path

from .. import cv_baseline_comparison as controls_tool
from ..inner_cv import manifest as cv
from ..models import STATUS, canonical_hash
from ..readiness import file_info
from ..research.data import check_identity, identity, local, read
from .transform import descriptor

VERSION = 'squat-count-xy-ablation-v1'
ENDPOINT = dict(epoch_index=49, epoch_one_based=50, selection='fixed_endpoint_no_minimum_search')


def code_identity():
    return {p.name: file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_controls(root, original_manifest, gru_oof, comparison_dir):
    """Replay saved predictions/coefficients, never fit baselines or execute a model."""
    root = Path(root).resolve()
    base, oof, evidence = controls_tool.verify_inputs(root, original_manifest, gru_oof)
    folder = local(root, comparison_dir)
    inventory = read(folder/'inventory.json')
    for name, expected in inventory['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or file_info(path) != expected:
            raise ValueError('Comparison inventory mismatch: '+name)
    inputs = cv.unseal(read(folder/'inputs.json'))
    verification = cv.unseal(read(check_identity(root, inputs['verification'])))
    if inputs['evidence'] != evidence or verification != evidence:
        raise ValueError('Comparison verification linkage mismatch')
    baseline = cv.unseal(read(folder/'baselines.json'))
    saved = cv.unseal(read(folder/'comparison.json'))
    gru = controls_tool.selected_gru(base, oof)
    replay = controls_tool.compare(base, gru, baseline)
    replay['inputs'] = evidence
    if canonical_hash(saved) != canonical_hash(replay):
        raise ValueError('Saved baseline comparison mismatch')
    if (folder/'REPORT.md').read_text(encoding='utf-8') != controls_tool.report_markdown(replay):
        raise ValueError('Saved comparison report mismatch')
    receipt = read(folder/'receipt.json')['result']
    if receipt['status'] != 'COMPLETE' or receipt['metrics'] != saved['metrics']:
        raise ValueError('Saved comparison receipt mismatch')
    files = {name: identity(root, folder/name) for name in
             ('inputs.json', 'baselines.json', 'comparison.json', 'inventory.json', 'receipt.json', 'REPORT.md')}
    links = dict(original_manifest=identity(root, local(root, original_manifest)),
        original_manifest_sha256=canonical_hash(base), gru_oof=identity(root, local(root, gru_oof)),
        comparison_dir=folder.relative_to(root).as_posix(), comparison_files=files,
        comparison_verification=inputs['verification'], controls_tool=controls_tool.tool_identity(),
        verification=evidence)
    controls = dict(gru=gru, constant=baseline['predictions']['constant'], duration=baseline['predictions']['duration'])
    return base, links, controls


def build(base, links):
    controls_tool.validate_partition(base)
    if [(len(f['train_ids']), len(f['evaluation_ids'])) for f in base['folds']] != [(20, 13), (20, 13), (26, 7)]:
        raise ValueError('Approved 20/20/26 versus 13/13/7 fold sizes changed')
    return dict(version=VERSION, **STATUS, transform=descriptor(), code=code_identity(),
        endpoint=ENDPOINT, controls=links, rows=base['rows'], folds=base['folds'],
        config=base['config'], config_sha256=canonical_hash(base['config']), core_identity=base['core_identity'],
        original_cv_code=base['cv_code'], epochs=50, oof_clips=33, oof_groups=12,
        initialization='frozen fresh_components for each fold; same seed; independent model/optimizer state',
        batch_order='torch.Generator.manual_seed(config.seed+epoch); randperm(train_ids); batch_size=8',
        inherited_selection_field='legacy Config.selection retained for compatibility; ignored; endpoint is authoritative',
        outer_validation_used=False, development_test_used=False, final_refit=False,
        development_test_previously_exposed=True,
        target_kind='AI_weak_reference_not_independent_ground_truth',
        scope='adaptive development ablation after viewing OOF; one seed; no equivalence or significance threshold',
        retained_pose_information='visibility/provenance/support/time retained; not removal of all pose information')


def load(root, path):
    payload = cv.unseal(read(local(root, path)))
    links = payload['controls']
    check_identity(root, links['original_manifest'])
    check_identity(root, links['gru_oof'])
    for item in links['comparison_files'].values():
        check_identity(root, item)
    base, current_links, controls = verify_controls(root, links['original_manifest']['path'],
        links['gru_oof']['path'], links['comparison_dir'])
    if payload != build(base, current_links):
        raise ValueError('Ablation manifest/code/transform/config/control identity mismatch; no implicit migration')
    # verify_controls already used frozen load_dataset; do not repeat its full-cohort audit.
    data = read(check_identity(root, base['dataset']))
    # Dataset is a sealed document, unlike the cached individual samples.
    data = cv.unseal(data)
    return payload, data, controls


def fold_identity(manifest, fold):
    if type(fold) is not int or fold not in range(3):
        raise ValueError('Fold must be 0, 1 or 2')
    return dict(version=VERSION, manifest_sha256=canonical_hash(manifest), fold=fold,
        partition=manifest['folds'][fold], transform=manifest['transform'],
        transform_sha256=canonical_hash(manifest['transform']), code=manifest['code'],
        config_sha256=manifest['config_sha256'], core_identity=manifest['core_identity'],
        original_cv_code=manifest['original_cv_code'], endpoint=ENDPOINT)
