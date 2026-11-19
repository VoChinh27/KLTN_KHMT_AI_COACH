"""Freeze velocity condition and replay all four saved controls without fitting."""
from pathlib import Path

from .. import cv_baseline_comparison as controls_tool
from ..inner_cv import manifest as cv
from ..models import canonical_hash
from ..readiness import file_info
from ..research.data import check_identity, identity, local, read
from ..xy_ablation import manifest as xy_manifest, runtime as xy_runtime, evaluation as xy_evaluation
from .transform import descriptor

VERSION = 'squat-count-velocity-v1'
ENDPOINT = dict(epoch_index=49, epoch_one_based=50, selection='fixed_endpoint_no_minimum_search')


def code_identity():
    return {p.name: file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_controls(root, original_manifest, gru_oof, comparison_dir, zero_xy_dir):
    root = Path(root).resolve(); folder = local(root, zero_xy_dir)
    for name, expected in read(folder/'inventory.json')['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or file_info(path) != expected:
            raise ValueError('Zero-xy inventory mismatch: '+name)
    receipt = read(folder/'receipt.json')
    command = receipt['command']
    if command.count('--manifest') != 1:
        raise ValueError('Zero-xy manifest source missing/ambiguous')
    xy_path = local(root, command[command.index('--manifest')+1])
    xy, data, controls = xy_manifest.load(root, xy_path)
    original_links = xy['controls']
    if (original_links['original_manifest'] != identity(root, local(root, original_manifest))
            or original_links['gru_oof'] != identity(root, local(root, gru_oof))
            or local(root, original_links['comparison_dir']) != local(root, comparison_dir)):
        raise ValueError('Requested controls do not match zero-xy experiment controls')
    base = cv.unseal(read(check_identity(root, original_links['original_manifest'])))
    if xy['rows'] != base['rows'] or xy['folds'] != base['folds'] or xy['config'] != base['config']:
        raise ValueError('Zero-xy cohort/config mismatch')
    saved = cv.unseal(read(folder/'comparison.json'))
    sources = saved['checkpoint_sources']
    paths = [check_identity(root, item) for item in sources]
    reports, replay_sources = xy_runtime.endpoint_reports(xy, paths)
    for source in replay_sources:
        source.update(identity(root, source['path']))
    replay = xy_evaluation.compare_reports(xy, reports, controls)
    replay.update(checkpoint_sources=replay_sources, controls=xy['controls'])
    if canonical_hash(replay) != canonical_hash(saved) or receipt['result'] != saved:
        raise ValueError('Zero-xy saved comparison/receipt differs from checkpoint replay')
    if (cv.unseal(read(folder/'per-video-errors.json')) != replay['records']
            or (folder/'REPORT.md').read_text(encoding='utf-8') != xy_evaluation.report_markdown(replay)):
        raise ValueError('Zero-xy saved per-video errors/report mismatch')
    zero_predictions = controls_tool.paired_records(base,
        [dict(r, fold=report['fold']) for report in reports for r in report['records']])
    links = dict(original_links, zero_xy=dict(directory=folder.relative_to(root).as_posix(),
        manifest=identity(root, xy_path), manifest_sha256=canonical_hash(xy),
        files={name: identity(root, folder/name) for name in
               ('comparison.json', 'per-video-errors.json', 'REPORT.md', 'receipt.json', 'inventory.json')},
        checkpoint_sources=replay_sources, metrics=replay['metrics'], replay_exact=True))
    return base, data, links, dict(controls, zero_xy=zero_predictions)


def build(base, links, audit):
    result = xy_manifest.build(base, links)
    result.update(version=VERSION, transform=descriptor(), code=code_identity(), input_audit=audit,
        input_audit_sha256=canonical_hash(audit), endpoint=ENDPOINT,
        scope='one explicit-velocity condition after viewing OOF/zero-xy; adaptive development, not independent evaluation',
        retained_pose_information='visibility/provenance/support/time retained; velocities are image-height/s, not m/s')
    return result


def load(root, path):
    payload = cv.unseal(read(local(root, path)))
    links = payload['controls']
    for item in (links['original_manifest'], links['gru_oof'], links['zero_xy']['manifest'],
                 *links['comparison_files'].values(), *links['zero_xy']['files'].values()):
        check_identity(root, item)
    base, data, current_links, controls = verify_controls(root, links['original_manifest']['path'],
        links['gru_oof']['path'], links['comparison_dir'], links['zero_xy']['directory'])
    from .audit import audit_inputs
    audit = audit_inputs(root, data, base)
    if payload != build(base, current_links, audit):
        raise ValueError('Velocity manifest/code/transform/config/controls/input-audit mismatch; no migration')
    return payload, data, controls


def fold_identity(manifest, fold):
    if type(fold) is not int or fold not in range(3):
        raise ValueError('Fold must be 0, 1 or 2')
    return dict(version=VERSION, manifest_sha256=canonical_hash(manifest), fold=fold,
        partition=manifest['folds'][fold], transform=manifest['transform'],
        transform_sha256=canonical_hash(manifest['transform']), code=manifest['code'],
        config_sha256=manifest['config_sha256'], core_identity=manifest['core_identity'],
        original_cv_code=manifest['original_cv_code'], endpoint=ENDPOINT)
