"""Prepare/verify/forward/aggregate safely; only the human runs train --user-run."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

from ..inner_cv.manifest import sealed
from ..models import canonical_hash
from ..readiness import file_info, write_json
from ..research.cli import environment
from ..research.data import identity, local
from . import manifest
from .safety import no_learning


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'forward-only', 'train', 'aggregate'))
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--manifest', type=Path)
    p.add_argument('--original-manifest', type=Path)
    p.add_argument('--gru-oof', type=Path)
    p.add_argument('--comparison-dir', type=Path)
    p.add_argument('--zero-xy-dir', type=Path)
    p.add_argument('--fold', type=int, choices=(0, 1, 2))
    p.add_argument('--user-run', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--checkpoints', type=Path, nargs='+')
    p.add_argument('--output-dir', type=Path, required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.command == 'train' and (not args.user_run or args.fold is None):
        p.error('Training reserved for user: --user-run and --fold required')
    if args.command != 'train' and (args.user_run or args.resume or args.fold is not None):
        p.error('Training options apply only to train')
    if args.command == 'prepare':
        if not all((args.original_manifest, args.gru_oof, args.comparison_dir, args.zero_xy_dir)) or args.manifest:
            p.error('prepare requires --original-manifest --gru-oof --comparison-dir --zero-xy-dir; no --manifest')
    elif not args.manifest or any((args.original_manifest, args.gru_oof, args.comparison_dir, args.zero_xy_dir)):
        p.error('Non-prepare commands require --manifest; use frozen controls only')
    if (args.command == 'aggregate') != bool(args.checkpoints):
        p.error('--checkpoints required only for aggregate')
    out = local(args.root, args.output_dir)
    if out.exists():
        p.error('Output directory already exists; never overwrite, including on resume')
    started = time.perf_counter()
    env = environment()
    command = [sys.executable, '-m', __package__+'.cli', *(argv if argv is not None else sys.argv[1:])]
    try:
        with no_learning():
            if args.command == 'prepare':
                base, data, links, _ = manifest.verify_controls(args.root, args.original_manifest,
                    args.gru_oof, args.comparison_dir, args.zero_xy_dir)
                from .audit import audit_inputs
                audit = audit_inputs(args.root, data, base)
                frozen = manifest.build(base, links, audit)
                out.mkdir(parents=True, exist_ok=False)
                write_json(out/'manifest.json', sealed(frozen))
                write_json(out/'velocity-audit.json', sealed(audit))
                result = dict(status='PREPARED_NOT_TRAINED', manifest_sha256=canonical_hash(frozen),
                    folds=frozen['folds'], endpoint=manifest.ENDPOINT, training_steps=0, baseline_fits=0)
            else:
                frozen, data, controls = manifest.load(args.root, args.manifest)
                if args.command != 'train':
                    out.mkdir(parents=True, exist_ok=False)
                if args.command == 'verify':
                    result = dict(status='VERIFIED_NOT_TRAINED', manifest_sha256=canonical_hash(frozen),
                        frozen_code_dataset_controls_verified=True, clips=33, groups=12,
                        endpoint=manifest.ENDPOINT, folds=frozen['folds'], training_steps=0, baseline_fits=0)
                elif args.command == 'forward-only':
                    from .runtime import forward_only
                    result = forward_only(args.root, data, frozen)
                elif args.command == 'aggregate':
                    from .runtime import endpoint_reports
                    from .evaluation import compare_reports, report_markdown
                    paths = [local(args.root, path) for path in args.checkpoints]
                    reports, sources = endpoint_reports(frozen, paths)
                    result = compare_reports(frozen, reports, controls)
                    for source in sources:
                        source.update(identity(args.root, source['path']))
                    result['checkpoint_sources'] = sources
                    result['controls'] = frozen['controls']
                    write_json(out/'comparison.json', sealed(result))
                    write_json(out/'per-video-errors.json', sealed(result['records']))
                    (out/'REPORT.md').write_text(report_markdown(result), encoding='utf-8')
        if args.command == 'train':
            from .runtime import train
            train(args.root, data, frozen, args.fold, out, user_run=True,
                  resume=local(args.root, args.resume) if args.resume else None)
            result = dict(status='USER_TRAINING_COMPLETE', fold=args.fold, endpoint=manifest.ENDPOINT,
                          comparison_pending=True)
        write_json(out/'receipt.json', dict(command=command, seconds=time.perf_counter()-started,
            environment=env, result=result))
        write_json(out/'inventory.json', dict(files={f.name: file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out), status=result.get('status', result.get('kind')))))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'failure.json', dict(status='INCOMPLETE_OR_INVALID', command=command,
            detail=type(exc).__name__+': '+str(exc), expected_clips=33,
            comparison_complete=False, endpoint=manifest.ENDPOINT))
        print(json.dumps(dict(status='INCOMPLETE_OR_INVALID', detail=str(exc))), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
