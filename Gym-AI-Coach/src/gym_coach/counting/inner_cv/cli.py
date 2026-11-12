"""Prepare/verify/forward/aggregate without learning. Train is exclusively user-run."""
import argparse
import json
from pathlib import Path
import sys
import time

from ..readiness import write_json
from ..research.cli import environment
from . import manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'forward', 'train', 'aggregate'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--fold', type=int, choices=(0, 1, 2))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--user-run', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoints', type=Path, nargs='*')
    args = parser.parse_args(argv)
    if args.command == 'train' and not args.user_run:
        parser.error('Learning reserved for the user: --user-run required')
    if args.command == 'train' and args.fold is None:
        parser.error('--fold required')
    if args.command != 'train' and (args.user_run or args.resume or args.fold is not None):
        parser.error('Training options only apply to train')
    if args.command != 'aggregate' and args.checkpoints is not None:
        parser.error('--checkpoints only applies to aggregate')
    start = time.perf_counter()
    env = environment()
    if args.command == 'prepare':
        if not args.dataset:
            parser.error('--dataset required')
        frozen = manifest.prepare(args.root, args.dataset, args.output_dir)
        result = dict(manifest_sha256=manifest.canonical_hash(frozen), folds=frozen['folds'])
    else:
        if not args.manifest:
            parser.error('--manifest required')
        frozen, data = manifest.load(args.root, args.manifest)
        if args.command == 'train':
            from .runtime import train
            train(args.root, data, frozen, args.fold, args.output_dir, user_run=args.user_run, resume=args.resume)
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if args.command == 'verify':
            result = dict(manifest_sha256=manifest.canonical_hash(frozen), folds=frozen['folds'],
                          source_hashes_and_frozen_links_verified=True, training_steps=0)
        elif args.command == 'forward':
            from .runtime import forward_only
            result = forward_only(args.root, data, frozen)
        else:
            from .runtime import aggregate_checkpoints
            result = aggregate_checkpoints(frozen, args.checkpoints or [])
            write_json(args.output_dir/'oof.json', manifest.sealed(result))
    receipt = dict(command=[sys.executable, '-m', __package__+'.cli', *(argv if argv is not None else sys.argv[1:])],
        seconds=time.perf_counter()-start, environment=env, result=result)
    write_json(args.output_dir/'receipt.json', receipt)
    print(json.dumps(dict(output_dir=str(args.output_dir), seconds=receipt['seconds'],
                         status=result.get('status', 'COMPLETE')), ensure_ascii=False))
    return 2 if result.get('status') == 'INCOMPLETE_OR_INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
