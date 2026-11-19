"""Prepare/verify/forward-only and compare without learning; training is user opt-in."""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import sys
import time
from ..inner_cv.manifest import sealed
from ..models import canonical_hash
from ..readiness import write_json,file_info
from ..research.cli import environment
from ..research.data import local,check_identity
from . import manifest
from .contract import SEEDS
from .safety import no_learning


def summary(audit):
    scales=[q['scale'] for q in audit.values()]
    return dict(unique_videos=len(audit),scale_units='image_height',scale_min=min(scales),
        scale_median=statistics.median(scales),scale_max=max(scales),scale_mean=statistics.mean(scales),
        scale_population_sd=statistics.pstdev(scales),total_frames=sum(q['total_frames'] for q in audit.values()),
        complete_frames=sum(q['complete_frames'] for q in audit.values()),
        scale_frames=sum(q['scale_frames'] for q in audit.values()),
        excluded_scale_frames=sum(q['excluded_scale_frames'] for q in audit.values()),
        normalized_complete_xy_min=min(q['xy_min'] for q in audit.values()),
        normalized_complete_xy_max=max(q['xy_max'] for q in audit.values()),
        max_abs_hip_midpoint=max(q['max_abs_hip_midpoint'] for q in audit.values()))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','forward-only','train','aggregate'))
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--fold',type=int,choices=(0,1,2))
    p.add_argument('--user-run',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--checkpoints',nargs=6,type=Path)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command=='train':
        if not args.user_run or args.seed is None or args.fold is None:
            p.error('Only user may train: --user-run --seed --fold required')
    elif args.user_run or args.seed is not None or args.fold is not None or args.resume:
        p.error('Training options only for train')
    if (args.command=='prepare') == bool(args.manifest):
        p.error('Use --manifest except for prepare, which verifies locked source receipts')
    if (args.command=='aggregate') != bool(args.checkpoints):
        p.error('Exactly six --checkpoints required only for aggregate')
    root=args.root.resolve();out=local(root,args.output_dir)
    if out.exists():p.error('Output directory exists; never overwrite, including resume')
    command=[sys.executable,'-m',__package__+'.cli',*(sys.argv[1:] if argv is None else argv)]
    start=time.perf_counter()
    try:
        with no_learning():
            if args.command=='prepare':
                frozen,data,controls=manifest.prepare(root)
                out.mkdir(parents=True,exist_ok=False)
                write_json(out/'manifest.json',sealed(frozen))
                write_json(out/'configs.json',sealed(frozen['configs']))
                write_json(out/'saved-controls.json',sealed(controls))
                write_json(out/'input-audit.json',sealed(frozen['input_audit']))
                write_json(out/'source-verification.json',sealed(frozen['sources']))
                from .runtime import forward_only
                preflight=forward_only(root,data,frozen)
                write_json(out/'preflight.json',sealed(preflight))
                result=dict(status='PREPARED_USER_TRAINING_PENDING',manifest_sha256=canonical_hash(frozen),
                    preflight=preflight,distribution=summary(frozen['input_audit']),
                    source_files_verified=len(frozen['sources']['sources']),training_steps=0,baseline_fits=0)
            else:
                frozen,data,controls=manifest.load(root,args.manifest)
                if args.command!='train':out.mkdir(parents=True,exist_ok=False)
                if args.command=='verify':
                    result=dict(status='VERIFIED_USER_TRAINING_PENDING',manifest_sha256=canonical_hash(frozen),
                        source_files_verified=len(frozen['sources']['sources']),
                        distribution=summary(frozen['input_audit']),training_steps=0,baseline_fits=0)
                elif args.command=='forward-only':
                    from .runtime import forward_only
                    result=forward_only(root,data,frozen)
                    write_json(out/'preflight.json',sealed(result))
                elif args.command=='aggregate':
                    from .evaluation import evaluate,report
                    result=evaluate(root,data,frozen,[local(root,x) for x in args.checkpoints],controls)
                    write_json(out/'result.json',sealed(result))
                    write_json(out/'per-video.json',sealed([r for q in result['seeds'].values() for r in q['records']]))
                    write_json(out/'paired-errors.json',sealed([r for q in result['seeds'].values() for r in q['paired_records']]))
                    (out/'REPORT.md').write_text(report(result),encoding='utf-8')
            for link in frozen['sources']['sources']:
                check_identity(root,link)
        if args.command=='train':
            from .runtime import train
            train(root,data,frozen,args.seed,args.fold,out,user_run=True,
                  resume=local(root,args.resume) if args.resume else None)
            result=dict(status='USER_TRAINING_COMPLETE',seed=args.seed,fold=args.fold,epoch_index=499,aggregate_pending=True)
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-start,environment=environment(),result=result))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out),status=result.get('status'))))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(status='BLOCKED_OR_INCOMPLETE',command=command,
            detail=type(exc).__name__+': '+str(exc),diagnostics=getattr(exc,'diagnostics',None),
            expected_clips=33,comparison_complete=False))
        print(json.dumps(dict(status='BLOCKED_OR_INCOMPLETE',detail=str(exc))),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
