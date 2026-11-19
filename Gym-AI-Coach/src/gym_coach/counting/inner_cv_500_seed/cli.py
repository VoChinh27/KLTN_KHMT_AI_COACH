"""Prepare/verify without learning; training requires an explicit user-run command."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from ..inner_cv.manifest import sealed
from ..models import canonical_hash
from ..readiness import write_json,file_info
from ..research.cli import environment
from ..research.data import local
from . import manifest
from .safety import no_learning


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','forward-only','train','aggregate'))
    p.add_argument('--root',type=Path,default=Path.cwd())
    for name in ('manifest','parent-manifest','comparison','diagnosis-report','review','resume','output-dir'):
        p.add_argument('--'+name,type=Path,required=name=='output-dir')
    p.add_argument('--fold',type=int,choices=(0,1,2));p.add_argument('--user-run',action='store_true')
    p.add_argument('--checkpoints-50',nargs=3,type=Path);p.add_argument('--checkpoints-500',nargs=3,type=Path)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    prep=(args.parent_manifest,args.comparison,args.diagnosis_report,args.review)
    if args.command=='train' and (not args.user_run or args.fold is None):p.error('Only user may train: --user-run --fold required')
    if args.command!='train' and (args.user_run or args.fold is not None or args.resume):p.error('Train options only for train')
    if args.command=='prepare':
        if not all(prep) or args.manifest:p.error('prepare requires four source paths')
    elif not args.manifest or any(prep):p.error('Require only --manifest outside prepare')
    if args.command=='aggregate':
        if not args.checkpoints_50 or not args.checkpoints_500:p.error('Require all six fixed-endpoint checkpoints')
    elif args.checkpoints_50 or args.checkpoints_500:p.error('Checkpoint lists only for aggregate')
    root=args.root.resolve();out=local(root,args.output_dir)
    if out.exists():p.error('Output directory exists; never overwrite, including resume')
    start=time.perf_counter();command=[sys.executable,'-m',__package__+'.cli',*(argv if argv is not None else sys.argv[1:])]
    try:
        with no_learning():
            if args.command=='prepare':
                frozen,data,controls=manifest.prepare_inputs(root,*prep)
                out.mkdir(parents=True,exist_ok=False)
                write_json(out/'manifest.json',sealed(frozen));write_json(out/'config.json',sealed(frozen['config']))
                write_json(out/'saved-controls.json',sealed(controls));write_json(out/'source-verification.json',sealed(frozen['evidence']))
                result=dict(status='PREPARED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),seed=20261027,
                    folds=frozen['folds'],expected_steps=frozen['expected_steps'],training_steps=0,baseline_fits=0)
            else:
                frozen,data,controls=manifest.load(root,args.manifest)
                if args.command!='train':out.mkdir(parents=True,exist_ok=False)
                if args.command=='verify':
                    result=dict(status='VERIFIED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),seed=20261027,
                        clips=33,groups=12,folds=frozen['folds'],training_steps=0,baseline_fits=0,
                        source_files_verified=frozen['evidence']['source_files']['files'])
                elif args.command=='forward-only':
                    from .runtime import forward_only
                    result=forward_only(root,data,frozen);write_json(out/'forward-only.json',sealed(result))
                elif args.command=='aggregate':
                    from .evaluation import evaluate,report_markdown
                    result=evaluate(root,data,frozen,[local(root,x) for x in args.checkpoints_50],
                        [local(root,x) for x in args.checkpoints_500],controls)
                    write_json(out/'result.json',sealed(result))
                    write_json(out/'per-video.json',sealed([dict(r,epoch_index=int(e)) for e,q in result['endpoints'].items() for r in q['records']]))
                    write_json(out/'paired-errors.json',sealed([dict(r,epoch_index=int(e)) for e,q in result['endpoints'].items() for r in q['paired_records']]))
                    (out/'REPORT.md').write_text(report_markdown(result),encoding='utf-8')
        if args.command=='train':
            from .runtime import train
            train(root,data,frozen,args.fold,out,user_run=True,resume=local(root,args.resume) if args.resume else None)
            result=dict(status='USER_TRAINING_COMPLETE',fold=args.fold,epoch_index=499,aggregate_pending=True)
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-start,environment=environment(),result=result))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out),status=result.get('status',result.get('kind')))))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(status='INCOMPLETE_OR_INVALID',command=command,detail=type(exc).__name__+': '+str(exc),
            expected_clips=33,expected_folds=3,comparison_complete=False))
        print(json.dumps(dict(status='INCOMPLETE_OR_INVALID',detail=str(exc))),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
