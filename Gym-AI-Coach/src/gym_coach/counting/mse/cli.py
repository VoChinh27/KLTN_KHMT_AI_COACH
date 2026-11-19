"""User-run MSE training; all other commands are guarded against learning."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from ..inner_cv.manifest import sealed
from ..models import canonical_hash
from ..readiness import file_info,write_json
from ..research.cli import environment
from ..research.data import local
from . import manifest
from .safety import no_learning


def runtime_environment():
    value=environment()
    # The legacy helper includes preparation-only counters; do not mislabel a user run.
    return {k:v for k,v in value.items() if k not in ('training_runs','training_runtime')}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','forward-only','train','aggregate'))
    p.add_argument('--root',type=Path,default=Path.cwd())
    for name in ('manifest','original-manifest','gru-run','diagnosis-dir','resume','output-dir'):
        p.add_argument('--'+name,type=Path,required=name=='output-dir')
    p.add_argument('--fold',type=int,choices=(0,1,2));p.add_argument('--user-run',action='store_true')
    p.add_argument('--checkpoints',type=Path,nargs='+')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command=='train' and (not args.user_run or args.fold is None):
        p.error('Training reserved for user: --user-run and --fold required')
    if args.command!='train' and (args.user_run or args.resume or args.fold is not None):
        p.error('Training options apply only to train')
    prep=(args.original_manifest,args.gru_run,args.diagnosis_dir)
    if args.command=='prepare':
        if not all(prep) or args.manifest:p.error('prepare requires --original-manifest --gru-run --diagnosis-dir')
    elif not args.manifest or any(prep):p.error('Non-prepare commands require only --manifest')
    if (args.command=='aggregate')!=bool(args.checkpoints):p.error('--checkpoints required only for aggregate')
    out=local(args.root,args.output_dir)
    if out.exists():p.error('Output directory already exists; never overwrite, including on resume')
    start=time.perf_counter();env=runtime_environment()
    command=[sys.executable,'-m',__package__+'.cli',*(argv if argv is not None else sys.argv[1:])]
    try:
        with no_learning():
            if args.command=='prepare':
                base,data,links,controls=manifest.verify_controls(args.root,*prep)
                frozen=manifest.build(base,links);out.mkdir(parents=True,exist_ok=False)
                write_json(out/'manifest.json',sealed(frozen));write_json(out/'config.json',sealed(frozen['config']))
                result=dict(status='PREPARED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),
                    folds=frozen['folds'],endpoint=manifest.ENDPOINT,loss=frozen['loss'],training_steps=0,baseline_fits=0)
            else:
                frozen,data,controls=manifest.load(args.root,args.manifest)
                if args.command!='train':out.mkdir(parents=True,exist_ok=False)
                if args.command=='verify':
                    result=dict(status='VERIFIED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),
                        clips=33,groups=12,folds=frozen['folds'],endpoint=manifest.ENDPOINT,
                        loss=frozen['loss'],frozen_code_dataset_controls_verified=True,training_steps=0,baseline_fits=0)
                elif args.command=='forward-only':
                    from .runtime import forward_only
                    result=forward_only(args.root,data,frozen)
                elif args.command=='aggregate':
                    from .evaluation import evaluate,report_markdown
                    result=evaluate(args.root,data,frozen,[local(args.root,x) for x in args.checkpoints],controls)
                    write_json(out/'comparison.json',sealed(result))
                    write_json(out/'per-video-errors.json',sealed(result['records']))
                    write_json(out/'paired-errors.json',sealed(result['paired_records']))
                    (out/'REPORT.md').write_text(report_markdown(result),encoding='utf-8')
        if args.command=='train':
            from .runtime import train
            train(args.root,data,frozen,args.fold,out,user_run=True,
                resume=local(args.root,args.resume) if args.resume else None)
            result=dict(status='USER_TRAINING_COMPLETE',fold=args.fold,endpoint=manifest.ENDPOINT,comparison_pending=True)
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-start,environment=env,result=result))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out),status=result.get('status',result.get('kind')))))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(status='INCOMPLETE_OR_INVALID',command=command,detail=type(exc).__name__+': '+str(exc),
            comparison_complete=False,expected_clips=33,endpoint=manifest.ENDPOINT))
        print(json.dumps(dict(status='INCOMPLETE_OR_INVALID',detail=str(exc))),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
