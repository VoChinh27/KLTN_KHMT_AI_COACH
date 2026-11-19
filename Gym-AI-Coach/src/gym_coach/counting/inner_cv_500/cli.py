"""User-run CV500 training; preparation and fixed endpoint evaluation prohibit fitting."""
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
    for name in ('manifest','original-manifest','gru-run','control-receipt','tiny-report-dir','tiny-review','resume','output-dir'):
        p.add_argument('--'+name,type=Path,required=name=='output-dir')
    p.add_argument('--fold',type=int,choices=(0,1,2));p.add_argument('--user-run',action='store_true')
    p.add_argument('--checkpoints-50',nargs='+',type=Path);p.add_argument('--checkpoints-500',nargs='+',type=Path)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    prep=(args.original_manifest,args.gru_run,args.control_receipt,args.tiny_report_dir,args.tiny_review)
    if args.command=='train' and (not args.user_run or args.fold is None):p.error('Only user may train: --user-run --fold required')
    if args.command!='train' and (args.user_run or args.fold is not None or args.resume):p.error('Train options only for train')
    if args.command=='prepare':
        if not all(prep) or args.manifest:p.error('prepare requires all five source paths')
    elif not args.manifest or any(prep):p.error('Non-prepare commands require only --manifest')
    if args.command=='aggregate':
        if not args.checkpoints_50 or not args.checkpoints_500:p.error('Require both --checkpoints-50 and --checkpoints-500')
    elif args.checkpoints_50 or args.checkpoints_500:p.error('Checkpoint lists only for aggregate')
    out=local(args.root,args.output_dir)
    if out.exists():p.error('Output directory exists; never overwrite, including resume')
    start=time.perf_counter();command=[sys.executable,'-m',__package__+'.cli',*(argv if argv is not None else sys.argv[1:])]
    env={k:v for k,v in environment().items() if k not in ('training_runs','training_runtime')}
    try:
        with no_learning():
            if args.command=='prepare':
                frozen,data,controls=manifest.prepare_inputs(args.root,*prep);out.mkdir(parents=True,exist_ok=False)
                write_json(out/'manifest.json',sealed(frozen));write_json(out/'config.json',sealed(frozen['config']))
                write_json(out/'tiny1000-verification.json',sealed(frozen['tiny1000_evidence']))
                write_json(out/'saved-controls.json',sealed(controls))
                result=dict(status='PREPARED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),folds=frozen['folds'],
                    expected_steps=frozen['expected_steps'],training_steps=0,baseline_fits=0)
            else:
                frozen,data,controls=manifest.load(args.root,args.manifest)
                if args.command!='train':out.mkdir(parents=True,exist_ok=False)
                if args.command=='verify':
                    result=dict(status='VERIFIED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),clips=33,groups=12,
                        folds=frozen['folds'],expected_steps=frozen['expected_steps'],old_verifiers_compatible=True,
                        training_steps=0,baseline_fits=0)
                elif args.command=='forward-only':
                    from .runtime import forward_only
                    result=forward_only(args.root,data,frozen)
                elif args.command=='aggregate':
                    from .evaluation import evaluate,report_markdown
                    result=evaluate(args.root,data,frozen,[local(args.root,x) for x in args.checkpoints_50],
                        [local(args.root,x) for x in args.checkpoints_500],controls)
                    write_json(out/'result.json',sealed(result));write_json(out/'reproducibility.json',sealed(result['reproducibility']))
                    records=[dict(r,epoch_index=int(e)) for e,q in result['endpoints'].items() for r in q['records']]
                    pairs=[dict(r,epoch_index=int(e)) for e,q in result['endpoints'].items() for r in q['paired_records']]
                    write_json(out/'per-video.json',sealed(records));write_json(out/'paired-errors.json',sealed(pairs))
                    (out/'REPORT.md').write_text(report_markdown(result),encoding='utf-8')
        if args.command=='train':
            from .runtime import train
            train(args.root,data,frozen,args.fold,out,user_run=True,resume=local(args.root,args.resume) if args.resume else None)
            result=dict(status='USER_TRAINING_COMPLETE',fold=args.fold,epoch_index=499,aggregate_pending=True)
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-start,environment=env,result=result))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out),status=result.get('status',result.get('kind')))))
        return 3 if result.get('status')=='REPRODUCTION_MISMATCH_REQUIRES_INVESTIGATION' else 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(status='INCOMPLETE_OR_INVALID',command=command,detail=type(exc).__name__+': '+str(exc),
            expected_clips=33,expected_folds=3,comparison_complete=False))
        print(json.dumps(dict(status='INCOMPLETE_OR_INVALID',detail=str(exc))),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
