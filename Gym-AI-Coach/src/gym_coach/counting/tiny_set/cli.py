"""Prepare/verify/forward/report are guarded; only the human invokes train --user-run."""
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
from ..research.data import local,identity
from . import manifest
from .safety import no_learning


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','forward-only','train','report'))
    p.add_argument('--root',type=Path,default=Path.cwd())
    for name in ('manifest','original-manifest','resume','checkpoint','output-dir'):
        p.add_argument('--'+name,type=Path,required=name=='output-dir')
    p.add_argument('--user-run',action='store_true')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command=='train' and not args.user_run:p.error('Training reserved for user: --user-run required')
    if args.command!='train' and (args.user_run or args.resume):p.error('Training options apply only to train')
    if args.command=='prepare':
        if not args.original_manifest or args.manifest:p.error('prepare requires only --original-manifest')
    elif not args.manifest or args.original_manifest:p.error('Non-prepare commands require only --manifest')
    if (args.command=='report')!=bool(args.checkpoint):p.error('--checkpoint required only for report')
    out=local(args.root,args.output_dir)
    if out.exists():p.error('Output directory already exists; never overwrite, including resume')
    command=[sys.executable,'-m',__package__+'.cli',*(argv if argv is not None else sys.argv[1:])]
    started=time.perf_counter()
    env={k:v for k,v in environment().items() if k not in ('training_runs','training_runtime')}
    try:
        with no_learning():
            if args.command=='prepare':
                frozen,data=manifest.prepare_inputs(args.root,args.original_manifest)
                out.mkdir(parents=True,exist_ok=False)
                write_json(out/'manifest.json',sealed(frozen));write_json(out/'config.json',sealed(frozen['config']))
                write_json(out/'input-audit.json',sealed(frozen['input_audit']))
                result=dict(status='PREPARED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),
                    selected_ids=frozen['training_ids'],criterion=frozen['criterion'],training_steps=0,baseline_fits=0)
            else:
                frozen,data=manifest.load(args.root,args.manifest)
                if args.command!='train':out.mkdir(parents=True,exist_ok=False)
                if args.command=='verify':
                    result=dict(status='VERIFIED_NOT_TRAINED',manifest_sha256=canonical_hash(frozen),clips=6,
                        groups=len({r['source']['group_id'] for r in frozen['rows']}),parent_fold=1,
                        all_sources_cache_labels_core_code_verified=True,selected_ids=frozen['training_ids'],
                        training_steps=0,baseline_fits=0)
                elif args.command=='forward-only':
                    from .runtime import forward_only
                    result=forward_only(args.root,data,frozen)
                elif args.command=='report':
                    from .evaluation import evaluate,report_markdown
                    checkpoint=local(args.root,args.checkpoint);result=evaluate(frozen,checkpoint)
                    result['checkpoint']=identity(args.root,checkpoint)
                    write_json(out/'result.json',sealed(result));write_json(out/'history.json',sealed(result['history']))
                    rows=[dict(r,epoch=h['epoch']) for h in result['history'] for r in h['evaluation']['records']]
                    write_json(out/'per-video.json',sealed(rows))
                    (out/'REPORT.md').write_text(report_markdown(result),encoding='utf-8')
        if args.command=='train':
            from .runtime import train
            train(args.root,data,frozen,out,user_run=True,resume=local(args.root,args.resume) if args.resume else None)
            result=dict(status='USER_TRAINING_COMPLETE',final_epoch_index=499,report_pending=True)
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-started,environment=env,result=result))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps(dict(output_dir=str(out),status=result.get('status',result.get('kind')))))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(status='INCOMPLETE_OR_INVALID',command=command,
            detail=type(exc).__name__+': '+str(exc),expected_clips=6,report_complete=False))
        print(json.dumps(dict(status='INCOMPLETE_OR_INVALID',detail=str(exc))),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
