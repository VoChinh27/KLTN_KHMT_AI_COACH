"""User-operated serial training with immutable attempt journals and fail-closed resume."""
import argparse
import json
from pathlib import Path
import re
import uuid
import os
from contextlib import contextmanager
from ..inner_cv.manifest import sealed,unseal
from ..models import canonical_hash
from ..readiness import write_json,file_info
from ..research.data import read,local,identity,check_identity
from .contract import VERSION,SEEDS
from . import manifest,runtime,evaluation
from .safety import no_learning


def append(root,name,payload):
    write_json(root/(name+'-'+uuid.uuid4().hex+'.json'),sealed(payload))


@contextmanager
def exclusive_run(run):
    """OS lock releases on process death; keep the lock file as an operational artifact."""
    stream=(run/'run.lock').open('a+b')
    try:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)
    finally:stream.close()


def discover(root,run,man,seed,fold,loader):
    """Read all attempts; latest corrupt artifact stops, never search backwards."""
    records=sorted([unseal(read(p)) for p in (run/'attempts').glob('*.json')],key=lambda r:r['sequence'])
    if len({r['sequence'] for r in records})!=len(records):raise ValueError('Duplicate attempt sequence')
    for r in records:
        if type(r['sequence']) is not int or r['sequence']<1:raise ValueError('Invalid attempt sequence')
        expected_identity=manifest.fold_identity(manifest.select(man,r['seed']),r['fold'])
        expected_path=run/f"seed-{r['seed']}"/f"fold-{r['fold']}"/f"attempt-{r['sequence']:04d}"
        if r['identity']!=expected_identity or local(root,r['output'])!=expected_path:
            raise ValueError('Unknown attempt condition/path')
    known={local(root,r['output']) for r in records}
    actual=set(run.glob('seed-*/fold-*/attempt-*'))
    if not actual<=known:raise ValueError('Untracked attempt/ambiguous endpoint directory')
    selected=manifest.select(man,seed);expected=manifest.fold_identity(selected,fold)
    matching=[r for r in records if r['seed']==seed and r['fold']==fold]
    latest=None;endpoint=None
    for r in matching:
        dest=local(root,r['output'])
        if not dest.is_relative_to(run):raise ValueError('Attempt outside runRoot')
        if r['identity']!=expected:raise ValueError('Attempt identity mismatch')
        if endpoint is not None:raise ValueError('Duplicate/continued completed endpoint')
        parent=r['resume']
        if (parent is None)!=(latest is None):raise ValueError('Broken resume chain')
        if parent is not None:
            if parent!=identity(root,latest):raise ValueError('Resume parent not exact previous latest checkpoint')
            check_identity(root,parent)
        if not dest.exists():continue  # planned but not started; next attempt records same parent
        runfile=dest/'run.json'
        if not runfile.exists():
            if list(dest.iterdir()):raise ValueError('Attempt artifacts without run identity')
            continue
        rr=read(runfile)
        if rr['identity']!=expected or rr['config']!=selected['config']:raise ValueError('Run condition/seed/fold/config differs')
        if rr['initial_fresh_state_sha256']!=man['sources']['controls'][str(seed)]['initializations'][str(fold)]:
            raise ValueError('Run fresh initialization differs from control')
        if parent is None:
            if rr['resume_checkpoint'] is not None or rr['resume_identity'] is not None or rr['first_epoch']!=0:
                raise ValueError('Fresh run has resume state')
        elif (local(root,rr['resume_checkpoint'])!=local(root,parent['path'])
              or rr['resume_identity']!={k:parent[k] for k in ('sha256','size_bytes')}):
            raise ValueError('Run resume provenance differs')
        candidates=[]
        for p in dest.glob('epoch-*'):
            match=re.fullmatch(r'epoch-(\d{3})\.(pt|pt\.sha256\.json|json)',p.name)
            if not match:raise ValueError('Unrecognized/incomplete checkpoint artifact: '+str(p))
            candidates.append(int(match[1]))
        if not candidates:
            if (dest/'complete.json').exists():raise ValueError('Complete without checkpoint')
            continue
        newest=max(candidates)
        if newest not in range(500):raise ValueError('Endpoint outside fixed budget')
        path=dest/f'epoch-{newest:03d}.pt'
        # This one must load. Do not fall back if it is truncated/missing/hash-invalid.
        cp=loader(path,selected,fold)
        first=rr['first_epoch']
        if parent is not None:
            prev=loader(local(root,parent['path']),selected,fold)
            if first!=prev['epoch']+1:raise ValueError('Resume epoch discontinuity')
        if cp['epoch']!=newest or newest<first:raise ValueError('Checkpoint filename/epoch mismatch')
        # The validated latest checkpoint embeds the full epoch history. Never
        # substitute an earlier checkpoint, even if that earlier file is valid.
        latest=path
        if newest==499:
            if not (dest/'complete.json').exists():raise ValueError('Endpoint exists without completion receipt; no automatic promotion')
            complete=read(dest/'complete.json')
            if complete['identity']!=expected or complete['checkpoint']!=file_info(path) or complete['final_checkpoint']!=path.name or complete['endpoint']!=man['endpoint']:
                raise ValueError('Completion receipt mismatch')
            endpoint=path
        elif (dest/'complete.json').exists():raise ValueError('Premature completion receipt')
    return dict(latest=latest,endpoint=endpoint,sequence=max([r['sequence'] for r in records],default=0)+1)


def execute(root,run,man,data,controls,train_fn,load_fn,evaluate_fn):
    """Dependency seams permit orchestration tests with non-learning mocks only."""
    endpoints=[]
    for seed in SEEDS:
        for fold in range(3):
            loader=lambda p,m,f:load_fn(seed,p,m,f)
            state=discover(root,run,man,seed,fold,loader)
            if state['endpoint'] is None:
                seq=state['sequence'];dest=run/f'seed-{seed}'/f'fold-{fold}'/f'attempt-{seq:04d}'
                if dest.exists():raise ValueError('Attempt output already exists')
                event=dict(sequence=seq,seed=seed,fold=fold,output=dest.relative_to(root).as_posix(),
                    identity=manifest.fold_identity(manifest.select(man,seed),fold),
                    resume=identity(root,state['latest']) if state['latest'] else None)
                write_json(run/'attempts'/f'{seq:04d}.json',sealed(event))
                print(json.dumps(dict(training='USER_RUN',seed=seed,fold=fold,output=str(dest),resume=str(state['latest']))),flush=True)
                train_fn(seed,fold,dest,state['latest'])
                state=discover(root,run,man,seed,fold,loader)
                if state['endpoint'] is None:raise ValueError('Trainer returned without verified endpoint499')
            endpoints.append(state['endpoint'])
    if len(set(endpoints))!=6:raise ValueError('Six unique endpoints required')
    complete_files=list(run.glob('completed-*.json'))
    if len(complete_files)>1:raise ValueError('Ambiguous completed comparisons')
    if complete_files:
        done=unseal(read(complete_files[0]))
        if done['checkpoints']!=[identity(root,p) for p in endpoints]:raise ValueError('Completed comparison endpoint lineage differs')
        for link in done['artifacts']:check_identity(root,link)
        return local(root,done['comparison'])
    comparison=run/('comparison-'+uuid.uuid4().hex);comparison.mkdir(exist_ok=False)
    append(run,'aggregate-started',dict(comparison=comparison.relative_to(root).as_posix(),checkpoints=[identity(root,p) for p in endpoints]))
    with no_learning():result=evaluate_fn(endpoints)
    if result['status']!='COMPLETE_DEVELOPMENT_ONLY' or result['manifest_sha256']!=canonical_hash(man):raise ValueError('Comparison incomplete or wrong manifest')
    write_json(comparison/'result.json',sealed(result))
    write_json(comparison/'paired-errors.json',sealed([r for q in result['seeds'].values() for r in q['paired_records']]))
    write_json(comparison/'per-video.json',sealed([r for q in result['seeds'].values() for r in q['records']]))
    (comparison/'REPORT.md').write_text(evaluation.report(result),encoding='utf-8')
    if unseal(read(comparison/'result.json'))!=result:raise ValueError('Comparison serialization mismatch')
    for link in man['sources']['sources']:check_identity(root,link)
    artifacts=[identity(root,p) for p in sorted(comparison.iterdir()) if p.is_file()]
    write_json(comparison/'receipt.json',dict(status='VERIFIED',manifest_sha256=canonical_hash(man),checkpoints=[identity(root,p) for p in endpoints],artifacts=artifacts))
    artifacts.append(identity(root,comparison/'receipt.json'))
    append(run,'completed',dict(comparison=comparison.relative_to(root).as_posix(),checkpoints=[identity(root,p) for p in endpoints],artifacts=artifacts))
    return comparison


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--manifest',required=True)
    p.add_argument('--run-root');p.add_argument('--user-run',action='store_true');args=p.parse_args(argv)
    if not args.user_run:p.error('Only user may invoke serial training: --user-run required')
    root=args.root.resolve();manifest_path=local(root,args.manifest)
    # Persist runRoot before verification/training so any interrupted invocation is discoverable.
    if args.run_root:
        run=local(root,args.run_root)
        receipt=unseal(read(run/'run-root.json'))
        if receipt['version']!=VERSION or receipt['manifest']!=identity(root,manifest_path) or receipt['run_root']!=run.relative_to(root).as_posix():raise ValueError('Wrong runRoot/manifest')
    else:
        run=local(root,'outputs/squat_count_fixed_origin/user-'+uuid.uuid4().hex)
        run.mkdir(parents=True,exist_ok=False);(run/'attempts').mkdir()
        write_json(run/'run-root.json',sealed(dict(version=VERSION,manifest=identity(root,manifest_path),run_root=run.relative_to(root).as_posix())))
    print('runRoot='+str(run),flush=True)
    try:
        with exclusive_run(run):
            with no_learning():
                man,data,controls=manifest.load(root,manifest_path)
                append(run,'verified',dict(manifest_sha256=canonical_hash(man),source_files=len(man['sources']['sources'])))
            def load(seed,path,selected,fold):return runtime.engine(man,seed)[0].load_checkpoint(path,selected,fold)
            comparison=execute(root,run,man,data,controls,
                lambda s,f,o,r:runtime.train(root,data,man,s,f,o,user_run=True,resume=r),load,
                lambda paths:evaluation.evaluate(root,data,man,paths,controls))
        print('runRoot='+str(run)+'\ncomparisonOut='+str(comparison)+'\nREPORT='+str(comparison/'REPORT.md'),flush=True)
        return 0
    except Exception as exc:
        append(run,'failure',dict(type=type(exc).__name__,detail=str(exc),run_root=str(run),training_may_have_started=True))
        raise


if __name__=='__main__':raise SystemExit(main())
