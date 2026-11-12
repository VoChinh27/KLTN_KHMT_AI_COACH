"""Opt-in research CLI. Only baseline/train commands with --user-run learn."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

from ..readiness import write_json, file_info
from .contract import Config
from .data import prepare, load_dataset, CountDataset


def environment():
    from importlib.metadata import version
    import torch
    if version('torch')!='2.14.0' or sys.version_info[:2]!=(3,11):
        raise RuntimeError('Frozen experiment requires tested Python3.11 and torch2.14.0; no automatic installation')
    return dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        numpy=version('numpy'),pydantic=version('pydantic'),device='cpu',cuda_available=torch.cuda.is_available(),
        threads=Config().torch_threads,training_runtime='NOT_MEASURED',training_runs=0)


def forward_only(root,data):
    import torch
    from .network import CountGRU, setup, collate
    setup();model=CountGRU();model.eval()
    dataset=CountDataset(root,data,'train',supervised=False)
    if not len(dataset):
        raise ValueError('No admitted train input for forward check')
    samples=[dataset[i] for i in range(min(3,len(dataset)))]
    before=[p.detach().clone() for p in model.parameters()]
    started=time.perf_counter()
    with torch.inference_mode():
        batch=collate(samples);output=model(batch)
    if (not torch.isfinite(output).all() or output.requires_grad
            or any(p.grad is not None or not torch.equal(p,b) for p,b in zip(model.parameters(),before))):
        raise ValueError('Forward-only invariant failed')
    return dict(kind='untrained_forward_only_not_performance',sample_ids=batch['video_ids'],
        tensor_shape=list(batch['x'].shape),output_shape=list(output.shape),finite=True,
        parameter_count=sum(p.numel() for p in model.parameters()),parameters_unchanged=True,
        tensor_bytes=batch['x'].numel()*batch['x'].element_size(),elapsed_seconds=time.perf_counter()-started,
        training_steps=0,baseline_fits=0,checkpoint_created=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('environment','prepare','verify','forward','baseline','train','evaluate'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--user-run',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--role',choices=('validation','development_test'),default='validation')
    parser.add_argument('--final-evaluation',action='store_true')
    args=parser.parse_args(argv)
    if args.command in ('train','baseline') and not args.user_run:
        parser.error('Learning is reserved for the user: explicit --user-run required')
    if args.command not in ('environment','verify') and args.output_dir is None:
        parser.error('--output-dir must be a NEW directory')
    start=time.perf_counter();result=None
    if args.command=='environment':
        result=environment()
    elif args.command=='prepare':
        if not args.plan:parser.error('--plan required')
        result=prepare(args.root,args.plan,args.output_dir)['summary']
    else:
        if not args.dataset:parser.error('--dataset required')
        data=load_dataset(args.root,args.dataset)
        if args.command=='verify':
            result=dict(dataset=file_info(args.dataset),summary=data['summary'],environment=environment())
        elif args.command=='forward':
            environment();result=forward_only(args.root,data)
        else:
            environment()
            from .experiment import train, run_baselines, evaluate_saved
            if args.command=='train':
                train(args.root,data,args.dataset,args.output_dir,user_run=args.user_run,resume=args.resume)
            elif args.command=='baseline':
                run_baselines(args.root,data,args.dataset,args.output_dir,user_run=args.user_run)
            else:
                evaluate_saved(args.root,data,args.dataset,args.output_dir,args.role,checkpoint=args.checkpoint,
                    baseline=args.baseline,final_evaluation=args.final_evaluation)
    if result is not None:
        receipt=dict(command=[sys.executable,'-m',__package__+'.cli',*(argv if argv is not None else sys.argv[1:])],
                     result=result,seconds=time.perf_counter()-start)
        if args.output_dir and args.command!='prepare':
            args.output_dir.mkdir(parents=True,exist_ok=False)
            write_json(args.output_dir/'receipt.json',receipt)
        elif args.command=='prepare':
            write_json(args.output_dir/'receipt.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
