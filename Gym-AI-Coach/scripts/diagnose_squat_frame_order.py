"""Inference-only perturbation on the two completed GRU500 runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from gym_coach.counting import frame_order_sensitivity as experiment
from gym_coach.counting.frame_order_sensitivity import require, SEEDS, REPEATS
from gym_coach.counting.inner_cv import manifest as cv, runtime as shared
from gym_coach.counting.inner_cv.oof import pair_identity
from gym_coach.counting.inner_cv_500 import manifest as old_manifest, runtime as old_runtime
from gym_coach.counting.inner_cv_500.contract import Config as OldConfig
from gym_coach.counting.inner_cv_500_seed import manifest as new_manifest, runtime as new_runtime
from gym_coach.counting.inner_cv_500_seed.contract import Config as NewConfig
from gym_coach.counting.inner_cv_500_seed.safety import no_learning
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json, file_info
from gym_coach.counting.research.contract import Config as CacheConfig
from gym_coach.counting.research.data import identity, local, read
from gym_coach.counting.research.experiment import experiment_identity
from gym_coach.counting.research.network import setup, CountGRU, collate

COMPARISONS = {
    20261026: 'outputs/squat_count_inner_cv_500/user-298d9a0a7e08444cb423d973c2a9ca05/comparison-d74d4c9bbe90404b9f2f0bd8549137b5',
    20261027: 'outputs/squat_count_inner_cv_500_seed/user-cbac509784bc438ca22746fb802f7659/comparison-765a0e5322df4f96b65d9120285fc092',
}


class Ledger:
    def __init__(self, root):
        self.root=root
        self.files={}

    def take(self,path,expected=None):
        p=local(self.root,path)
        require(not {'KLTN_AI_COACH','_archive'}.intersection(p.parts),'Out-of-scope source')
        info=identity(self.root,p)
        if expected:
            require(all(info[k]==expected[k] for k in ('sha256','size_bytes')),f'Hash mismatch: {p}')
        require(self.files.setdefault(info['path'],info)==info,f'Source changed: {p}')
        return p

    def link(self,item):
        return self.take(item['path'],item)

    def preserved(self):
        for item in list(self.files.values()):
            self.link(item)
        return sorted(self.files.values(),key=lambda i:i['path'])


def load_condition(root,seed,ledger):
    comp=local(root,COMPARISONS[seed])
    for name,info in read(ledger.take(comp/'inventory.json'))['files'].items():
        require((comp/name).resolve().is_relative_to(comp),'Escaping comparison path')
        ledger.take(comp/name,info)
    receipt=read(ledger.take(comp/'receipt.json'))
    result=cv.unseal(read(ledger.take(comp/'result.json')))
    require(receipt['result']==result and result['status']=='COMPLETE_NOT_VALIDATION','Comparison receipt/status mismatch')
    cmd=receipt['command']
    manifest_path=ledger.take(cmd[cmd.index('--manifest')+1])
    man=cv.unseal(read(manifest_path))
    require(canonical_hash(man)==result['manifest_sha256'] and man['config']['seed']==seed,'Wrong manifest/seed')
    config_type, runtime, code, method=(OldConfig,old_runtime,old_manifest,'cv500') if seed==SEEDS[0] else (
        NewConfig,new_runtime,new_manifest,'seed20261027')
    config=config_type.model_validate_json(json.dumps(man['config']))
    require(code.code_identity()==man['code'] and cv.code_identity()==man['cv_code'],'Frozen CV source mismatch')
    data_path=ledger.link(man['dataset'])
    require(experiment_identity(data_path,CacheConfig())==man['core_identity'],'Core source/environment mismatch')
    data=cv.unseal(read(data_path))
    rows=[r for r in data['rows'] if r['role']=='train' and r['supervised_eligible']]
    selected,_,folds=cv.partition(rows)
    require(selected==man['rows'] and folds==man['folds'],'Incompatible cohort/folds')
    held=experiment.held_map(folds)
    endpoint=result['endpoints']['499']
    require(endpoint['epoch_index']==499,'Wrong evaluation endpoint')
    saved_oof=[r for r in endpoint['records'] if r['method']==method and r['side']=='held_out']
    experiment.exact_ids(saved_oof,held)
    predictions={r['video_id']:r for r in saved_oof}
    cp_links=[s for s in result['checkpoint_sources'] if s['epoch']==499]
    require(len(cp_links)==3 and {s['fold'] for s in cp_links}=={0,1,2},'Missing/duplicate endpoint checkpoint')
    models={};hashes={};links={}
    for link in cp_links:
        f=link['fold']
        p=ledger.link(link)
        require(p.is_relative_to(comp.parent) and p.name=='epoch-499.pt','Wrong run/checkpoint')
        ledger.take(str(p)+'.sha256.json')
        complete=read(ledger.take(p.parent/'complete.json'))
        run=read(ledger.take(p.parent/'run.json'))
        cp=runtime.load_checkpoint(p,man,f)
        require(cp['epoch']==499 and cp['kind']=='user_trained' and cp['training_steps']==man['expected_steps'][f]
                and cp['identity']==run['identity']==complete['identity'] and complete['checkpoint']==file_info(p)
                and run['config']==man['config'],'Invalid fold checkpoint lineage')
        report=cv.unseal(cp['epoch_reports'][499])
        require(report==cv.unseal(read(ledger.take(p.parent/'epoch-499.json'))),'Endpoint log mismatch')
        for r in report['records']:
            v=r['video_id'];saved=predictions[v]
            require(held[v]==saved['fold']==f and saved['status']=='estimated' and saved['reason'] is None
                    and all(r[k]==saved[k] for k in ('video_id','source_sha256','target','row_sha256','label_record_hash','status','reason')),
                    'OOF source/target identity mismatch')
        setup(config);model=CountGRU(config);model.load_state_dict(cp['model'],strict=True);model.eval()
        hashes[f]=shared.state_hash(model.state_dict())
        require(hashes[f]==cp['model_sha256'] and not model.training,'Invalid model for inference')
        models[f]=model
        links[f]=dict(**link,model_sha256=hashes[f])
        # Keep the endpoint-log value independent from the comparison OOF value during replay.
        for r in report['records']:
            predictions[r['video_id']]=dict(predictions[r['video_id']],checkpoint_prediction=r['prediction'])
    return dict(manifest=man,manifest_link=identity(root,manifest_path),comparison=identity(root,comp/'result.json'),
        receipt=identity(root,comp/'receipt.json'),data={'rows':selected},held=held,models=models,links=links,hashes=hashes,
        oof=predictions,metrics=endpoint['results'][method]['aggregate'],fold0=endpoint['results'][method]['folds']['0'])


def sample_info(sample,row):
    mask=sample['complete'];values=sample['values'];source=row['source'];n=source['frame_count'];fps=source['fps']
    require(n==len(mask)==len(values) and sample['duration_seconds']==n/fps and any(mask),'Invalid timeline/support')
    for i,(v,m) in enumerate(zip(values,mask,strict=True)):
        require(len(v)==58 and all(math.isfinite(x) for x in v) and v[56:]==[i/fps,1/fps if i else 0.0]
                and all(v[j*7+3] in (0,1) for j in range(8)) and m==all(v[j*7+3] for j in range(8)),
                f'Cache timeline/raw58/mask mismatch at frame {i}')
    segments=list(experiment.runs(mask));sizes=[end-start for start,end in segments]
    return dict(video_id=sample['video_id'],group_id=source['group_id'],target=row['target'],source_sha256=source['sha256'],
        cache=row['input_artifact'],sample_sha256=canonical_hash(sample),frames=n,duration_seconds=n/fps,complete_frames=sum(mask),
        movable_frames=sum(s for s in sizes if s>=2),singletons=sizes.count(1),run_lengths=sizes)


def batch_hash(batch):
    return shared.state_hash({k:batch[k] for k in ('x','lengths','complete','duration')})


def infer(model,batch):
    require(not model.training and torch.is_inference_mode_enabled(),'Require eval and inference mode')
    value=float(model(batch)[0])
    require(math.isfinite(value) and value>=0,'Invalid count prediction')
    return value


def weight_receipts(conditions):
    receipts=[]
    for seed,c in conditions.items():
        for f,model in c['models'].items():
            after=shared.state_hash(model.state_dict())
            require(after==c['hashes'][f] and all(p.grad is None for p in model.parameters()), 'Weights or gradients changed')
            receipts.append(dict(seed=seed,fold=f,before=c['hashes'][f],after=after))
    return receipts


def input_plan(root,out,conditions,ledger):
    first,second=(conditions[s] for s in SEEDS)
    a,b=(c['manifest'] for c in (first,second))
    for k in ('rows','folds','dataset','core_identity','cv_code'):
        require(a[k]==b[k],f'Two conditions differ in {k}')
    require(b['parent_manifest_sha256']==canonical_hash(a) and b['sources']['parent_manifest']==first['manifest_link']
            and b['sources']['comparison']==first['comparison'] and b['parent500_code']==a['code']
            and dict(a['config'],seed=SEEDS[1])==b['config'],'Seed lineage mismatch')
    samples={};info={};plans={};datasets={}
    for f in a['folds']:
        index=f['fold'];ids=f['evaluation_ids']
        ds=shared.subset(root,first['data'],ids)
        ds2=shared.subset(root,second['data'],ids)
        datasets[index]=ds
        for i,row in enumerate(ds.rows):
            v=row['source']['video_id'];require(v==ids[i] and v not in samples,'Duplicate or reordered video')
            for link in [row['input_artifact'],*row['source_links'],*row['label_links']]:
                ledger.link(link)
            sample=ds[i];require(sample==ds2[i], 'Seed inputs differ')
            for c in conditions.values():
                require(all(c['oof'][v][k]==val for k,val in pair_identity(row).items()),'OOF row mismatch')
            samples[v]=sample;info[v]=dict(sample_info(sample,row),fold=index)
            plans[v]=[]
            for r in range(REPEATS):
                order=experiment.permutation(v,sample['complete'],r)
                plans[v].append(dict(repeat=r,source_for_destination=order,sha256=canonical_hash(order)))
    require(len(samples)==33 and len({r['group_id'] for r in info.values()})==12,'Wrong cohort')
    write_json(out/'permutations.json',cv.sealed(plans))
    write_json(out/'manifest.json',cv.sealed(dict(policy=experiment.policy(),policy_sha256=canonical_hash(experiment.policy()),
        permutations=file_info(out/'permutations.json'),inputs=info,
        conditions={str(s):{k:c[k] for k in ('manifest_link','comparison','receipt','links')} for s,c in conditions.items()},
        environment=dict(torch=torch.__version__,python=sys.version,device='cpu',threads=torch.get_num_threads()),
        code={p.name:file_info(p) for p in (Path(__file__),Path(experiment.__file__))})))
    return datasets,samples,info,plans


def reproduce(out,conditions,datasets,samples):
    originals={}
    for seed,c in conditions.items():
        for f,ds in datasets.items():
            model=c['models'][f];records=shared.predict(model,ds)
            for pred in records:
                v=pred['video_id'];saved=c['oof'][v];sample=samples[v]
                ident=experiment.apply_order(sample,list(range(len(sample['values']))),identity_control=True)
                require(ident==sample and batch_hash(collate([ident]))==batch_hash(collate([sample])),'Identity input drift')
                identity_prediction=infer(model,collate([ident]))
                value=pred['prediction']
                originals[v,seed]=dict(video_id=v,seed=seed,fold=f,checkpoint=c['links'][f],prediction=value,
                    saved_oof=saved['prediction'],saved_checkpoint=saved['checkpoint_prediction'],reason=pred['reason'],
                    identity_prediction=identity_prediction,delta_vs_oof=value-saved['prediction'] if value is not None else None,
                    exact=value==saved['prediction']==saved['checkpoint_prediction']==identity_prediction)
    bad=[r for r in originals.values() if not r['exact']]
    replay=dict(records=list(originals.values()),comparisons=len(originals),mismatches=bad,weights=weight_receipts(conditions))
    write_json(out/'original-replay.json',cv.sealed(replay))
    require(len(originals)==66 and not bad,'Original/identity replay mismatch. See original-replay.json. Stopped before perturbation.')
    return originals


def measure(conditions,samples,info,plans,originals):
    draws=[]
    for v,sample in samples.items():
        f=info[v]['fold'];original_batch=collate([sample]);moved=[];changed=[]
        for item in plans[v]:
            order=item['source_for_destination']
            variant=experiment.apply_order(sample,order)
            batch=collate([variant]);digest=batch_hash(batch)
            require(not batch['target_mask'].any() and not torch.is_grad_enabled(),'Perturbed target/grad enabled')
            for key in ('duration','complete','lengths'):
                require(torch.equal(batch[key],original_batch[key]),'Mask/time/length changed')
            x=original_batch['x'];y=batch['x']
            require(torch.equal(y[:,:,:56],x[:,order,:56]) and torch.equal(y[:,:,56:],x[:,:,56:]),'Pose block or timeline mismatch')
            moves=sum(j!=i for i,j in enumerate(order))
            changes=int((x[:,:,:56]!=y[:,:,:56]).any(dim=-1).sum())
            moved.append(moves);changed.append(changes)
            for seed,c in conditions.items():
                base=originals[v,seed]['prediction'];value=infer(c['models'][f],batch);delta=value-base
                draws.append(dict(video_id=v,seed=seed,fold=f,repeat=item['repeat'],original_prediction=base,
                    shuffled_prediction=value,delta=delta,abs_delta=abs(delta),positions_moved=moves,positions_changed=changes,
                    model_input_sha256=digest,permutation_sha256=item['sha256'],checkpoint_sha256=c['links'][f]['sha256']))
            require(batch_hash(batch)==digest,'Forward mutated shared input')
        require(canonical_hash(sample)==info[v]['sample_sha256'],'Source sample mutated')
        info[v].update(mean_positions_moved=statistics.mean(moved),mean_positions_changed=statistics.mean(changed),
            min_positions_moved=min(moved),max_positions_moved=max(moved),identity_orders=moved.count(0))
    return draws


def video_table(rows):
    lines=['|Video|Fold|Group|AI count gốc|Sai số gốc 026|Mean ∣Δ∣ 026|Mean Δ 026|Sai số gốc 027|Mean ∣Δ∣ 027|Mean Δ 027|027 − 026 (∣Δ∣)|',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        a,b=[r[str(s)] for s in SEEDS]
        nums=[a['original_signed_error'],a['mean_abs_delta'],a['mean_signed_delta'],
              b['original_signed_error'],b['mean_abs_delta'],b['mean_signed_delta'],r['paired_difference']]
        lines.append('|'+ '|'.join([r['video_id'],str(r['fold']),r['group_id'],str(r['target']),*[f'{v:.6f}' for v in nums]])+'|')
    return '\n'.join(lines)


def run_length_histogram(rows):
    # JSON object keys are strings. Convert before hashing, so sort order survives disk roundtrip.
    return dict(Counter(str(length) for row in rows for length in row['run_lengths']))


def rerun_guide():
    return """# Tái chạy phép thử trong PowerShell

Chạy từ project Gym. Lệnh tạo thư mục mới và dùng đúng hai run đã khóa trong script.

```powershell
Set-Location 'D:\\KLTN_K28_TKD'
$orderRun = 'outputs/squat_frame_order_sensitivity/run-' + [guid]::NewGuid().ToString('N')
& .venv/Scripts/python.exe scripts/diagnose_squat_frame_order.py --output-dir $orderRun
if ($LASTEXITCODE -ne 0) { throw "Inference failed. Inspect $orderRun/failure.json and original-replay.json." }
& .venv/Scripts/python.exe -m pytest tests/test_squat_frame_order_sensitivity.py tests/test_count_inner_cv_500.py tests/test_count_inner_cv_500_seed.py -q -ra "--junitxml=$orderRun/tests.xml" *> "$orderRun/tests.log"
$orderTestExit = $LASTEXITCODE
Get-Content "$orderRun/tests.log" -Tail 20
if ($orderTestExit -ne 0) { throw 'Regression failed' }
Write-Output "$orderRun/REPORT.md"
```

Manifest và hoán vị được ghi trước model forward. Nếu replay gốc hoặc hoán vị đồng nhất
không khớp chính xác thì script lưu sai khác và dừng trước phép đo độ nhạy.
Các lệnh trên chỉ suy luận và kiểm thử. Xem kết quả kiểm thử của lượt chạy ở
[tests.log](tests.log) và [tests.xml](tests.xml).
"""


def render(rows,summary,coverage,metrics):
    lines=['# GRU500 · Độ nhạy với thứ tự frame pose trong đoạn hợp lệ','',
        'Đơn vị là lần squat theo đầu ra liên tục. Mỗi video đóng góp một giá trị bằng trung bình của 20 độ thay đổi tuyệt đối. '
        'Kết quả tổng thể là trung bình đều của 33 video (12 nhóm). 026 và 027 chỉ seed 20261026 và 20261027.','',
        '|Tập|Video|Mean ∣Δ∣ 026|Median ∣Δ∣ 026¹|Mean Δ 026|Mean ∣Δ∣ 027|Median ∣Δ∣ 027¹|Mean Δ 027|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,q in summary.items():
        fields=[q[str(s)][k] for s in SEEDS for k in ('mean_abs_delta','median_abs_delta','mean_signed_delta')]
        lines.append('|'+ '|'.join([name,str(q['videos']),*[f'{v:.6f}' for v in fields]])+'|')
    q=summary['all']['paired_new_minus_old']
    lines+=['','¹ Median giữa các video của giá trị mean ∣Δ∣/video. Δ có dấu là prediction xáo trừ prediction gốc.',
        f"Cùng video, mean ∣Δ∣ của 027 trừ 026 trung bình {q['mean']:+.6f}, trung vị {q['median']:+.6f}. "
        f"Seed 027 cao hơn trên {q['higher']}/33 video, thấp hơn trên {q['lower']}/33, bằng nhau trên {q['equal']}/33.",'',
        '## Kiểm chứng trước khi xáo','',
        'Đã dùng đúng checkpoint index 499 của fold giữ lại từng video. Có 13/13/7 video ở ba fold. '
        'Cả 66 prediction gốc khớp chính xác OOF và log trong checkpoint. Cả 66 lượt qua phép đồng nhất khớp đúng cả input và prediction. '
        'Sáu model dùng eval(), inference_mode(), CPU và batch một clip như đường predict gốc. Hash weights trước/sau trùng cả sáu fold–seed. '
        '[Replay](original-replay.json) · [Manifest và nguồn](manifest.json) · [Receipt](receipt.json).','',
        '|Seed|Train MAE gốc đã lưu, mean 2 AE/video rồi mean 33|OOF MAE gốc trên 33 video|Fold 0 train / held MAE gốc|',
        '|---|---:|---:|---:|']
    for s in SEEDS:
        m=metrics[str(s)]
        lines.append(f"|{s}|{m['train_MAE']:.6f}|{m['oof_MAE']:.6f}|{m['fold0_train_MAE']:.6f} / {m['fold0_held_MAE']:.6f}|")
    lines+=['','## Phép xáo và mức tác động thực tế','',
        'Raw58 có tám khớp theo thứ tự left/right shoulder, hip, knee, ankle. Mỗi khớp có x, y, visibility, usable, observed, low_visibility, missing. '
        'Di chuyển đồng thời các cột 0–55. Elapsed (56), dt (57), duration, số frame, mask, các dòng thiếu và biên reset ở nguyên vị trí. '
        'Mỗi đoạn là một dãy complete=True tối đa. Đoạn một frame ở nguyên vị trí. Hai seed nhận cùng tensor đã xáo, có hash cho từng lượt. '
        'Đầu vào xáo mang target null. Toàn bộ hoán vị được lưu trước khi chạy model trên dữ liệu xáo. '
        'Quy tắc tất định và ánh xạ đích → nguồn được khóa trong [manifest](manifest.json) và [permutations](permutations.json).','',
        f"Trên {coverage['frames']} frame, {coverage['complete']} complete và {coverage['movable']} thuộc đoạn dài ≥2. "
        f"Có {coverage['singleton_runs']} đoạn chỉ một frame trên {coverage['videos_with_singletons']}/33 video. "
        f"{coverage['videos_without_movable_frames']}/33 video không có vị trí nào có thể di chuyển. "
        f"Độ dài đoạn lớn nhất của video ít support nhất theo tiêu chí này là {coverage['minimum_longest_run']} frame. "
        f"Độ dài các đoạn và số vị trí thực sự đổi của từng clip nằm trong [per-video.json](per-video.json).",'',
        f"Trong 20 lượt trên từng clip có {coverage['total_moved']} lượt vị trí đổi chỉ số nguồn, "
        f"{coverage['total_changed']} lượt vị trí thực sự đổi vector float32, trên {coverage['movable']*REPEATS} lượt vị trí có thể di chuyển. "
        f"Có {coverage['identity_orders']}/660 ánh xạ toàn video đồng nhất, đều được giữ. Không dùng ngưỡng để phân loại độ nhạy.",'',
        '## 13 video held-out của fold 0','',
        'Sai số gốc trong bảng là prediction trên video thật trừ AI count gốc. Mỗi mean ∣Δ∣ và mean Δ tính trên đúng 20 phép xáo.',
        '',video_table([r for r in rows if r['fold']==0]),'']
    for group in ('g11','g20'):
        rs=[r for r in rows if r['fold']==0 and r['group_id']==group]
        signed=[statistics.mean(r[str(s)]['original_signed_error'] for r in rs) for s in SEEDS]
        sensitivity=[summary[group][str(s)]['mean_abs_delta'] for s in SEEDS]
        delta=[summary[group][str(s)]['mean_signed_delta'] for s in SEEDS]
        lines.append(f"- **{group} ({len(rs)} video)**. Sai số có dấu trung bình trên video gốc {signed[0]:+.6f} / {signed[1]:+.6f} "
            f"(026 / 027). Mean ∣Δ∣ khi xáo {sensitivity[0]:.6f} / {sensitivity[1]:.6f}. Mean Δ {delta[0]:+.6f} / {delta[1]:+.6f}.")
    lines+=['','Các thay đổi đầu ra mô tả sự phụ thuộc vào trật tự pose trong những đoạn được thử, khi giữ duration và support. '
        'Phép xáo có thể tạo chuỗi ngoài phân bố và phá tính liên tục cục bộ của chuyển động. Clip xáo không có count tham chiếu. '
        'Không tính MAE/accuracy cho clip xáo. Kết quả không xác nhận GRU nhận đúng từng sự kiện squat hay giải thích nguyên nhân sai số g11/g20. '
        'Hai seed và 20 lượt xáo dùng lại cùng 33 video. OOF đã được dùng định hướng nghiên cứu, nhãn gốc là AI weak reference.','',
        'NOT_VALIDATED, official_eligibility=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS. '
        'Không training, fit baseline, chọn lại checkpoint hoặc sử dụng outer validation/development-test.','',
        '[Bảng 33 video](PER_VIDEO.md) · [Dữ liệu từng lượt](draws.jsonl) · [Kiểm thử và lệnh PowerShell](REPRODUCE.md).','']
    return '\n'.join(lines)


def execute(root,out):
    ledger=Ledger(root)
    conditions={s:load_condition(root,s,ledger) for s in SEEDS}
    datasets,samples,info,plans=input_plan(root,out,conditions,ledger)
    print('Locked 660 permutations and six index499 checkpoints',flush=True)
    originals=reproduce(out,conditions,datasets,samples)
    print('Exact original and identity reproduction: 66/66',flush=True)
    draws=measure(conditions,samples,info,plans,originals)
    rows=experiment.summarize(draws,originals,info)
    require(len(draws)==1320,'Wrong draw coverage')
    with (out/'draws.jsonl').open('x',encoding='utf8') as fp:
        for d in draws:fp.write(json.dumps(d,allow_nan=False)+'\n')
    write_json(out/'per-video.json',cv.sealed(rows))
    (out/'PER_VIDEO.md').write_text('# Tất cả 33 video · đơn vị lần squat\n\n'+video_table(rows)+'\n',encoding='utf8')
    subsets={'all':rows,**{f'fold {f}':[r for r in rows if r['fold']==f] for f in range(3)},
             **{g:[r for r in rows if r['group_id']==g and r['fold']==0] for g in ('g11','g20')}}
    require([len(v) for v in subsets.values()]==[33,13,13,7,3,4],'Missing reporting subgroup')
    summary={k:experiment.describe(v) for k,v in subsets.items()}
    coverage=dict(frames=sum(r['frames'] for r in rows),complete=sum(r['complete_frames'] for r in rows),
        movable=sum(r['movable_frames'] for r in rows),singleton_runs=sum(r['singletons'] for r in rows),
        videos_with_singletons=sum(r['singletons']>0 for r in rows),
        videos_without_movable_frames=sum(r['movable_frames']==0 for r in rows),
        minimum_longest_run=min(max(r['run_lengths']) for r in rows),
        run_length_histogram=run_length_histogram(rows),
        total_moved=sum(d['positions_moved'] for d in draws if d['seed']==SEEDS[0]),
        total_changed=sum(d['positions_changed'] for d in draws if d['seed']==SEEDS[0]),
        identity_orders=sum(r['identity_orders'] for r in rows))
    metrics={}
    for s,c in conditions.items():
        observed=statistics.mean(r[str(s)]['original_AE'] for r in rows)
        require(observed==c['metrics']['held_out_pooled_MAE'],'Original OOF metric replay mismatch')
        metrics[str(s)]=dict(train_MAE=c['metrics']['train_video_equal_MAE'],oof_MAE=observed,
            fold0_train_MAE=c['fold0']['train']['MAE'],fold0_held_MAE=c['fold0']['held_out']['MAE'])
    write_json(out/'summary.json',cv.sealed(dict(sensitivity=summary,coverage=coverage,original_metrics=metrics)))
    proof=dict(weights=weight_receipts(conditions),sources=ledger.preserved(),original_forwards=66,
        identity_forwards=66,shuffled_forwards=1320,training_steps=0,baseline_fits=0)
    for artifact in ('manifest.json','summary.json','permutations.json','per-video.json','original-replay.json'):
        cv.unseal(read(out/artifact))
    (out/'REPRODUCE.md').write_text(rerun_guide(),encoding='utf8')
    (out/'REPORT.md').write_text(render(rows,summary,coverage,metrics),encoding='utf8')
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    root=Path.cwd();out=local(root,args.output_dir)
    require(out.is_relative_to(root/'outputs'),'Output must be under outputs/')
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        with no_learning(),torch.inference_mode():
            proof=execute(root,out)
    except Exception as error:
        write_json(out/'failure.json',dict(status='INCOMPLETE',detail=str(error),elapsed_seconds=time.perf_counter()-start))
        raise
    write_json(out/'receipt.json',dict(status='COMPLETE_DEVELOPMENT_SENSITIVITY_ONLY',command=[sys.executable,*sys.argv],
        elapsed_seconds=time.perf_counter()-start,**proof))
    print('Complete:',out)


if __name__=='__main__':main()
