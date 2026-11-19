"""Read/replay saved tiny result and discover baseline controls from verified receipts."""
from pathlib import Path
from ..mse import manifest as mse
from ..tiny_set_1000 import manifest as tm,evaluation as te
from ..inner_cv.manifest import unseal
from ..models import canonical_hash
from ..readiness import file_info
from ..research.data import local,read,identity,check_identity


def verify_tiny(root,folder,review_path):
    folder=local(root,folder);receipt=read(folder/'receipt.json');command=receipt['command']
    path=local(root,mse.command_arg(command,'--manifest'));manifest,_=tm.load(root,path)
    final=local(root,mse.command_arg(command,'--checkpoint'));middle=local(root,mse.command_arg(command,'--checkpoint-500'))
    result=te.evaluate(root,manifest,final,middle)
    result['checkpoints']=dict(final=identity(root,final),epoch500=identity(root,middle))
    if unseal(read(folder/'result.json'))!=result or receipt['result']!=result:
        raise ValueError('Tiny1000 result differs from checkpoint replay')
    if unseal(read(folder/'history.json'))!=result['history'] or unseal(read(folder/'reproducibility.json'))!=result['reproducibility']:
        raise ValueError('Tiny1000 history/reproduction differs')
    rows=[dict(r,epoch=h['epoch']) for h in result['history'] for r in h['evaluation']['records']]
    if unseal(read(folder/'per-video.json'))!=rows or (folder/'REPORT.md').read_text(encoding='utf-8')!=te.report_markdown(result):
        raise ValueError('Tiny1000 per-video/report differs')
    for name,h in read(folder/'inventory.json')['files'].items():
        if file_info(local(root,folder/name))!=h:raise ValueError('Tiny1000 inventory mismatch')
    for e,h in enumerate(result['history']):
        cp=final.parent/f'epoch-{e:03d}.pt'
        if file_info(cp)!=read(str(cp)+'.sha256.json') or unseal(read(final.parent/f'epoch-{e:03d}.json'))!=h:
            raise ValueError('Tiny1000 committed checkpoint/history mismatch')
    review=read(local(root,review_path));new=result['new1000']
    if (review['status']!='VERIFIED' or review['epochs_verified']!=1000 or review['video_epoch_records']!=6000
            or review['checkpoint']!=identity(root,final) or review['result_sha256']!=canonical_hash(result)
            or review['criterion']!=new['criterion'] or review['reproducibility']!=result['reproducibility']['status']
            or review['reproduction_compared_epochs']!=500 or local(root,review['manifest'])!=path):
        raise ValueError('Tiny1000 review provenance/criterion mismatch')
    for e in ('49','499','999'):
        if review['endpoints'][e]!={k:new['endpoints'][e][k] for k in ('MAE','RMSE','max_absolute_error')}:
            raise ValueError('Tiny1000 review endpoint mismatch')
    if not new['criterion']['met'] or not result['reproducibility']['identical']:
        raise ValueError('Tiny1000 evidence contradicts stated motivation; do not substitute design')
    links=dict(manifest=identity(root,path),checkpoints=result['checkpoints'],
        report={p.name:identity(root,p) for p in sorted(folder.iterdir()) if p.is_file()},review=identity(root,local(root,review_path)))
    evidence=dict(status='VERIFIED_SAVED_ONLY',criterion=new['criterion'],endpoints=new['endpoints'],
        reproducibility=result['reproducibility']['status'],reproduction_compared_epochs=500,
        finite_gradients=new['finite_gradients_all_steps'],no_parameter_change_steps=new['no_parameter_change_steps'],
        verified_epochs=1000,video_epoch_records=6000,new_forward_calls=0,new_training_steps=0,new_fits=0)
    return links,evidence


def verify_controls(root,original_manifest,gru_run,control_receipt):
    receipt_path=local(root,control_receipt);command=read(receipt_path)['command']
    path=local(root,mse.command_arg(command,'--manifest'))
    condition,data,all_records=mse.load(root,path);links=condition['controls']
    if links['original_manifest']!=identity(root,local(root,original_manifest)) or local(root,links['original_gru_run'])!=local(root,gru_run):
        raise ValueError('Control receipt lineage does not match requested original CV/run')
    base=unseal(read(check_identity(root,links['original_manifest'])))
    control_links=dict(links,discovery_receipt=identity(root,receipt_path),discovered_manifest=identity(root,path))
    controls=[r for r in all_records if r['method'] in ('gru','duration','constant')]
    if len(controls)!=297:raise ValueError('Require297 saved control appearances, no fitting')
    return base,data,control_links,controls
