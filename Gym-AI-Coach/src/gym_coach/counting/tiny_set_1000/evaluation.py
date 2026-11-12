"""Fixed endpoints and exact500epoch trajectory comparison, no learning/inference."""
from ..tiny_set import manifest as old_m,runtime as old_rt
from ..inner_cv.manifest import unseal
from ..models import STATUS,canonical_hash
from ..research.data import check_identity
from .contract import BUDGET,ENDPOINTS,REPRO_FIELDS
from .diagnostics import final_criterion
from .manifest import experiment_identity
from . import runtime


def compare_500(old,middle,final):
    """Call only after both families' checkpoint loaders validate their histories."""
    if old['epoch']!=499 or middle['epoch']!=499 or final['epoch']!=999:
        raise ValueError('Reproduction requires old499, new499 and new999')
    if len(old['epoch_reports'])!=500 or len(middle['epoch_reports'])!=500 or len(final['epoch_reports'])!=1000:
        raise ValueError('Reproduction missing history')
    if middle['epoch_reports']!=final['epoch_reports'][:500]:
        raise ValueError('New499 is not the exact history prefix of new999; wrong run/lineage')
    differing=[];projections=[]
    for e,(a,b) in enumerate(zip(old['epoch_reports'],middle['epoch_reports'])):
        a=unseal(a);b=unseal(b)
        for key in REPRO_FIELDS:
            if key not in a or key not in b:raise ValueError('Missing meaningful reproduction field')
            if canonical_hash(a[key])!=canonical_hash(b[key]):
                differing.append(dict(epoch_index=e,field=key,old_sha256=canonical_hash(a[key]),new_sha256=canonical_hash(b[key])))
        projections.append(dict(epoch_index=e,old_sha256=canonical_hash({k:a[k] for k in REPRO_FIELDS}),
            new_sha256=canonical_hash({k:b[k] for k in REPRO_FIELDS})))
    import torch
    old_state=old['model'];new_state=middle['model']
    if set(old_state)!=set(new_state):raise ValueError('Checkpoint parameter names differ')
    tensors=[]
    for k in sorted(old_state):
        a=old_state[k];b=new_state[k]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Checkpoint parameter layout differs')
        tensors.append(dict(name=k,equal=torch.equal(a,b),max_abs_difference=float((a.double()-b.double()).abs().max())))
    identical=not differing and all(x['equal'] for x in tensors)
    old_records=unseal(old['epoch_reports'][-1])['evaluation']['records']
    new_records=unseal(middle['epoch_reports'][-1])['evaluation']['records']
    if [r['video_id'] for r in old_records]!=[r['video_id'] for r in new_records]:raise ValueError('Reproduction video pairing differs')
    predictions=[dict(video_id=a['video_id'],target=a['target'],old_prediction=a['prediction'],
        new_prediction=b['prediction'],difference=b['prediction']-a['prediction']) for a,b in zip(old_records,new_records)]
    return dict(status='EXACT_MATCH' if identical else 'MISMATCH_REQUIRES_INVESTIGATION',identical=identical,
        criterion='exact equality, zero tolerance fixed before user run',compared_epochs=500,clips=6,
        ignored_fields=['identity_sha256','elapsed_segment_seconds'],meaningful_fields=list(REPRO_FIELDS),
        differing_fields=differing,history_projections=projections,tensors=tensors,predictions=predictions,
        old_weights_sha256=old['model_sha256'],new_epoch500_weights_sha256=middle['model_sha256'],
        interpretation='Mismatch must be investigated; never attributed automatically to1000epoch budget or corrected by changing tolerance.')


def evaluate(root,manifest,checkpoint,checkpoint_500):
    saved=runtime.load_checkpoint(checkpoint,manifest);middle=runtime.load_checkpoint(checkpoint_500,manifest)
    if saved['epoch']!=999 or middle['epoch']!=499:raise ValueError('Report requires new indices499 and999')
    old_manifest,_=old_m.load(root,check_identity(root,manifest['parent_tiny_manifest']))
    old=old_rt.load_checkpoint(check_identity(root,manifest['old500_sources']['checkpoint']),old_manifest)
    reproduction=compare_500(old,middle,saved)
    history=[unseal(x) for x in saved['epoch_reports']];criterion=final_criterion(history)
    observations=[h['step_observation'] for h in history]
    if sum(x['steps'] for x in observations)!=BUDGET:raise ValueError('Require1000 one-batch steps')
    result=dict(**STATUS,identity=experiment_identity(manifest),manifest_sha256=canonical_hash(manifest),
        status='DIAGNOSTIC_COMPLETE_NOT_VALIDATION' if reproduction['identical'] else 'REPRODUCTION_MISMATCH_REQUIRES_INVESTIGATION',
        old500=manifest['old500_evidence'],reproducibility=reproduction,
        new1000=dict(criterion=criterion,report_epoch_indices=list(ENDPOINTS),
            endpoints={str(i):history[i]['evaluation'] for i in ENDPOINTS},clips=6,
            total_steps=BUDGET,steps_exceeding_clip=sum(x['exceeded_clip_threshold'] for x in observations),
            fraction_steps_exceeding_clip=sum(x['exceeded_clip_threshold'] for x in observations)/BUDGET,
            finite_gradients_all_steps=all(x['gradient']['gradients_finite'] for x in observations),
            all_parameters_have_gradients_all_steps=all(x['gradient']['loss_reaches_all_trainable_parameters'] for x in observations),
            no_parameter_change_steps=sum(x['update']['changed_tensors']==0 for x in observations),
            initial_model_sha256=saved['initial_model_sha256'],final_model_sha256=saved['model_sha256']),
        history=history,input_audit=manifest['input_audit'],new_forward_calls=0,new_training_steps=0,new_fits=0,
        limitation='Adaptive development after viewing500epoch result; six in-sample AI weak targets, one seed; not independent evaluation/generalization/form validation.')
    return result


def report_markdown(result):
    q=result['new1000'];old=result['old500'];repro=result['reproducibility']
    conclusion='Đạt quy ước chẩn đoán trong20epoch cuối.' if q['criterion']['met'] else 'Chưa học khớp trong thiết lập và ngân sách này.'
    lines=['# Tiny-set1000 — user-run diagnostic','',
        'AI weak reference; adaptive development trên cùng6video train, không independent accuracy/generalization/form validation.',
        'NOT_VALIDATED; official_eligibility=false; assessment=false.','',
        '## Kết quả500epoch cũ (giữ nguyên)','',
        f"MAE={old['endpoints']['499']['MAE']:.9f}; maxAE={old['endpoints']['499']['max_absolute_error']:.9f}; final20 pass={old['criterion']['passing_final_epochs']}/20.",
        f"Conclusion: {old['criterion']['conclusion']}.",'',
        '## Kết quả lượt1000epoch mới','',conclusion,'',
        '|Epoch(index)|MAE|RMSE|maxAE|','|---|---:|---:|---:|']
    for i in ENDPOINTS:
        s=q['endpoints'][str(i)];lines.append(f"|{i+1}({i})|{s['MAE']:.9f}|{s['RMSE']:.9f}|{s['max_absolute_error']:.9f}|")
    lines.extend(['',f"Tiêu chí indices980..999: {q['criterion']['passing_final_epochs']}/20; cần đồng thời MAE≤0.10/maxAE≤0.25 trong tất cả20.",
        f"Preclip norm>1: {q['steps_exceeding_clip']}/1000 ({q['fraction_steps_exceeding_clip']:.6f}); no-update steps={q['no_parameter_change_steps']}.",
        f"Finite gradients={q['finite_gradients_all_steps']}; gradients reach all parameters={q['all_parameters_have_gradients_all_steps']}.",
        f"Initial weights: `{q['initial_model_sha256']}`; final: `{q['final_model_sha256']}`.",'',
        '## Tái lập mốc500','',f"Status: **{repro['status']}**; exact comparison, không tolerance.",
        f"Khác biệt fields/epochs: {len(repro['differing_fields'])}; xem reproducibility.json (weights, predictions, meaningful history).",
        'Chỉ bỏ identity metadata và elapsed time; không bỏ loss, sampler, gradients, parameter deltas hoặc prediction.',
        'Nếu mismatch: điều tra trước khi diễn giải tác động ngân sách; không sửa kết quả hay thay tolerance.','',
        'Đạt tiêu chí và khả năng tái lập là hai kết luận riêng. Không kết luận capacity/pooling/code causality từ pass/fail.',
        'history.json và per-video.json có1000epochs/6000video-epoch rows; vẫn chỉ6video, không6000mẫu độc lập.',
        'Loss trong update khác MAE checkpoint cố định. Không best selection/early-stop/automatic extension.',''])
    return '\n'.join(lines)
