"""Report saved, complete500epoch history; never infer, select or learn."""
from ..inner_cv.manifest import unseal
from ..models import STATUS,canonical_hash
from .contract import BUDGET
from .diagnostics import final_criterion
from .manifest import experiment_identity
from . import runtime


def evaluate(manifest,checkpoint):
    saved=runtime.load_checkpoint(checkpoint,manifest)
    if saved['epoch']!=BUDGET-1:raise ValueError('Report requires final index499; no early/best checkpoint')
    history=[unseal(x) for x in saved['epoch_reports']]
    criterion=final_criterion(history)
    observations=[r['step_observation'] for r in history]
    if sum(x['steps'] for x in observations)!=500:raise ValueError('Require500 one-batch steps')
    return dict(**STATUS,identity=experiment_identity(manifest),manifest_sha256=canonical_hash(manifest),
        status='DIAGNOSTIC_COMPLETE_NOT_VALIDATION',criterion=criterion,
        scope='six same training videos; in-sample; no held-out use; AI weak reference',
        clips=6,groups=len({r['source']['group_id'] for r in manifest['rows']}),
        report_epoch_indices=[49,499],endpoints={str(i):history[i]['evaluation'] for i in (49,499)},
        initial_model_sha256=saved['initial_model_sha256'],final_model_sha256=saved['model_sha256'],
        total_steps=500,steps_exceeding_clip=sum(x['exceeded_clip_threshold'] for x in observations),
        fraction_steps_exceeding_clip=sum(x['exceeded_clip_threshold'] for x in observations)/500,
        finite_gradients_all_steps=all(x['gradient']['gradients_finite'] for x in observations),
        all_parameters_have_gradients_all_steps=all(x['gradient']['loss_reaches_all_trainable_parameters'] for x in observations),
        no_parameter_change_steps=sum(x['update']['changed_tensors']==0 for x in observations),
        input_audit=manifest['input_audit'],history=history,
        new_forward_calls=0,new_training_steps=0,new_fits=0,
        limitation='Thresholds are diagnostic conventions, not accuracy/form standards. Not a single-factor comparison with full CV. No causal/capacity conclusion from failure.')


def report_markdown(result):
    conclusion=('Đạt quy ước học khớp tiny-set trong 20 epoch cuối.' if result['criterion']['met']
                else 'Chưa học khớp trong thiết lập và ngân sách này.')
    lines=['# Tiny-set diagnostic — user-run result','',conclusion,'',
        'AI weak reference; six training clips only. NOT_VALIDATED; official_eligibility=false; assessment=false.',
        'Không phải accuracy độc lập, generalization, model selection hoặc form validation.',
        'Không suy nguyên nhân, capacity hoặc lỗi code chỉ từ pass/fail. Không chọn best epoch.','',
        '| Epoch (index) | Clips | MAE | RMSE | Max AE |','|---|---:|---:|---:|---:|']
    for i in (49,499):
        q=result['endpoints'][str(i)]
        lines.append(f"| {i+1} ({i}) | 6 | {q['MAE']:.9f} | {q['RMSE']:.9f} | {q['max_absolute_error']:.9f} |")
    lines.extend(['',f"Last20 passing epochs: {result['criterion']['passing_final_epochs']}/20; require all20.",
        f"Preclip norm >1: {result['steps_exceeding_clip']}/500 ({result['fraction_steps_exceeding_clip']:.6f}).",
        f"All gradients finite: {result['finite_gradients_all_steps']}; all parameters receive gradients: {result['all_parameters_have_gradients_all_steps']}.",
        f"Steps with no parameter change: {result['no_parameter_change_steps']}/500.",
        f"Initial weights SHA-256: `{result['initial_model_sha256']}`.",f"Final weights SHA-256: `{result['final_model_sha256']}`.",'',
        'Loss is SmoothL1 on the updating model; reported MAE/RMSE are post-update eval/inference_mode on the same six videos.',
        'See history.json for all500 epochs, predictions/errors, preclip gradients, parameter deltas and weight hashes.',
        'See per-video.json for3000 video-epoch rows; these are repeated observations of6 videos, not3000 independent samples.',
        'Gradient/update runtime evidence is provided by user training; preparation tests only checked fixture telemetry.',''])
    return '\n'.join(lines)
