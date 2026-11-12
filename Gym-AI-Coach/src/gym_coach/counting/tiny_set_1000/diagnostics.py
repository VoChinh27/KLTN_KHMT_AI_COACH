"""Reuse frozen metrics/telemetry; only fixed final window moves to980..999."""
import math
from ..tiny_set.diagnostics import metrics,gradient_observation,parameter_change,step_log
from .contract import CRITERION,BUDGET


def final_criterion(history):
    if len(history)!=BUDGET or any(type(r.get('epoch')) is not int or r['epoch']!=i for i,r in enumerate(history)):
        raise ValueError('Require ordered complete1000epochs; never select best/shorten denominator')
    checks=[]
    for r in history:
        q=r['evaluation']
        if q['clips']!=6 or any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) or q[k]<0 for k in ('MAE','RMSE','max_absolute_error')):
            raise ValueError('Invalid criterion metrics')
        checks.append(q['MAE']<=CRITERION['MAE_lte'] and q['max_absolute_error']<=CRITERION['max_absolute_error_lte'])
    last=checks[-20:];met=all(last)
    return dict(criterion=CRITERION,met=met,checked_epoch_indices=list(range(980,1000)),
        passing_final_epochs=sum(last),required_final_consecutive_epochs=20,
        conclusion='MET_DIAGNOSTIC_FIT_CONVENTION' if met else 'NOT_FIT_IN_THIS_SETUP_AND_BUDGET',
        independent_evaluation=False)
