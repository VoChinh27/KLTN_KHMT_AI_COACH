"""Agreement with AI reference, never independent accuracy or event evaluation."""
import math
from collections import Counter
from .contract import ROLES


def rounded_count(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite count prediction required')
    return math.floor(max(0.0,value)+0.5)


def evaluate(rows, predictions, role):
    if role not in ROLES:
        raise ValueError('Unknown split')
    subset = [r for r in rows if r['role']==role]
    ids = [r['source']['video_id'] for r in subset]
    if len(ids)!=len(set(ids)) or set(predictions)-set(ids):
        raise ValueError('Duplicate cohort or unexpected prediction ID')
    records, errors, exact = [], [], []
    for row in subset:
        video_id = row['source']['video_id']
        pred = predictions.get(video_id)
        value, reason = None, None
        if row['input_status']!='accepted':
            status, reason = 'unavailable', 'research_input_'+row['input_status']
            if pred is not None:
                raise ValueError('Prediction supplied for inadmissible input')
        elif pred is None:
            status, reason = 'not_run', 'prediction_not_supplied'
        elif isinstance(pred,dict) and pred.get('status')=='failed' and pred.get('reason'):
            status, reason = 'failed', pred['reason']
        elif type(pred) not in (float,int) or not math.isfinite(pred):
            status, reason = 'failed', 'non_finite_or_invalid_prediction'
        else:
            status, value = 'estimated', max(0.0,pred)
        target = row['target']
        if target is not None and (type(target) is not int or target<0):
            raise ValueError('Invalid AI count target')
        rounded = rounded_count(value) if value is not None else None
        if target is not None and value is not None:
            errors.append(value-target)
            exact.append(rounded==target)
        records.append(dict(video_id=video_id, source_sha256=row['source']['sha256'],
            label_record_hash=row.get('label_record_hash'), target=target, prediction=value,
            rounded_prediction=rounded, status=status, reason=reason,
            operational_detail=pred.get('detail') if isinstance(pred,dict) else None,
            input_reasons=row.get('input_reasons',[]), comparable=target is not None and value is not None))
    n, comparable = len(subset),len(errors)
    labels = sum(r['target'] is not None for r in subset)
    admitted = sum(r['input_status']=='accepted' for r in subset)
    predicted = sum(r['prediction'] is not None for r in records)
    return dict(kind='agreement_with_AI_reference_not_independent_accuracy',role=role,
        cohort=n,labels_available=labels,input_accepted=admitted,predictions_available=predicted,
        comparable=comparable,coverage=dict(input_over_cohort=admitted/n if n else None,
            label_over_cohort=labels/n if n else None,prediction_over_cohort=predicted/n if n else None,
            comparable_over_cohort=comparable/n if n else None,comparable_over_labels=comparable/labels if labels else None),
        statuses=dict(Counter(r['status'] for r in records)),
        continuous_MAE=sum(map(abs,errors))/comparable if comparable else None,
        signed_error=sum(errors)/comparable if comparable else None,
        rounded_exact_match=sum(exact)/comparable if comparable else None,
        exact_match_rule='floor(max(0, continuous_prediction)+0.5)',records=records)


def better_checkpoint(score, best):
    """Strict improvement; earliest epoch wins ties. No test data accepted here."""
    return score is not None and math.isfinite(score) and (best is None or score<best)
