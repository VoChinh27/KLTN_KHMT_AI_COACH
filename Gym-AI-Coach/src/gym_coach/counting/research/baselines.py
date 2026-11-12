"""User-run fitting only. Nothing is fitted on import or during preparation."""
import statistics
from ..models import canonical_hash


def fit_baselines(rows):
    train = [r for r in rows if r['role']=='train' and r['supervised_eligible']]
    if not train:
        raise ValueError('No admitted train targets')
    y = [r['target'] for r in train]
    x = [r['source']['frame_count']/r['source']['fps'] for r in train]
    mx,my = statistics.mean(x),statistics.mean(y)
    variance = sum((t-mx)**2 for t in x)
    slope = sum((t-mx)*(v-my) for t,v in zip(x,y))/variance if variance else 0.0
    return dict(version='squat-count-baselines-v1',constant=statistics.median(y),
        duration_intercept=my-slope*mx,duration_slope=slope,
        duration_degenerate=variance==0,
        training_ids=sorted(r['source']['video_id'] for r in train),
        training_subset_hash=canonical_hash(sorted((r['source']['video_id'],r['source']['sha256'],r['target']) for r in train)),
        label_kind='AI_weak_video_complete_count')


def predict(parameters,rows,role,method):
    if method not in ('constant','duration'):
        raise ValueError('Unknown baseline')
    return {r['source']['video_id']:max(0.0,parameters['constant'] if method=='constant' else
                parameters['duration_intercept']+parameters['duration_slope']*r['source']['frame_count']/r['source']['fps'])
            for r in rows if r['role']==role and r['input_status']=='accepted'}
