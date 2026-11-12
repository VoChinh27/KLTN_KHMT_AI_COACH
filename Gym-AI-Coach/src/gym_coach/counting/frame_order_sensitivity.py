"""Fixed, model-independent within-run pose permutations and equal-video summaries."""
import hashlib
import json
import statistics
from collections import defaultdict

from .models import canonical_hash
from .research.contract import CHANNELS, JOINTS, INPUT_SIZE

VERSION = 'squat-raw58-within-run-frame-order-v1'
REPEATS = 20
POSE_WIDTH = len(CHANNELS) * len(JOINTS)
SEEDS = (20261026, 20261027)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy():
    require((POSE_WIDTH, INPUT_SIZE) == (56, 58), 'Unexpected raw58 schema')
    return dict(version=VERSION, repeats=REPEATS, endpoint_index=499, cohort=33,
        seeds=list(SEEDS), joint_order=list(JOINTS), channels_per_joint=list(CHANNELS),
        pose_columns=list(range(56)), fixed_columns={'56':'elapsed_seconds','57':'dt_seconds'},
        mask='original complete (all eight joints usable); recurrent state resets at false',
        run='maximal consecutive True samples; numbered from zero within each video',
        permutation='destination -> source. Sort each run by SHA-256 of UTF-8 compact JSON array '
        '[version,video_id,run_index,repeat_index,absolute_source_index]. Tie-break by source index. '
        'Repeat indices 0..19. Keep identical orders and repeated permutations.',
        identity_control='separate identity outside the 20 draws; exact original input and output required',
        eligible='all complete frames; movable means length of complete run >=2',
        short_support='report any singleton, only singletons, and run-length histograms; no significance cutoff',
        endpoint='mean of 20 absolute prediction changes per video, then equal mean on 33 videos',
        signed='shuffled minus original continuous prediction in squat repetitions',
        labels='original video only; perturbations have null target and no accuracy/error metric')


def runs(mask):
    require(all(type(x) is bool for x in mask), 'Nonboolean mask')
    start = None
    for i, valid in enumerate([*mask, False]):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            yield start, i
            start = None


def permutation(video_id, mask, repeat):
    require(type(repeat) is int and 0 <= repeat < REPEATS, 'Invalid repeat')
    order = list(range(len(mask)))
    for k, (start, end) in enumerate(runs(mask)):
        def rank(i):
            key = json.dumps([VERSION,video_id,k,repeat,i],ensure_ascii=False,separators=(',',':'))
            return hashlib.sha256(key.encode('utf8')).digest(), i
        order[start:end] = sorted(range(start,end),key=rank)
    return order


def apply_order(sample, order, *, identity_control=False):
    values, mask = sample['values'], sample['complete']
    n = len(mask)
    require(len(values)==len(order)==n and all(len(row)==INPUT_SIZE for row in values), 'Invalid raw58 shape')
    require(all(type(i) is int for i in order) and sorted(order)==list(range(n)), 'Invalid permutation')
    if identity_control:
        require(order==list(range(n)), 'Identity must keep every index')
    for start,end in runs(mask):
        require(set(order[start:end])==set(range(start,end)), 'Moved pose across gap')
    require(all(valid or order[i]==i for i,valid in enumerate(mask)), 'Moved invalid frame')
    result = dict(sample, complete=list(mask), target=sample['target'] if identity_control else None,
                  values=[values[j][:POSE_WIDTH] + values[i][POSE_WIDTH:] for i,j in enumerate(order)])
    return result


def exact_ids(items, ids):
    actual = [r['video_id'] for r in items]
    require(len(actual)==len(set(actual))==len(ids) and set(actual)==set(ids), 'Missing, duplicate or wrong video')


def held_map(folds):
    require([f['fold'] for f in folds]==[0,1,2], 'Wrong three folds')
    held = {}
    for fold,n in zip(folds,(13,13,7),strict=True):
        ids=fold['evaluation_ids']
        require(len(ids)==n and len(set(ids))==n and not(set(ids)&(set(fold['train_ids'])|held.keys())),
                'Repeated held video, leakage or incorrect fold size')
        held.update(dict.fromkeys(ids,fold['fold']))
    require(len(held)==33,'Expected 33 unique held-out videos')
    return held


def summarize(draws, originals, metadata):
    ids=set(metadata)
    require(len(ids)==33 and set(originals)=={(v,s) for v in ids for s in SEEDS}, 'Expected 66 originals')
    by_pair=defaultdict(dict)
    for d in draws:
        key=(d['video_id'],d['seed']);r=d['repeat']
        require(key in originals and type(r) is int and 0<=r<REPEATS and r not in by_pair[key],
                'Unknown or duplicate video/seed/repeat')
        original=originals[key]
        require(d['fold']==original['fold'] and d['original_prediction']==original['prediction']
                and d['delta']==d['shuffled_prediction']-d['original_prediction']
                and d['abs_delta']==abs(d['delta']), 'Prediction/fold/delta mismatch')
        by_pair[key][r]=d
    rows=[]
    for video in sorted(ids):
        row=dict(metadata[video])
        for seed in SEEDS:
            records=by_pair[(video,seed)]
            require(set(records)==set(range(REPEATS)), 'Missing draws')
            values=list(records.values()); orig=originals[video,seed]
            row[str(seed)]=dict(original_prediction=orig['prediction'],original_signed_error=orig['prediction']-row['target'],
                original_AE=abs(orig['prediction']-row['target']),mean_abs_delta=statistics.mean(d['abs_delta'] for d in values),
                mean_signed_delta=statistics.mean(d['delta'] for d in values))
        row['paired_difference']=row[str(SEEDS[1])]['mean_abs_delta']-row[str(SEEDS[0])]['mean_abs_delta']
        for r in range(REPEATS):
            a,b=[by_pair[video,s][r] for s in SEEDS]
            require(all(a[k]==b[k] for k in ('permutation_sha256','model_input_sha256','positions_moved','positions_changed')),
                    'Seed inputs differ')
        rows.append(row)
    return rows


def describe(rows):
    result={'videos':len(rows)}
    for seed in SEEDS:
        values=[r[str(seed)]['mean_abs_delta'] for r in rows]
        result[str(seed)]=dict(mean_abs_delta=statistics.mean(values),median_abs_delta=statistics.median(values),
            mean_signed_delta=statistics.mean(r[str(seed)]['mean_signed_delta'] for r in rows))
    pairs=[r['paired_difference'] for r in rows]
    result['paired_new_minus_old']=dict(mean=statistics.mean(pairs),median=statistics.median(pairs),
        higher=sum(x>0 for x in pairs),lower=sum(x<0 for x in pairs),equal=pairs.count(0.0))
    return result
