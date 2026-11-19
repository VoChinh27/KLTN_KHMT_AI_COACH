"""Descriptive diagnostics only. Does not import a model or production transform."""
import copy
import math
import statistics
from collections import Counter


def require(condition,message):
    if not condition:raise ValueError(message)


def describe(values):
    values=list(values)
    require(all(type(x) in (float,int) and math.isfinite(x) for x in values),'Nonfinite descriptive input')
    if not values:return dict(n=0,minimum=None,maximum=None,mean=None,median=None,population_sd=None)
    return dict(n=len(values),minimum=min(values),maximum=max(values),mean=statistics.mean(values),
                median=statistics.median(values),population_sd=statistics.pstdev(values))


def exact_ids(rows,expected):
    ids=[r['video_id'] for r in rows]
    require(len(ids)==len(set(ids))==len(expected)==len(set(expected)) and set(ids)==set(expected),
            'Missing/duplicate/unexpected video; denominator cannot shrink')


def runs(mask,value):
    result=[];start=None
    for i,item in enumerate([*mask,not value]):
        if item==value and start is None:start=i
        if item!=value and start is not None:
            result.append(dict(start=start,end=i-1,frames=i-start));start=None
    return result


def independent_geometry(sample,epsilon):
    """Independent indexing, arithmetic and sorted median; never calls normalize/midpoint."""
    values=sample['values'];mask=sample['complete']
    require(len(values)==len(mask)>0 and all(type(m) is bool for m in mask),'Invalid mask')
    lengths=[];centers={}
    for i,row in enumerate(values):
        require(len(row)==58 and all(type(v) in (int,float) and math.isfinite(v) for v in row),'Invalid raw58')
        if not mask[i]:continue
        hx=(row[14]+row[21])/2;hy=(row[15]+row[22])/2
        sx=(row[0]+row[7])/2;sy=(row[1]+row[8])/2
        length=math.sqrt((sx-hx)**2+(sy-hy)**2)
        if math.isfinite(length) and length>epsilon:lengths.append(length)
        centers[i]=(hx,hy)
    require(bool(lengths),'No independent valid scale')
    ordered=sorted(lengths);n=len(ordered)
    scale=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    out=copy.deepcopy(sample)
    for i,(hx,hy) in centers.items():
        for col in range(0,56,7):
            out['values'][i][col]=(values[i][col]-hx)/scale
            out['values'][i][col+1]=(values[i][col+1]-hy)/scale
    return out,scale


def check_transform(raw,normalized,epsilon,tolerance):
    independent,scale=independent_geometry(raw,epsilon)
    require(set(raw)==set(normalized),'Sample structure changed')
    for key in raw:
        if key!='values':require(raw[key]==normalized[key],'Non-coordinate metadata changed: '+key)
    require(len(raw['values'])==len(normalized['values']),'Frame count changed')
    max_difference=0.;residual=0.
    for i,(a,b,c) in enumerate(zip(raw['values'],normalized['values'],independent['values'])):
        require(len(b)==58 and all(math.isfinite(x) for x in b),'Nonfinite transformed input')
        for j in range(58):
            if not raw['complete'][i] or j>=56 or j%7 not in (0,1):
                require(a[j]==b[j],f'Preserved channel changed: frame{i} column{j}')
            max_difference=max(max_difference,abs(b[j]-c[j]))
        if raw['complete'][i]:residual=max(residual,abs((b[14]+b[21])/2),abs((b[15]+b[22])/2))
    require(max_difference<=tolerance and residual<=tolerance,'Independent geometry mismatch')
    return dict(independent_scale=scale,max_abs_coordinate_difference=max_difference,
        max_abs_hip_midpoint=residual,tolerance=tolerance,incomplete_and_nonxy_exact=True)


def measure(sample,normalized,scale,fps):
    """No learned/outlier threshold. Steps use adjacent complete samples only."""
    require(fps>0,'Invalid FPS')
    raw=sample['values'];new=normalized['values'];mask=sample['complete']
    frames=[];hips=[];lengths=[];steps=[]
    for i,(row,valid) in enumerate(zip(raw,mask)):
        record=dict(frame_index=i,timestamp_seconds=i/fps,complete=valid,hip=None,torso_length=None)
        if valid:
            h=[(row[14]+row[21])/2,(row[15]+row[22])/2]
            s=[(row[0]+row[7])/2,(row[1]+row[8])/2]
            length=math.dist(s,h);hips.append(h);lengths.append(length)
            record.update(hip=h,torso_length=length,torso_over_video_scale=length/scale)
            if i and mask[i-1]:
                old=raw[i-1];prev=[(old[14]+old[21])/2,(old[15]+old[22])/2]
                distances=[math.dist(row[j:j+2],old[j:j+2]) for j in range(0,56,7)]
                distances_new=[math.dist(new[i][j:j+2],new[i-1][j:j+2]) for j in range(0,56,7)]
                steps.append(dict(previous_frame=i-1,frame_index=i,timestamp_seconds=i/fps,
                    hip_step=math.dist(h,prev),raw_max_joint_step=max(distances),raw_mean_joint_step=statistics.mean(distances),
                    normalized_max_joint_step=max(distances_new),normalized_mean_joint_step=statistics.mean(distances_new)))
        frames.append(record)
    gaps=runs(mask,False);valid_runs=runs(mask,True)
    for span in gaps+valid_runs:span.update(start_seconds=span['start']/fps,end_seconds=span['end']/fps)
    torso=describe(lengths)
    torso['population_cv']=torso['population_sd']/torso['mean'] if torso['mean'] else None
    torso['max_min_ratio']=torso['maximum']/torso['minimum'] if torso['minimum'] else None
    amplitudes={}
    for name,vs in (('raw',raw),('normalized',new)):
        amplitudes[name]={axis:describe(v[j+a] for v,ok in zip(vs,mask) if ok for j in range(0,56,7))
                          for a,axis in enumerate(('x','y'))}
        amplitudes[name]['per_joint_ranges']=[dict(joint_index=k,
            x_range=max(v[7*k] for v,ok in zip(vs,mask) if ok)-min(v[7*k] for v,ok in zip(vs,mask) if ok),
            y_range=max(v[7*k+1] for v,ok in zip(vs,mask) if ok)-min(v[7*k+1] for v,ok in zip(vs,mask) if ok)) for k in range(8)]
    return dict(frames=len(mask),complete_frames=sum(mask),complete_fraction=sum(mask)/len(mask),
        incomplete_frames=sum(not x for x in mask),gaps=gaps,complete_runs=valid_runs,
        longest_gap_frames=max([r['frames'] for r in gaps],default=0),hidden_zero_samples=sum(not x for x in mask),
        resets_after_complete=sum(a and not b for a,b in zip(mask,mask[1:])),
        restarts_after_gap=sum(not a and b for a,b in zip(mask,mask[1:])),scale=scale,torso=torso,
        hip=dict(x=describe(h[0] for h in hips),y=describe(h[1] for h in hips),
                 x_range=max(h[0] for h in hips)-min(h[0] for h in hips),
                 y_range=max(h[1] for h in hips)-min(h[1] for h in hips),
                 path_adjacent_complete=sum(r['hip_step'] for r in steps)),
        coordinates=amplitudes,adjacent_complete_edges=len(steps),
        steps={key:describe(s[key] for s in steps) for key in ('hip_step','raw_max_joint_step','normalized_max_joint_step')},
        largest_steps=sorted(steps,key=lambda s:(-s['raw_max_joint_step'],s['frame_index']))[:3],
        per_frame=frames,nonfinite_values=0,
        identity_switch='not inferable from these numbers alone; requires visual review')


def compare_distribution(train,held):
    def fields(r):
        s=r['pose'];return dict(target=r['target'],duration=r['duration_seconds'],scale=s['scale'],
            complete_fraction=s['complete_fraction'],longest_gap=s['longest_gap_frames'],torso_cv=s['torso']['population_cv'],
            hip_y_range=s['hip']['y_range'])
    result={}
    for key in fields(train[0]):
        a=[fields(r)[key] for r in train];b=[fields(r)[key] for r in held]
        result[key]=dict(train=describe(a),held_out=describe(b),
            held_inside_train_minmax=sum(min(a)<=v<=max(a) for v in b),held_denominator=len(b),
            outside_train_range=[r['video_id'] for r,v in zip(held,b) if not min(a)<=v<=max(a)])
    return dict(train_n=len(train),held_n=len(held),
        count_histograms={side:dict(sorted(Counter(str(r['target']) for r in rows).items())) for side,rows in (('train',train),('held_out',held))},
        summaries=result,interpretation='Marginal empirical ranges; overlap does not establish multivariate support or subject/generalization validity')


def paired_table(records,metadata,expected):
    exact_ids([r for r in records if r['method']=='body_normalized'],expected)
    lookup={}
    for r in records:
        key=r['method'],r['video_id']
        require(key not in lookup,'Duplicate paired row');lookup[key]=r
    require(len(lookup)==len(expected)*3,'Require all body/raw/duration predictions')
    out=[]
    for v in expected:
        meta=metadata[v];peers={k:lookup[k,v] for k in ('body_normalized','raw58','duration')}
        for r in peers.values():
            require(r['target']==meta['target'] and r['source_sha256']==meta['source']['sha256'] and r['fold']==1,'Paired source/target/fold mismatch')
            require(r['status']=='estimated' and r['reason'] is None and math.isfinite(r['prediction']),'Invalid paired prediction')
        ae={k:abs(r['prediction']-meta['target']) for k,r in peers.items()}
        out.append(dict(video_id=v,group_id=meta['group_id'],target=meta['target'],duration_seconds=meta['duration_seconds'],fold=1,
            source_sha256=meta['source']['sha256'],predictions={k:r['prediction'] for k,r in peers.items()},
            signed_errors={k:r['prediction']-meta['target'] for k,r in peers.items()},absolute_errors=ae,
            delta_AE=ae['body_normalized']-ae['raw58']))
    groups={}
    for g in sorted({r['group_id'] for r in out}):
        rows=[r for r in out if r['group_id']==g];total=math.fsum(r['delta_AE'] for r in rows)
        groups[g]=dict(n=len(rows),methods={k:dict(MAE=statistics.mean(r['absolute_errors'][k] for r in rows),
            signed_mean=statistics.mean(r['signed_errors'][k] for r in rows)) for k in ('body_normalized','raw58','duration')},
            total_delta_AE=total,contribution_to_fold_MAE_increase=total/len(expected),
            contribution_fraction_of_net_delta=total/math.fsum(r['delta_AE'] for r in out) if math.fsum(r['delta_AE'] for r in out) else None)
    return dict(records=out,groups=groups,n=len(out),delta_MAE=statistics.mean(r['delta_AE'] for r in out),
        unit='13 unique held-out videos per seed; group contribution=sum(delta_AE in group)/13; negative contributions offset increases')
