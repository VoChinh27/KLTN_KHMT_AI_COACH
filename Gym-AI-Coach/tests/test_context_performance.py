"""Replay/benchmark integrity boundaries; tiny synthetic sources, no Data I/O."""
import copy
import json

import pytest

from gym_coach.contracts import Viewpoint
from gym_coach.reporting import context_performance as perf
from gym_coach.reporting import real_descriptive_demo as single
from stage5_6_fixture import fixture_result


def job_source(tmp_path):
    result=fixture_result(n=9);folder=tmp_path/'saved';folder.mkdir()
    raw=tuple(f.model_copy(update={'viewpoint':Viewpoint.UNKNOWN}) for f in result.raw_frames)
    path='Data/UCF-101_GYM/BodyWeightSquats/synthetic.avi'
    pairs=[single.rank_pair(path,side,raw,result.quality) for side in ('left','right')]
    files={'preprocessing.json':result.model_dump_json().encode(),
        'source.json':json.dumps(dict(decoder_metadata=result.metadata.model_dump(mode='json'),
            candidate=dict(path=path,sha256='1'*64),camera_view='UNKNOWN',
            assumption_status='visually_assessed_unverified')).encode(),
        'raw-pose-unknown.jsonl':b'\n'.join(f.model_dump_json().encode() for f in raw)}
    for name,data in files.items():(folder/name).write_bytes(data)
    return dict(directory=str(folder),path=path,pairs=pairs,source_video_sha256='1'*64,
        files={n:dict(sha256=perf.digest_bytes(v),size_bytes=len(v)) for n,v in files.items()}),result.profile


def install_reference(job,reference):
    from pathlib import Path
    for side in ('left','right'):
        data=(reference/(side+'.jsonl')).read_bytes();(Path(job['directory'])/(side+'.jsonl')).write_bytes(data)
        job['files'][side+'.jsonl']=dict(sha256=perf.digest_bytes(data),size_bytes=len(data))


def test_identical_work_full_rows_and_validation_counts(tmp_path):
    job,profile=job_source(tmp_path)
    reference=perf.replay_clip(job,tmp_path/'reference','reference',profile)
    optimized=perf.replay_clip(job,tmp_path/'session','session',profile)
    assert reference['pairs']==optimized['pairs']
    assert reference['receipt']['validation_counts']==dict(load=1,session_setup=0,per_frame=18,total=19)
    assert optimized['receipt']['validation_counts']==dict(load=1,session_setup=1,per_frame=0,total=2)
    assert optimized['receipt']['adapter_calls']==18 and optimized['receipt']['factory_calls']==9
    assert optimized['receipt']['session_metadata'] and not reference['receipt']['session_metadata']
    install_reference(job,tmp_path/'reference')
    assert perf.compare_records(job,tmp_path/'session',validate_models=True)['matched']==18
    row=json.loads((tmp_path/'session/left.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert 'session_metadata' not in row and 'session_version' not in row['observation']
    assert set(reference['receipt']['timing'])==set(optimized['receipt']['timing'])
    assert all(v>=0 for v in optimized['receipt']['timing'].values())


@pytest.mark.parametrize('field',['value','provenance','input_hash','extra_field','missing_row'])
def test_comparison_does_not_hide_any_baseline_difference(tmp_path,field):
    job,profile=job_source(tmp_path)
    perf.replay_clip(job,tmp_path/'reference','reference',profile);install_reference(job,tmp_path/'reference')
    perf.replay_clip(job,tmp_path/'session','session',profile)
    p=tmp_path/'session/left.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    if field=='value':rows[0]['observation']['value']+=1.
    elif field=='provenance':rows[0]['observation']['joint_inputs'][0]['source_status']='missing'
    elif field=='input_hash':rows[0]['source_linkage']['measurement_input_hash']='0'*64
    elif field=='extra_field':rows[0]['unapproved_metadata']=True
    else:rows.pop()
    p.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    with pytest.raises(ValueError):perf.compare_records(job,tmp_path/'session')


@pytest.mark.parametrize('fault',['bytes','side','quality','profile'])
def test_source_compatibility_and_hash_checks_remain_mandatory(tmp_path,fault):
    job,profile=job_source(tmp_path)
    if fault=='bytes':
        from pathlib import Path
        p=Path(job['directory'])/'preprocessing.json';p.write_bytes(p.read_bytes()+b' ')
    elif fault=='side':job['pairs'].reverse()
    elif fault=='quality':job['pairs'][0]['metric']['action']='skip'
    else:profile=profile.model_copy(update={'profile_version':'other'})
    with pytest.raises(ValueError):perf.replay_clip(job,tmp_path/'attempt','session',profile)


def test_unknown_mode_and_existing_output_are_not_overwritten(tmp_path):
    job,profile=job_source(tmp_path)
    with pytest.raises(ValueError):perf.replay_clip(job,tmp_path/'new','other',profile)
    dest=tmp_path/'existing';dest.mkdir();(dest/'keep').write_text('saved')
    with pytest.raises(FileExistsError):perf.replay_clip(job,dest,'session',profile)
    assert (dest/'keep').read_text()=='saved'


def test_memory_is_reported_without_claiming_session_allocation():
    value=perf.memory_usage()
    assert isinstance(value['available'],bool)
    if value['available']:
        assert value['process_peak_working_set_bytes']>=value['working_set_bytes']>0
