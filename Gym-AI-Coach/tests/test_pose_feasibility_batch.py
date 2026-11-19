"""Synthetic orchestration checks; never creates a MediaPipe backend."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_quality import make_frame
from gym_coach.contracts import Viewpoint
from gym_coach.pose.serialization import write_pose_frames
from gym_coach.pose.coverage import summarize_pose_sequence

spec=importlib.util.spec_from_file_location('feasibility',Path(__file__).resolve().parents[1]/'scripts/squat_pose_feasibility.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)


@pytest.fixture
def case(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    video=Path('Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi')
    video.parent.mkdir(parents=True);video.write_bytes(b'synthetic fixture, not real video')
    source=dict(video_id=video.name,path=video.as_posix(),sha256=batch.file_info(video)['sha256'],
                group_id='g01',fps=30.,frame_count=6,width=320,height=240,timestamp_policy='fps_derived_cfr_assumption')
    config=batch.configuration();config['mediapipe_version']='synthetic';config['model_asset']['sha256']=None
    row=dict(source=source,action='infer');plan=dict(version=batch.VERSION,configuration=config,videos=[row])
    run=(tmp_path/'outputs/run');run.mkdir(parents=True)
    monkeypatch.setattr(batch,'read_plan',lambda p:plan)
    def extractor(path,exercise,folder,config=None):
        folder.mkdir()
        frames=[make_frame(i,viewpoint=Viewpoint.UNKNOWN).model_copy(update={'video_id':video.name}) for i in range(6)]
        write_pose_frames(folder/'frames.jsonl',frames)
        summary=summarize_pose_sequence(frames,video_id=video.name,exercise=exercise,backend=frames[0].backend,config=frames[0].config)
        batch.write_json(folder/'summary.json',summary.model_dump(mode='json'))
        metadata={k:source[k] for k in ('video_id','fps','frame_count','width','height','timestamp_policy')}
        batch.write_json(folder/'manifest.json',dict(completed=True,input_path=str(path),input_sha256=source['sha256'],metadata=metadata,frames_written=6,pose_frames_sha256=batch.file_info(folder/'frames.jsonl')['sha256']))
    import gym_coach.pose.run
    monkeypatch.setattr(gym_coach.pose.run,'extract_video',extractor)
    attempt=run/'clips'/video.name/'attempt-test';attempt.mkdir(parents=True)
    return run,plan,row,attempt


def test_unknown_and_exact_joint_requirements(case):
    run,plan,row,attempt=case
    assert batch.worker(run,row['source']['video_id'],attempt)==0
    result=batch.verify_completed(plan,row,attempt)
    assert result['sequence']['action']=='allow'
    for side,key in zip(('left','right'),batch.KEYS):
        metric=result['metrics'][key]
        assert metric['required_joints']==[side+'_hip',side+'_knee',side+'_ankle']
        assert metric['action']=='skip' and 'viewpoint_unknown' in metric['reasons']


def test_resume_never_calls_inference(case,monkeypatch):
    run,plan,row,attempt=case
    assert batch.worker(run,row['source']['video_id'],attempt)==0
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**k:pytest.fail('valid output reinferred'))
    receipt=batch.execute(run)
    assert receipt['attempted']==0 and receipt['states']=={'verified_resume':1}


@pytest.mark.parametrize('mutation',['source','frames','quality','plan'])
def test_resume_rejects_tampering(case,mutation):
    run,plan,row,attempt=case
    assert batch.worker(run,row['source']['video_id'],attempt)==0
    if mutation=='source':Path(row['source']['path']).write_bytes(b'changed')
    elif mutation=='plan':plan['configuration']['camera_view']='side'
    else:(attempt/('pose/frames.jsonl' if mutation=='frames' else 'quality.json')).write_text('{}')
    with pytest.raises(Exception):batch.verify_completed(plan,row,attempt)


def test_interrupted_attempt_kept_and_all_rows_reported(case):
    run,plan,row,attempt=case
    result=batch.aggregate(run)
    assert result['video_denominator']==1
    assert result['rows'][0]['status']=='failed_or_incomplete'
    assert result['rows'][0]['conditions_B']['timeline']=='UNKNOWN'
    assert result['conditions_B_quality_subset_pass']==0 and result['accuracy'] is None
    assert attempt.exists()


def test_failure_receipt_and_continue_next_clip(case,monkeypatch):
    run,plan,row,attempt=case
    second=json.loads(json.dumps(row));second['source']['video_id']='v_BodyWeightSquats_g01_c02.avi'
    second['source']['path']=str(Path(row['source']['path']).with_name(second['source']['video_id']))
    Path(second['source']['path']).write_bytes(Path(row['source']['path']).read_bytes());plan['videos'].append(second)
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=7))
    receipt=batch.execute(run)
    assert receipt['attempted']==2 and receipt['states']=={'failed':2}
    assert len(batch.aggregate(run)['rows'])==2


def test_condition_checks_do_not_promote_counting(case):
    run,plan,row,attempt=case;assert batch.worker(run,row['source']['video_id'],attempt)==0
    result=batch.aggregate(run);r=result['rows'][0]
    assert r['conditions_A']['documented_side']=='FAIL'
    assert r['conditions_B']['sequence_allow'] is True and r['conditions_B']['left_metric_allow'] is False
    assert r['conditions_B']['processed_count_input_gap_policy']=='UNKNOWN'
    assert r['temporal_eligibility'] is False and result['training_readiness']=='NOT_READY'


def test_full_112_denominator_includes_unprocessed_and_incomplete(case,monkeypatch):
    run,plan,row,attempt=case
    rows=[]
    for i in range(112):
        copy=json.loads(json.dumps(row))
        if i:copy['source']['video_id']=f'synthetic-not-processed-{i}'
        rows.append(copy)
    plan['videos']=rows
    monkeypatch.setattr(batch,'source_check',lambda source:None)
    result=batch.aggregate(run)
    assert result['video_denominator']==len(result['rows'])==112
    assert result['cohorts']['fresh_strict_unknown']['states']=={'failed_or_incomplete':1,'not_processed':111}
    assert result['cohorts']['fresh_strict_unknown']['processed_clips']==0


def test_strict_and_exploratory_cohorts_separate(case):
    run,plan,row,attempt=case
    assert batch.worker(run,row['source']['video_id'],attempt)==0
    for side in (Viewpoint.SIDE,Viewpoint.UNKNOWN):
        folder=run/('reuse-'+side.value);folder.mkdir()
        frames=[make_frame(i,viewpoint=side) for i in range(6)]
        q=batch.quality(frames,plan['configuration'])
        batch.write_json(folder/'preprocessing.json',dict(quality=q.model_dump(mode='json')))
        saved=dict(directory=str(folder),files={'preprocessing.json':batch.file_info(folder/'preprocessing.json')})
        plan['videos'].append(dict(source=row['source'],action='reuse',snapshot=saved))
    result=batch.aggregate(run)
    assert {k:v['clips'] for k,v in result['cohorts'].items()}=={
        'fresh_strict_unknown':1,'reused_exploratory_side':1,'reused_strict_unknown':1}
    assert result['conditions_A_pass']==0
    assert result['conditions_B_quality_subset_pass']==1  # Synthetic hypothetical row only.
    assert all(r['conditions_B']['overall']=='BLOCKED' for r in result['rows'])
