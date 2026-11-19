"""Bridge tests: nhãn JSON v2, legacy boundary list và pose-cache."""
import json
import numpy as np
import pytest
from stage3_features.phase_model.support.phase_common import payload_boundaries,load_pose_clip,make_payload

def test_dict_phases_convert_to_half_open_boundaries():
    payload={'schema_version':2,'fps':30,'total_frames':20,'phases':{'vao_cho':{'start_frame':3,'end_frame':4},'static':{'start_frame':5,'end_frame':6},'sprint_start':{'start_frame':7,'end_frame':8},'drive_phase':{'start_frame':9,'end_frame':10}}}
    assert payload_boundaries(payload)==[3,5,7,9,11]

def test_absent_phase_uses_repeated_boundary():
    payload={'fps':30,'total_frames':20,'phases':{'vao_cho':{'absent':True},'static':{'start_frame':3,'end_frame':4},'sprint_start':{'start_frame':5,'end_frame':6},'drive_phase':{'start_frame':7,'end_frame':8}}}
    assert payload_boundaries(payload)==[3,3,5,7,9]

def test_gap_and_out_of_range_are_rejected():
    phases={'vao_cho':{'start_frame':1,'end_frame':2},'static':{'start_frame':4,'end_frame':5},'sprint_start':{'start_frame':6,'end_frame':7},'drive_phase':{'start_frame':8,'end_frame':9}}
    with pytest.raises(ValueError):payload_boundaries({'fps':30,'total_frames':12,'phases':phases})
    good={'vao_cho':{'start_frame':1,'end_frame':2},'static':{'start_frame':3,'end_frame':4},'sprint_start':{'start_frame':5,'end_frame':6},'drive_phase':{'start_frame':7,'end_frame':8}}
    with pytest.raises(ValueError):payload_boundaries({'fps':30,'total_frames':8,'phases':good})

def test_legacy_boundaries_and_schema_v2_payload():
    payload=make_payload('x.mp4',30,20,[1,3,5,7,9],'auto')
    assert payload['schema_version']==2
    assert payload_boundaries({'fps':30,'total_frames':20,'boundaries_frames':[1,3,5,7,9]})==[1,3,5,7,9]

def test_load_json_pose_cache_keeps_nan_placeholders(tmp_path):
    frames=[]
    for i in range(2):
        xy=np.zeros((17,2),dtype=float).tolist();conf=[1.0]*17
        if i==1:xy[4]=[float('nan'),float('nan')];conf[4]=0.0
        frames.append({'frame_idx':i,'timestamp_s':i/30,'xy':xy,'conf':conf})
    path=tmp_path/'clip.json';path.write_text(json.dumps({'fps':30,'frames':frames}),encoding='utf-8')
    poses,fps=load_pose_clip('clip',tmp_path)
    assert poses.shape==(2,17,3) and fps==30
    assert np.isnan(poses[1,4,:2]).all()
