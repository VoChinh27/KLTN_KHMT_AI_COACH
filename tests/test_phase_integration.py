"""Integration contracts for schema, lazy decoder posteriors and review gating."""
import numpy as np
from core.phase_schema import PhaseSet,Rep,PhaseWindow,save_phases,load_phases
from stage3_features.phase_model.core.phase_decoder import PhaseDecoder
from core.phase_resolver import resolve_phases

def test_derived_compatibility_fields_are_not_reemitted_but_unknown_metadata_survives():
    data={'schema_version':1,'video':'x.mp4','fps':30,'total_frames':20,'n_frames':20,'boundaries_frames':[1,3,5,7,9],'front_leg_override':'left','phases':{'vao_cho':{'start_frame':1,'end_frame':2},'static':{'start_frame':3,'end_frame':4},'sprint_start':{'start_frame':5,'end_frame':6},'drive_phase':{'start_frame':7,'end_frame':8}}}
    ps=PhaseSet.from_dict(data);out=ps.to_dict()
    assert 'n_frames' not in out and 'boundaries_frames' not in out
    assert out['front_leg_override']=='left' and out['schema_version']==2

def test_decoder_defers_forward_backward_until_confidence_is_requested():
    p=np.full((48,6),0.01);labels=[0]*8+[1]*8+[2]*8+[3]*8+[4]*8+[5]*8
    for i,label in enumerate(labels):p[i,label]=0.95
    result=PhaseDecoder(skip_penalty=1.0,emission_scale=1.0).decode(p,30,compute_posteriors=False)
    assert result.success and result.start_post is None
    result.boundary_confidence(tol=2)
    assert result.start_post is not None

def test_manual_phases_are_selected_without_review(tmp_path):
    rep=Rep.from_windows({'vao_cho':(1,3),'static':(4,7),'sprint_start':(8,9),'drive_phase':(10,15)})
    ps=PhaseSet(video='coach_HLV01.mp4',fps=30,total_frames=20,reps=[rep],marked_by='manual')
    path=tmp_path/'coach_HLV01.phases.json';save_phases(path,ps)
    windows,source,needs=resolve_phases('coach_HLV01.mp4',np.zeros((20,17,3)),30,{'phases_dir':str(tmp_path)},manual_windows={'vao_cho':(1,3)},manual_source=f'manual:{path}')
    assert source.startswith('manual:') and not needs and windows=={'vao_cho':(1,3)}
