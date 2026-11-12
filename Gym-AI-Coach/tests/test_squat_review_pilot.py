"""Synthetic selection/linkage tests; no video access or label generation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from test_squat_counting import source, P

spec = importlib.util.spec_from_file_location('review_pilot', Path(__file__).resolve().parents[1]/'scripts/prepare_squat_review_pilot.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fixture():
    sources = [source(g, c).model_dump(mode='json') for g in range(1, 8) for c in (1, 2)]
    manifest = dict(version='squat-count-dataset-v1', videos=[dict(source=s) for s in sources])
    split = dict(version='squat-count-split-proposal-v1', dataset_manifest_sha256=P,
        assignments=[dict(**{k:s[k] for k in ('video_id','path','sha256','group_id')},
                          role='validation' if s['group_id']=='g01' else 'train') for s in sources])
    rows = [dict(source=s, cohort='fresh_strict_unknown', status='completed',
        source_camera_view='UNKNOWN', quality_viewpoint='unknown',
        sequence=dict(eligible=True, action='allow'),
        metrics={review.LEFT:dict(eligible=False,action='skip',required_joints=review.LEFT_JOINTS[:],reasons=['viewpoint_unknown'])}) for s in sources]
    summary = dict(version='squat-pose-feasibility-audit-v1',video_denominator=len(rows),rows=rows)
    return manifest, split, summary


def test_deterministic_pilot_distinct_train_groups_no_quality_dependency():
    manifest, split, summary = fixture()
    chosen = review.select_pilot(manifest, split, P)
    assert [s['video_id'] for s in chosen] == [source(g).video_id for g in range(2,7)]
    manifest['videos'].reverse(); split['assignments'].reverse()
    for r in manifest['videos']:
        r.update(quality='skip', prediction=999, viewpoint='side')
    assert review.select_pilot(manifest, split, P)==chosen
    assert len(manifest['videos'])==14  # Parent cohort never filtered.


@pytest.mark.parametrize('mutation', ['hash','role','duplicate','missing','insufficient_groups','group_leak'])
def test_pilot_rejects_invalid_split(mutation):
    manifest, split, _ = fixture()
    if mutation=='hash':split['assignments'][0]['sha256']='0'*64
    elif mutation=='role':split['assignments'][0]['role']='test'
    elif mutation=='duplicate':split['assignments'][-1]=deepcopy(split['assignments'][0])
    elif mutation=='missing':split['assignments'].pop()
    elif mutation=='insufficient_groups':
        for r in split['assignments']:r['role']='validation'
    else:split['assignments'][1]['role']='train'
    with pytest.raises(ValueError):review.select_pilot(manifest, split, P)


def test_split_manifest_linkage():
    manifest, split, _ = fixture()
    with pytest.raises(ValueError):review.select_pilot(manifest, split, '0'*64)


@pytest.mark.parametrize('case', ['exploratory','side','degrade_sequence','extra_reason','allow_metric','missing_metric','failed'])
def test_viewpoint_selection_exact_and_conservative(case):
    manifest, _, summary = fixture()
    row=summary['rows'][0]
    if case=='exploratory':row['cohort']='reused_exploratory_side'
    elif case=='side':row['quality_viewpoint']='side'
    elif case=='degrade_sequence':row['sequence']['action']='degrade'
    elif case=='extra_reason':row['metrics'][review.LEFT]['reasons'].append('low_pose_visibility')
    elif case=='allow_metric':row['metrics'][review.LEFT].update(action='allow',eligible=True)
    elif case=='missing_metric':row['metrics']={}
    else:row['status']='failed'
    selected=review.select_viewpoint(summary,manifest)
    assert len(selected)==13 and row['source']['video_id'] not in {r['source']['video_id'] for r in selected}


@pytest.mark.parametrize('mutation', ['joint','source_hash','time','missing','duplicate','version'])
def test_viewpoint_rejects_linkage_and_canonical_joint_corruption(mutation):
    manifest, _, summary = fixture()
    summary=deepcopy(summary)
    if mutation=='joint':summary['rows'][0]['metrics'][review.LEFT]['required_joints'][0]='right_hip'
    elif mutation=='source_hash':summary['rows'][0]['source']['sha256']='0'*64
    elif mutation=='time':summary['rows'][0]['source']['fps']=30.0
    elif mutation=='missing':summary['rows'].pop()
    elif mutation=='duplicate':summary['rows'][-1]=deepcopy(summary['rows'][0])
    else:summary['version']='unknown'
    with pytest.raises(ValueError):review.select_viewpoint(summary,manifest)


def test_pending_whole_clip_json_preserves_no_human_claims_and_input():
    manifest, _, summary=fixture(); before=deepcopy(summary)
    result=review.select_viewpoint(summary,manifest)
    assert result==json.loads(json.dumps(result)) and summary==before
    for row in result:
        assert row['status']=='VIEWPOINT_REVIEW_PENDING' and row['documented'] is False
        assert all(row[k] is None for k in ('reviewer','reviewed_at','protocol_reference',
            'declared_whole_clip_viewpoint','ai_visual_assessment'))
        assert row['reviewed_intervals']==[]
        assert row['review_scope']['inclusive_end_frame']==250
        assert row['review_scope']['last_sample_seconds']==10.0
        assert row['review_scope']['container_duration_seconds']==10.04


def test_inventory_rejects_tamper_and_path_escape(tmp_path):
    p=tmp_path/'source.json';p.write_text('{}')
    identity=review.file_info(p)
    index=tmp_path/'inventory.json'
    index.write_text(json.dumps(dict(files={'source.json':identity})))
    assert review.verify_inventory(tmp_path,index.name)==1
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):review.verify_inventory(tmp_path,index.name)
    index.write_text(json.dumps(dict(files={'../outside.json':identity})))
    with pytest.raises(ValueError):review.verify_inventory(tmp_path,index.name)


def test_canonical_left_joint_definition_matches_existing_registry():
    from gym_coach.features import GEOMETRY_REGISTRY
    assert review.LEFT_JOINTS==[j.value for j in GEOMETRY_REGISTRY.get(review.LEFT).required_joints]
