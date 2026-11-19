"""Selection linkage regression and readback of the explicitly supplied fresh run."""

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from gym_coach.features import GEOMETRY_REGISTRY, GeometryFeatureObservation
from gym_coach.features.models import content_hash
from gym_coach.flags import GymError
from gym_coach.preprocessing import read_preprocessing_result
from gym_coach.preprocessing.models import content_hash as source_hash
from gym_coach.quality import QualityReport

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage5_angle_audit',ROOT/'scripts/audit_stage5_angles.py')
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fake_selection(root):
    (root/'clip.avi').write_bytes(b'video placeholder, no inference in selection test')
    video_hash=audit.sha(root/'clip.avi')
    (root/'sample.png').write_bytes(b'placeholder image for linkage-only unit test')
    samples=[dict(frame_index=i,path='sample.png',sha256=audit.sha(root/'sample.png'),agent_viewed=True) for i in (0,5)]
    audit.write_json(root/'visual.json',dict(candidates=[dict(review_id='T1',video_id='clip.avi',path='clip.avi',sha256=video_hash,
        assessment=dict(viewpoint='likely_side'),samples=samples)]))
    data=dict(selection_version='development-angle-audit-v1',inference_started=False,candidates=[dict(
        review_id='T1',path='clip.avi',video_id='clip.avi',sha256=video_hash,exercise='BodyWeightSquats',
        exploratory_viewpoint='side',evidence_status='visually_assessed_unverified',documented=False,
        visual_evidence_path='visual.json',visual_evidence_sha256=audit.sha(root/'visual.json'))])
    audit.write_json(root/'selection.json',data)
    return data


@pytest.mark.parametrize('fault',['selection_hash','video_hash','video_id','escape','evidence_hash','evidence_link','documented','duplicate','exercise','viewpoint','visual_viewpoint','sample_hash','unviewed_sample'])
def test_preinference_selection_rejects_inconsistent_links(tmp_path,fault):
    data=fake_selection(tmp_path)
    c=data['candidates'][0]
    if fault=='selection_hash': expected='0'*64
    else:
        if fault=='video_hash': c['sha256']='0'*64
        if fault=='video_id': c['video_id']='other.avi'
        if fault=='escape': c['path']='../clip.avi'
        if fault=='evidence_hash': c['visual_evidence_sha256']='0'*64
        if fault=='evidence_link': c['review_id']='missing'
        if fault=='documented': c['documented']=True
        if fault=='duplicate': data['candidates'].append(dict(c))
        if fault=='exercise': c['exercise']='other'
        if fault=='viewpoint': c['exploratory_viewpoint']='other'
        if fault=='visual_viewpoint': c['exploratory_viewpoint']='front'
        if fault in ('sample_hash','unviewed_sample'):
            evidence=json.loads((tmp_path/'visual.json').read_text('utf-8'))
            sample=evidence['candidates'][0]['samples'][0]
            if fault=='sample_hash': sample['sha256']='0'*64
            else: sample['agent_viewed']=False
            (tmp_path/'visual.json').write_text(json.dumps(evidence),'utf-8')
            c['visual_evidence_sha256']=audit.sha(tmp_path/'visual.json')
        (tmp_path/'selection.json').write_text(json.dumps(data),'utf-8')
        expected=audit.sha(tmp_path/'selection.json')
    with pytest.raises(GymError) as caught: audit.load_selection(tmp_path,tmp_path/'selection.json',expected)
    assert caught.value.flag.code=='feature_context_mismatch'
    assert caught.value.flag.evidence['reason']=='selection_linkage_invalid'


def test_valid_frozen_selection_and_exclusive_outputs(tmp_path):
    data=fake_selection(tmp_path)
    assert audit.load_selection(tmp_path,tmp_path/'selection.json',audit.sha(tmp_path/'selection.json'))==data
    with pytest.raises(FileExistsError): audit.write_json(tmp_path/'selection.json',data)


@pytest.fixture(scope='module')
def development_results():
    value=os.environ.get('GYM_STAGE5_AUDIT_DIR')
    if not value:
        pytest.skip('Set GYM_STAGE5_AUDIT_DIR to an explicitly generated fresh development audit')
    directory=Path(value)
    if not directory.is_absolute(): directory=ROOT/directory
    data=json.loads((directory/'results.json').read_text('utf-8'))
    return directory,data


@pytest.mark.integration
@pytest.mark.parametrize('review_id',['R106','H002','H004'])
@pytest.mark.parametrize('branch',['original_unknown','exploratory'])
def test_fresh_development_pipeline_and_every_outcome(development_results,review_id,branch):
    directory,data=development_results
    manifest=json.loads((directory/'manifest.json').read_text('utf-8'))
    selection=audit.load_selection(ROOT,ROOT/manifest['selection_path'],manifest['selection_sha256'])
    candidate=next(c for c in selection['candidates'] if c['review_id']==review_id)
    row=next(r for r in data['results'] if r['review_id']==review_id and Path(r['output']).name==review_id+'_'+branch)
    assert row['completed'] and row['input_byte_sha256']==candidate['sha256']
    assert row['documented'] is False and row['official_eligibility'] is False
    assert row['source_metadata_viewpoint']=='unknown'
    assert not row['temporal_analysis_eligible'] and row['assessment_status']=='blocked_pending_expert_validation'
    output=ROOT/row['output']
    for name,expected in row['artifact_byte_hashes'].items(): assert audit.sha(output/name)==expected
    result=read_preprocessing_result(output/'preprocessing.json')
    report=QualityReport.model_validate_json((output/'quality.json').read_text('utf-8'))
    assert result.quality==report and content_hash(report)==row['quality_report_hash']
    assert source_hash(result)==row['preprocessing_result_hash']
    links={v['frame_index']:v for v in (json.loads(line) for line in (output/'frame_links.jsonl').read_text('utf-8').splitlines())}
    assert len(links)==row['decoded_frames']==len(result.frames)
    counts={f.feature_id:Counter() for f in GEOMETRY_REGISTRY.for_exercise(candidate['exercise'])}
    reason_counts={key:Counter() for key in counts}
    with (output/'features.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            obs=GeometryFeatureObservation.model_validate_json(line,strict=True)
            assert obs.availability.input.frame==result.frames[obs.frame_index]
            assert obs.availability.input.quality==report
            assert obs.availability.input_hash==links[obs.frame_index]['measurement_input_hash']
            assert obs.timestamp_seconds==result.raw_frames[obs.frame_index].timestamp_seconds
            metric=report.metric_eligibility[obs.feature_id]
            assert tuple(metric.required_joints)==GEOMETRY_REGISTRY.get(obs.feature_id).required_joints
            assert not obs.temporal_analysis_eligible and obs.measurement_validation_status=='unvalidated'
            counts[obs.feature_id][obs.status]+=1
            reason_counts[obs.feature_id].update(obs.reasons)
            if obs.value is not None: assert metric.eligible and metric.action=='allow' and branch=='exploratory'
            else: assert obs.flags and obs.reasons
    for key,count in counts.items():
        assert count['available']+count['unavailable']==row['decoded_frames']
        assert count['available']==row['feature_outcomes'][key]['available']
        assert dict(reason_counts[key])==row['feature_outcomes'][key]['reasons']
    assert row['sequence_status']==report.sequence_status and row['sequence_decision']==report.eligibility.model_dump(mode='json')
    if branch=='original_unknown' or review_id=='H002': assert all(c['available']==0 for c in counts.values())
