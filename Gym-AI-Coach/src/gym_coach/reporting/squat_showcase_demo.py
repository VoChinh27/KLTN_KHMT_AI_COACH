"""Assemble saved synthetic JSON artifacts; never call a pipeline/producer.

python -m gym_coach.reporting.squat_showcase_demo --sources <selection.json> --output-dir <new-dir>
All selected inputs must resolve beneath the explicit workspace outputs/ root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import platform
import sys

from ..segmentation.models import SquatTemporalInput
from ..segmentation.point_resolution_models import read_point_resolution_json
from .squat_models import SquatFeedbackResult
from .squat_showcase import build_showcase_payload,presentation_error,render_showcase


def _json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError(f'Duplicate JSON key: {key}')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda token:(_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {token}')))


def _artifact(workspace,path,expected=None):
    root=(workspace/'outputs').resolve()
    selected=Path(path)
    selected=(workspace/selected).resolve() if not selected.is_absolute() else selected.resolve()
    if not selected.is_relative_to(root) or selected.suffix.lower()!='.json':
        raise ValueError('Only saved JSON beneath workspace outputs/ can be read; Data is forbidden')
    raw=selected.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and (not re.fullmatch(r'[a-f0-9]{64}',expected) or digest!=expected):
        raise ValueError(f'Artifact byte hash mismatch: {selected.name}')
    _json(raw)  # Reject nonfinite values/duplicate keys before model validation.
    return selected,raw,digest


def load_selection(selection_path,*,workspace=None):
    workspace=Path(workspace or Path.cwd()).resolve()
    try:
        selection_file,selection_raw,selection_hash=_artifact(workspace,selection_path)
        config=_json(selection_raw)
        if config.get('selection_version')!='squat-showcase-sources-v1':
            raise ValueError('Unsupported showcase selection version')
        feedbacks=[];points=[];snapshots=[];refs=[]
        def retain(role,label,path,raw,digest):
            target=f'sources/{role}.json'
            snapshots.append((target,raw))
            refs.append(dict(role=role,label=label,snapshot_path=target,sha256=digest,
                original_relative_path=path.relative_to(workspace).as_posix()))
        for i,entry in enumerate(config['feedback']):
            path,raw,digest=_artifact(workspace,entry['path'],entry['sha256'])
            saved=SquatFeedbackResult.model_validate_json(raw,strict=True)
            if saved.source.kind!='synthetic_contract':raise ValueError('Real preview is outside this synthetic showcase')
            feedbacks.append(saved);retain(f'feedback-{i}',entry['label'],path,raw,digest)
        entry=config['point_manifest']
        manifest_path,manifest_raw,manifest_hash=_artifact(workspace,entry['path'],entry['sha256'])
        manifest=_json(manifest_raw)
        if (manifest.get('demo_version')!='squat-point-resolution-demo-v1'
                or manifest.get('synthetic_only') is not True or manifest.get('official_eligibility') is not False
                or manifest.get('assessment_enabled') is not False or manifest.get('validation_status')!='NOT_VALIDATED'):
            raise ValueError('Expected explicit unvalidated synthetic point-resolution manifest')
        for i,case in enumerate(manifest['cases']):
            candidate=Path(case['result_path'])
            input_file=Path(case['input_path'])
            if (candidate.is_absolute() or input_file.is_absolute() or '..' in candidate.parts or '..' in input_file.parts):
                raise ValueError('Point manifest paths cannot escape their run')
            path,raw,digest=_artifact(workspace,manifest_path.parent/candidate,case['result_sha256'])
            saved=read_point_resolution_json(raw.decode('utf-8'))
            _,input_raw,_=_artifact(workspace,manifest_path.parent/input_file,case['input_sha256'])
            original_input=SquatTemporalInput.model_validate_json(input_raw,strict=True)
            if original_input!=saved.base_result.legacy.input:
                raise ValueError('Point manifest input is not the exact input linked by audit')
            if (case['frames']!=len(original_input.samples) or case['fps']!=original_input.fps
                    or case['accepted_point_count']!=saved.accepted_point_count
                    or case['pvp_candidate_count']!=saved.base_result.candidate_count
                    or case['necessary_condition_count']!=saved.necessary_condition_count
                    or case['policy_hash']!=saved.policy_hash or case['profile_hash']!=saved.base_result.legacy.profile_hash):
                raise ValueError('Point manifest summary/linkage mismatch')
            points.append(saved);retain(f'point-{i}',case['case'],path,raw,digest)
        entry=config['noise_review']
        path,raw,digest=_artifact(workspace,entry['path'],entry['sha256'])
        noise=_json(raw)
        if (noise.get('review_version')!='squat-showcase-noise-review-v1'
                or noise.get('independent_sample_noise_evidence_found') is not False
                or not isinstance(noise.get('summary'),str)):
            raise ValueError('This showcase requires the scoped read-only noise review, never an override')
        retain('noise-review','Read-only noise evidence review',path,raw,digest)
        if not feedbacks or not points:raise ValueError('Both panels require nonempty saved synthetic cases')
        return dict(feedbacks=tuple(feedbacks),points=tuple(points),snapshots=snapshots,refs=refs,noise=noise,
            selection_raw=selection_raw,selection_hash=selection_hash,
            point_manifest_sha256=manifest_hash,point_manifest_original=manifest_path.relative_to(workspace).as_posix())
    except (ValueError,TypeError,AttributeError,KeyError,OSError) as exc:
        raise presentation_error(f'Cannot load showcase selection: {exc}') from exc


def write_showcase(selection_path,output_dir,*,workspace=None):
    workspace=Path(workspace or Path.cwd()).resolve()
    output=Path(output_dir)
    output=(workspace/output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to((workspace/'outputs').resolve()):
        raise presentation_error('Showcase output must be a new directory beneath workspace outputs/')
    if output.exists():raise FileExistsError(f'Refusing existing showcase output: {output}')
    loaded=load_selection(selection_path,workspace=workspace)
    kwargs=dict(source_refs=loaded['refs'],noise_review=loaded['noise'])
    payload=build_showcase_payload(loaded['feedbacks'],loaded['points'],**kwargs)
    markup=render_showcase(loaded['feedbacks'],loaded['points'],**kwargs)
    output.mkdir(parents=True,exist_ok=False)
    (output/'sources').mkdir()
    def write(name,raw):
        with (output/name).open('xb') as stream:stream.write(raw)
    for target,raw in loaded['snapshots']:write(target,raw)
    write('selection.json',loaded['selection_raw'])
    write('noise-review.json',json.dumps(loaded['noise'],ensure_ascii=False,indent=2,allow_nan=False).encode())
    write('presentation.json',json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    write('index.html',markup.encode('utf-8'))
    files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in output.rglob('*') if p.is_file()}
    manifest=dict(showcase_version='gym-squat-showcase-v1',synthetic_only=True,new_inference=False,
        new_geometry=False,new_detector=False,official_eligibility=False,assessment_enabled=False,
        validation_status='NOT_VALIDATED',g57='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',g6='IN_PROGRESS',
        linkage=payload['linkage'],feedback_cases=len(loaded['feedbacks']),point_cases=len(loaded['points']),
        selection_sha256=loaded['selection_hash'],point_manifest_sha256=loaded['point_manifest_sha256'],
        point_manifest_original=loaded['point_manifest_original'],source_artifacts=loaded['refs'],files=files,
        python=sys.version,platform=platform.platform())
    write('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode())
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    result=write_showcase(args.sources,args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir),feedback_cases=result['feedback_cases'],
        point_cases=result['point_cases'],linkage='independent_panels',status='NOT_VALIDATED'),ensure_ascii=False))


if __name__=='__main__':main()
