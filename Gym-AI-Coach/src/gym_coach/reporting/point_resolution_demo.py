"""Exclusive-output synthetic point-resolution audit/demo (no positive claims).

python -m gym_coach.reporting.point_resolution_demo --output-dir <new-directory>
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import sys

from ..contracts import JointProvenance
from ..flags import FlagCode
from ..preprocessing.models import InterpolationSource
from ..segmentation.models import SquatSignalSample,SquatTemporalInput,strict_json
from ..segmentation.point_resolution import segment_squat_point_resolution
from ..segmentation.point_resolution_models import SquatPointResolutionAudit,read_point_resolution_json
from .temporal_v2_renderer import render_temporal_v2


def render_point_audit(result: SquatPointResolutionAudit) -> str:
    if not isinstance(result,SquatPointResolutionAudit):
        raise ValueError('Explicit point-resolution envelope required')
    result=SquatPointResolutionAudit.model_validate_json(strict_json(result),strict=True)
    rows=[]
    for check in result.checks:
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            check.bottom_frame,f'{check.raw_support.start_frame}..{check.raw_support.end_frame}',
            check.strict_local_minimum,check.necessary_conditions_met,'Unavailable',
            '; '.join(r.reason for r in check.reasons)))+'</tr>')
    if not rows:
        rows=['<tr><td colspan="6">Unavailable — no_full_pvp_candidate</td></tr>']
    identity=json.dumps(dict(policy=result.policy.model_dump(mode='json'),policy_hash=result.policy_hash,
        base_hash=result.base_hash,input_hash=result.base_result.legacy.input_hash,
        profile_hash=result.base_result.legacy.profile_hash),indent=2)
    diagnostics=json.dumps(dict(run_reasons=[r.model_dump(mode='json') for r in result.base_result.diagnostics],
        original_flags=[f.model_dump(mode='json') for f in result.base_result.legacy.flags]),indent=2,ensure_ascii=False)
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Point-resolution evidence audit</title><style>
body{font:16px system-ui;background:#101820;color:#eef4f6;max-width:1100px;margin:2rem auto;padding:1rem}
table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #61747f;padding:.6rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{color:#ffd17d}a{color:#9bceff}
</style><h1>Squat · point-resolution evidence audit</h1><p class="warning">EXPERIMENTAL / NOT VALIDATED · SYNTHETIC ONLY</p>
<p>G5.7 BLOCKED · G6 IN_PROGRESS · official_eligibility=false · assessment.enabled=false</p>
<p>A raw strict minimum is a necessary sample fact. It is not an accepted point-turn or a validated physical reversal.</p>
<p>Accepted point-turns: 0. Arrival, departure, pause and motion phase: Unavailable (null), never zero pause.</p>
<p>Independent noise evidence is absent; even smooth raw minima abstain. No angles, form score or technical advice.</p>
<table><thead><tr><th>PVP anchor frame</th><th>Raw support (inclusive)</th><th>Strict raw minimum</th><th>Necessary conditions met</th><th>Point-turn</th><th>Reasons</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p><a href="base.html">Open unchanged conservative v2 event/phase renderer</a></p>
<h2>Input / run diagnostics</h2><pre>'''+html.escape(diagnostics)+'''</pre>
<h2>Provenance</h2><pre>'''+html.escape(identity)+'''</pre></html>'''


def synthetic_source(case: str, fps: float) -> SquatTemporalInput:
    """Direct raw scalar generator; no latent history or references are returned."""
    samples=[]
    count=round(2*fps)+1 if case=='clip_edge' else round(6*fps)+1
    for i in range(count):
        t=i/fps;value=140.-30.*math.cos(math.pi*t)
        if case=='plateau' and any(abs(t-b)<=.2 for b in (2.,4.)):
            value=110.
        if case=='jitter' and any(abs(t-b)<=.2 for b in (2.,4.)):
            value+=.04 if i%2 else -.04
        missing=case=='gap' and i==round(2*fps)
        imputed=case=='imputed_support' and i==round(2*fps)-1
        provenance=(JointProvenance.MISSING,)*3 if missing else (JointProvenance.IMPUTED,JointProvenance.OBSERVED,JointProvenance.OBSERVED) if imputed else (JointProvenance.OBSERVED,)*3
        refs=(InterpolationSource(left_frame_index=i-1,right_frame_index=i+1,
            left_timestamp_seconds=(i-1)/fps,right_timestamp_seconds=(i+1)/fps),) if imputed else ()
        samples.append(SquatSignalSample(frame_index=i,timestamp_seconds=t,value=None if missing else value,
            side_source='right' if case=='side_switch' and i>=round(2*fps) else 'left',
            joint_provenance=provenance,imputation_references=refs,
            reasons=(FlagCode.MISSING_JOINT,) if missing else ()))
    digest=hashlib.sha256(json.dumps([s.model_dump(mode='json') for s in samples],sort_keys=True).encode()).hexdigest()
    return SquatTemporalInput(video_id='synthetic-point-resolution-demo',fps=fps,source_hash=digest,samples=tuple(samples))


def write_point_demo(output_dir: Path) -> dict:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    def write(path,text):
        with (output_dir/path).open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(text)
    def digest(path):return hashlib.sha256((output_dir/path).read_bytes()).hexdigest()
    cases=[]
    for case,fps in [('smooth',25.),('smooth',30.),('smooth',60.),('plateau',30.),
                     ('jitter',30.),('gap',30.),('imputed_support',30.),('side_switch',30.),('clip_edge',30.)]:
        folder=Path(f'{case}-{int(fps)}fps');(output_dir/folder).mkdir()
        source=synthetic_source(case,fps);audit=segment_squat_point_resolution(source)
        input_path,result_path,page_path,base_path=(folder/name for name in ('input.json','audit.json','index.html','base.html'))
        write(input_path,strict_json(source));write(result_path,strict_json(audit))
        assert read_point_resolution_json((output_dir/result_path).read_text(encoding='utf-8'))==audit
        write(page_path,render_point_audit(audit));write(base_path,render_temporal_v2(audit.base_result))
        cases.append(dict(case=folder.as_posix(),frames=len(source.samples),fps=fps,
            pvp_candidate_count=audit.base_result.candidate_count,
            necessary_condition_count=audit.necessary_condition_count,accepted_point_count=audit.accepted_point_count,
            reasons=sorted({r.reason for c in audit.checks for r in c.reasons}|{r.reason for r in audit.diagnostics}),
            run_reasons=sorted({r.reason for r in audit.base_result.diagnostics}),
            original_flag_codes=sorted({f.code.value for f in audit.base_result.legacy.flags}),
            input_path=input_path.as_posix(),input_sha256=digest(input_path),
            result_path=result_path.as_posix(),result_sha256=digest(result_path),
            html_path=page_path.as_posix(),html_sha256=digest(page_path),base_path=base_path.as_posix(),base_sha256=digest(base_path),
            policy_hash=audit.policy_hash,profile_hash=audit.base_result.legacy.profile_hash))
    manifest=dict(demo_version='squat-point-resolution-demo-v1',synthetic_only=True,
        reference_labels_consumed=False,accepted_point_count=0,official_eligibility=False,assessment_enabled=False,
        validation_status='NOT_VALIDATED',g57='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',g6='IN_PROGRESS',
        python=sys.version,platform=platform.platform(),cases=cases)
    write(Path('manifest.json'),json.dumps(manifest,indent=2,ensure_ascii=False))
    write(Path('schema.json'),json.dumps(SquatPointResolutionAudit.model_json_schema(),indent=2,ensure_ascii=False))
    links=''.join(f'<li><a href="{c["html_path"]}">{c["case"]}</a>: raw conditions {c["necessary_condition_count"]}/{c["pvp_candidate_count"]}; accepted point-turns 0</li>' for c in cases)
    write(Path('index.html'),'<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic point-resolution audit</title><h1>EXPERIMENTAL / NOT VALIDATED · SYNTHETIC ONLY</h1><p>G5.7 BLOCKED · G6 IN_PROGRESS · official=false · assessment=false</p><p>No accepted point-turn: noise is not identifiable from available evidence. A raw strict-minimum fact is not an accepted event. All arrival/departure/pause estimates remain null.</p><ul>'+links+'</ul></html>')
    lines=['# Point-resolution synthetic audit','', 'No accepted point-turn; NOT_VALIDATED; no real data or geometry calculation.',
        '', '| Input | FPS | Frames | PVP candidates | Raw necessary conditions met | Accepted points | Reasons |',
        '|---|---:|---:|---:|---:|---:|---|']
    lines.extend(f'| {c["case"]} | {c["fps"]} | {c["frames"]} | {c["pvp_candidate_count"]} | {c["necessary_condition_count"]} | 0 | {", ".join(c["reasons"])} |' for c in cases)
    write(Path('report.md'),'\n'.join(lines)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    manifest=write_point_demo(args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir),cases=len(manifest['cases']),
        accepted_point_count=manifest['accepted_point_count'],status='NOT_VALIDATED'),ensure_ascii=False))


if __name__=='__main__':
    main()
