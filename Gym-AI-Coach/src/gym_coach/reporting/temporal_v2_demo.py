"""Exclusive-output synthetic v2 demo; never reads dataset or fixture references.

Run: python -m gym_coach.reporting.temporal_v2_demo --output-dir <new-directory>
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from ..contracts import JointProvenance
from ..flags import FlagCode
from ..segmentation.models import SquatSignalSample, SquatTemporalInput, strict_json
from ..segmentation.temporal_v2 import segment_squat_v2
from ..segmentation.temporal_v2_models import SquatTemporalResultV2, read_temporal_json
from .temporal_v2_renderer import render_temporal_v2


def synthetic_input(case: str, fps: float) -> SquatTemporalInput:
    """Measured input samples only; no cycle/arrival/phase reference labels."""
    samples = []
    end = round(2.*fps)+1 if case == 'clip_ends_in_pause' else round(6.*fps)+1
    for i in range(end):
        t = i/fps
        value = 140.-30.*math.cos(math.pi*t)
        if case in ('dwell_signal', 'clip_ends_in_pause') and any(abs(t-b) <= .2 for b in (2., 4.)):
            value = 110.
        missing = case == 'gap' and i == round(2.*fps)
        samples.append(SquatSignalSample(frame_index=i, timestamp_seconds=t, value=None if missing else value,
            side_source='right' if case == 'side_switch' and t >= 3. else 'left',
            joint_provenance=(JointProvenance.MISSING,)*3 if missing else (JointProvenance.OBSERVED,)*3,
            reasons=(FlagCode.MISSING_JOINT,) if missing else ()))
    raw = json.dumps([s.model_dump(mode='json') for s in samples], sort_keys=True, allow_nan=False).encode()
    return SquatTemporalInput(video_id=f'synthetic-v2-{case}-{int(fps)}fps', fps=fps,
        source_hash=hashlib.sha256(raw).hexdigest(), samples=tuple(samples))


def write_demo(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    def write(path, value):
        with (output_dir/path).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(value)
    def digest(path):
        return hashlib.sha256((output_dir/path).read_bytes()).hexdigest()
    cases = []
    for case, fps in [('dwell_signal',25.),('dwell_signal',30.),('dwell_signal',60.),
                      ('point_turn_signal',30.),('gap',30.),('side_switch',30.),('clip_ends_in_pause',30.)]:
        key = f'{case}-{int(fps)}fps'
        folder = Path(key)
        (output_dir/folder).mkdir()
        supplied = synthetic_input(case, fps)
        result = segment_squat_v2(supplied)
        input_path, result_path, html_path = (folder/p for p in ('input.json','result.json','index.html'))
        write(input_path, strict_json(supplied))
        write(result_path, strict_json(result))
        assert read_temporal_json((output_dir/result_path).read_text(encoding='utf-8')) == result
        write(html_path, render_temporal_v2(result))
        cases.append(dict(case=key, frames=len(supplied.samples), fps=fps,
            candidate_count=result.candidate_count, bounds_modes=[c.bottom.mode for c in result.cycles],
            pause_candidates=sum(c.bottom.pause.status=='candidate' for c in result.cycles),
            phase_candidates=sum(c.motion.status=='candidate' for c in result.cycles),
            input_path=input_path.as_posix(), input_sha256=digest(input_path),
            result_path=result_path.as_posix(),result_sha256=digest(result_path),
            html_path=html_path.as_posix(),html_sha256=digest(html_path),
            profile_hash=result.legacy.profile_hash,policy_hash=result.policy_hash))
    write(Path('schema-v2.json'),json.dumps(SquatTemporalResultV2.model_json_schema(),indent=2,ensure_ascii=False))
    manifest = dict(demo_version='squat-temporal-v2-demo-v1',synthetic_only=True,
        reference_labels_consumed=False,angle_calculation=False,official_eligibility=False,assessment_enabled=False,
        validation_status='NOT_VALIDATED',official_gate='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',
        python=sys.version,platform=platform.platform(),cases=cases)
    write(Path('manifest.json'),json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False))
    links = ''.join(f'<li><a href="{e["html_path"]}">{e["case"]}</a>: {e["candidate_count"]} candidate(s); bounds/phase/pause unavailable</li>' for e in cases)
    write(Path('index.html'),'<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic temporal v2</title><h1>EXPERIMENTAL / NOT VALIDATED</h1><p>G5.7 BLOCKED · G6 IN_PROGRESS · official=false · assessment=false</p><p>Anchor-only producer. Dwell and point-turn names describe generated input signals, not detected modes. No angles calculated.</p><ul>'+links+'</ul></html>')
    lines=['# Synthetic temporal v2 demo','', 'NOT_VALIDATED; G5.7 BLOCKED; G6 IN_PROGRESS; assessment=false.',
        'No fixture reference labels, real data or geometry calculation. All detected modes: bounds_unresolved.',
        '', '| Input | FPS | Frames | PVP candidates | Resolved phase/pause |','|---|---:|---:|---:|---|']
    lines.extend(f'| {e["case"]} | {e["fps"]} | {e["frames"]} | {e["candidate_count"]} | unavailable |' for e in cases)
    write(Path('report.md'),'\n'.join(lines)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    manifest=write_demo(args.output_dir)
    print(json.dumps({'output_dir':str(args.output_dir),'cases':len(manifest['cases']),
                      'status':'EXPERIMENTAL / NOT VALIDATED'},ensure_ascii=False))


if __name__ == '__main__':
    main()
