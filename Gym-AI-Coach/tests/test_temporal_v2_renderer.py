"""Renderer execution/parity checks, not a browser or visual-layout audit."""
import io
import json
import re
import shutil
import subprocess

import pytest

from gym_coach import axis_geometry, geometry
from gym_coach.reporting.temporal_v2_renderer import render_temporal_v2
from gym_coach.segmentation.temporal_v2 import phase_at_v2, segment_squat_v2
from test_temporal_v2 import contract_example, source


@pytest.mark.parametrize('mode',['dwell_candidate','point_turn_at_resolution','bounds_unresolved','gap','empty'])
def test_javascript_projection_matches_python_without_browser(mode,tmp_path):
    node=shutil.which('node')
    if node is None:
        pytest.skip('Optional local renderer QA requires an already installed Node; no install performed')
    if mode in ('dwell_candidate','point_turn_at_resolution'):
        result=contract_example(mode)  # explicit output-contract acceptance, never producer input
    elif mode=='gap':
        result=segment_squat_v2(source(gap=[60]))
    elif mode=='empty':
        result=segment_squat_v2(source(end=0))
    else:
        result=segment_squat_v2(source())
    markup=render_temporal_v2(result)
    payload=re.search(r'<script id="temporal-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    code=re.findall(r'<script>(.*?)</script>',markup,re.S)[0]
    runner=r'''
const fs=require('node:fs'),vm=require('node:vm');
const request=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements={};
const document={getElementById(id){
 if(!elements[id])elements[id]={textContent:'',value:'0',addEventListener(name,fn){this[name]=fn;}};
 return elements[id];
}};
document.getElementById('temporal-data').textContent=request.payload;
const context=vm.createContext({document});
new vm.Script(request.code).runInContext(context);
const frames=[];
for(let i=0;i<request.frames;i++){
 const scrub=document.getElementById('scrub');scrub.value=String(i);scrub.input();
 frames.push({motion:elements.motion.textContent,events:elements.events.textContent,reasons:elements.reasons.textContent});
}
console.log(JSON.stringify({frames,empty:elements.motion.textContent}));
'''
    request=tmp_path/'request.json';script=tmp_path/'qa.cjs'
    request.write_text(json.dumps(dict(code=code,payload=payload,frames=len(result.legacy.input.samples))),encoding='utf-8')
    script.write_text(runner,encoding='utf-8')
    completed=subprocess.run([node,str(script),str(request)],capture_output=True,text=True,encoding='utf-8')
    assert completed.returncode==0,completed.stderr
    actual=json.loads(completed.stdout)
    if mode=='empty':
        assert actual['empty']=='Unavailable — empty input'
    for i,rendered in enumerate(actual['frames']):
        expected=phase_at_v2(result,i)
        assert rendered['motion']==(expected.motion_phase or f'Unavailable ({expected.motion_status})')
        for event in expected.events:
            assert event.kind in rendered['events']
        for role in expected.event_roles:
            assert f'{role.cycle_id}: {role.role}' in rendered['events']
        for reason in expected.reasons:
            assert reason.reason in rendered['reasons']


def test_no_data_io_or_angle_computation_in_slice(monkeypatch):
    supplied=source(dwell=True)
    def forbidden(*args,**kwargs):
        raise AssertionError('Geometry calculation is outside this slice')
    monkeypatch.setattr(geometry,'joint_angle',forbidden)
    monkeypatch.setattr(axis_geometry,'midpoint_axis_angle_vertical',forbidden)
    original=io.open
    def guarded(file,*args,**kwargs):
        if 'data' in str(file).replace('\\','/').lower().split('/'):
            raise AssertionError('No dataset/metadata access')
        return original(file,*args,**kwargs)
    monkeypatch.setattr(io,'open',guarded)
    result=segment_squat_v2(supplied)
    assert 'Unavailable' in render_temporal_v2(result)
