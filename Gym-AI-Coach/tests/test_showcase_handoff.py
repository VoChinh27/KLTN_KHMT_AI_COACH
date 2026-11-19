"""Offline handoff checks; DOM/JS checks are not browser or native keyboard QA."""
import hashlib
import copy
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit

import pytest

from gym_coach import geometry,axis_geometry
from gym_coach.flags import GymError
from gym_coach.reporting import squat_showcase_demo
from gym_coach.reporting.showcase_handoff import read_saved_bundle,write_handoff
from gym_coach.reporting.squat_showcase import render_showcase
from gym_coach.segmentation import squat,temporal_v2,point_resolution

ORIGINAL=Path(__file__).resolve().parents[1]/'outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/demo-final'


@pytest.fixture(scope='module')
def packaged(tmp_path_factory):
    workspace=tmp_path_factory.mktemp('handoff-workspace')
    source=workspace/'outputs'/'saved-bundle';source.parent.mkdir()
    shutil.copytree(ORIGINAL,source)
    output=workspace/'outputs'/'handoff-new'
    manifest=write_handoff(source,output,workspace=workspace)
    return workspace,source,output,manifest


class Structure(HTMLParser):
    def __init__(self):super().__init__();self.nodes=[]
    def handle_starttag(self,tag,attrs):self.nodes.append((tag,dict(attrs)))


def test_bundle_portable_uses_only_snapshots_and_never_audits_or_runs_pipeline(packaged,monkeypatch):
    workspace,source,_,_=packaged
    def forbidden(*args,**kwargs):raise AssertionError('No detector/inference/geometry/noise audit/source reload')
    for module,name in ((geometry,'joint_angle'),(axis_geometry,'midpoint_axis_angle_vertical'),
        (squat,'segment_squat'),(temporal_v2,'segment_squat_v2'),(point_resolution,'segment_squat_point_resolution'),
        (squat_showcase_demo,'load_selection'),(subprocess,'run')):
        monkeypatch.setattr(module,name,forbidden)
    old_open=io.open
    def local_only(file,*args,**kwargs):
        try:p=Path(file).resolve()
        except TypeError:return old_open(file,*args,**kwargs)
        if not p.is_relative_to(source):raise AssertionError('Handoff may read only saved bundle, not original paths/Data')
        return old_open(file,*args,**kwargs)
    monkeypatch.setattr(io,'open',local_only)
    loaded=read_saved_bundle(source,workspace=workspace)
    assert len(loaded['feedbacks'])==2 and len(loaded['points'])==9


def test_all_payload_snapshots_unchanged_and_new_hashes_valid(packaged):
    _,source,output,manifest=packaged
    upstream=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['copied_payload_files'])==set(upstream['files'])-{'index.html'}
    for relative,expected in manifest['copied_payload_files'].items():
        assert (output/relative).read_bytes()==(source/relative).read_bytes()
        assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==expected
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==expected
    assert (output/'upstream-manifest.json').read_bytes()==(source/'manifest.json').read_bytes()
    assert manifest['browser_qa']=='NOT_RUN' and manifest['screenshots']==[]
    assert manifest['noise_audit_repeated'] is manifest['new_detector'] is manifest['new_inference'] is False


def test_saved_nulls_gates_and_independent_provenance_not_changed(packaged):
    _,_,output,manifest=packaged
    d=json.loads((output/'presentation.json').read_text(encoding='utf-8'))
    assert d['official_eligibility'] is d['assessment_enabled'] is False
    assert d['g6_status']=='IN_PROGRESS' and d['official_gate']=='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT'
    assert not manifest['linkage']['synchronized'] and not manifest['linkage']['temporal_overlay_applied_to_skeleton']
    assert any(m['value'] is None for c in d['feedback_cases'] for f in c['frames'] for m in f['metrics'])
    for c in d['point_cases']:
        assert c['accepted_point_count']==0 and c['motion_status']=='unavailable'
        assert c['arrival'] is c['departure'] is c['pause_duration_seconds'] is None


@pytest.mark.parametrize('page',['index.html','START_HERE.html'])
def test_local_links_fragments_and_dynamic_sidecars_exist(packaged,page):
    _,_,output,manifest=packaged
    parsed=Structure();parsed.feed((output/page).read_text(encoding='utf-8'))
    ids={attrs['id'] for _,attrs in parsed.nodes if 'id' in attrs}
    for tag,attrs in parsed.nodes:
        if tag!='a':continue
        target=urlsplit(attrs['href']);assert not target.scheme and not target.netloc
        if target.path:
            path=(output/target.path).resolve()
            assert path.is_relative_to(output) and path.is_file()
        else:assert target.fragment in ids
    for ref in manifest['source_artifacts']:assert (output/ref['snapshot_path']).is_file()


def test_opening_guide_is_short_self_contained_and_does_not_bypass_security(packaged):
    _,_,output,_=packaged
    raw=(output/'HUONG_DAN_MO_DEMO.txt').read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf')
    text=raw.decode('utf-8-sig')
    for phrase in ('START_HERE.html','index.html','manifest.json','presentation.json','noise-review.json',
                   'selection.json','upstream-manifest.json','sources/','Tab/Shift+Tab','Home/End',
                   'Unavailable/null','không phải0','Không cần Python','không tắt bảo vệ',
                   'VÌ SAO KHÔNG CHẤM KỸ THUẬT','assessment=false','browser_qa=NOT_RUN'):
        assert phrase in text
    assert len(text.splitlines())<65
    start=(output/'START_HERE.html').read_text(encoding='utf-8')
    assert '<script' not in start and 'Mở demo' in start and 'EXPERIMENTAL / NOT VALIDATED' in start


def test_keyboard_navigation_contract_labels_order_and_focus_style(packaged):
    _,_,output,_=packaged
    markup=(output/'index.html').read_text(encoding='utf-8');parsed=Structure();parsed.feed(markup)
    ids={attrs['id']:(tag,attrs) for tag,attrs in parsed.nodes if 'id' in attrs}
    for prefix in ('pose','point'):
        button=ids[prefix+'-play'][1];slider=ids[prefix+'-scrub'][1]
        assert button['type']=='button' and button['aria-pressed']=='false'
        assert prefix+'-scrub' in button['aria-controls']
        assert slider['type']=='range' and slider['aria-label'] and slider['step']=='1'
        assert 'keyboard-help' in slider['aria-describedby']
        assert ids[prefix+'-clock'][1]['for']==prefix+'-scrub'
    controls=[attrs.get('id') for tag,attrs in parsed.nodes if tag in ('select','button','input')]
    assert controls==['feedback-select','pose-play','pose-scrub','point-select','point-play','point-scrub']
    assert all(int(attrs.get('tabindex','0'))<=0 for _,attrs in parsed.nodes)
    assert all(selector in markup for selector in ('a:focus-visible','summary:focus-visible','select:focus-visible'))
    assert 'preventDefault' not in markup and 'keydown' not in markup


def test_json_embedding_escaping_and_no_network_dependency(packaged):
    _,_,output,_=packaged
    markup=(output/'index.html').read_text(encoding='utf-8')
    embedded=re.search(r'<script id="showcase-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    assert json.loads(embedded)==json.loads((output/'presentation.json').read_text(encoding='utf-8'))
    assert '<' not in embedded and '&' not in embedded
    assert 'innerHTML' not in markup and 'fetch(' not in markup and "connect-src 'none'" in markup


def test_template_marker_and_script_strings_remain_json_data(packaged):
    workspace,source,_,_=packaged;loaded=read_saved_bundle(source,workspace=workspace)
    refs=copy.deepcopy(loaded['refs']);label='__HANDOFF_GUIDE__ </script><img src=x> & \u2028'
    refs[0]['label']=label
    markup=render_showcase(loaded['feedbacks'],loaded['points'],source_refs=refs,noise_review=loaded['noise'],handoff_guide=True)
    encoded=re.search(r'<script id="showcase-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    assert json.loads(encoded)['source_refs'][0]['label']==label
    assert '<' not in encoded and '&' not in encoded and '\\u2028' in encoded


def test_exclusive_output_preserves_existing_handoff(packaged):
    workspace,source,output,_=packaged
    before={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):write_handoff(source,output,workspace=workspace)
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize('bad',['hash','path','presentation','gate','version','linkage'])
def test_tampered_bundle_is_rejected(packaged,tmp_path,bad):
    workspace,source,_,_=packaged
    target=workspace/'outputs'/('corrupt-'+bad);shutil.copytree(source,target)
    manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    if bad=='hash':manifest['files']['presentation.json']='0'*64
    elif bad=='path':manifest['files']['../Data/no.json']='0'*64
    elif bad=='presentation':
        d=json.loads((target/'presentation.json').read_text(encoding='utf-8'));d['linkage']['synchronized']=True
        raw=json.dumps(d).encode();(target/'presentation.json').write_bytes(raw)
        manifest['files']['presentation.json']=hashlib.sha256(raw).hexdigest()
    elif bad=='gate':manifest['assessment_enabled']=True
    elif bad=='linkage':manifest['linkage']['synchronized']=True
    else:manifest['showcase_version']='99'
    (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(GymError):read_saved_bundle(target,workspace=workspace)


def test_data_and_outside_workspace_input_rejected_before_open(packaged,monkeypatch):
    workspace,_,_,_=packaged
    def forbidden(*args,**kwargs):raise AssertionError('Do not open Data or outside workspace')
    monkeypatch.setattr(io,'open',forbidden)
    with pytest.raises(GymError):read_saved_bundle(workspace/'Data',workspace=workspace)


def test_local_js_accessible_state_and_scrubbers_remain_independent(packaged,tmp_path):
    node=shutil.which('node')
    if node is None:pytest.skip('Optional local JS QA requires existing Node, never installed by this task')
    _,_,output,_=packaged
    markup=(output/'index.html').read_text(encoding='utf-8')
    payload=re.search(r'<script id="showcase-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    code=re.findall(r'<script>(.*?)</script>',markup,re.S)[0]
    runner=r'''
const fs=require('node:fs'),vm=require('node:vm'),r=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements={};let raf;
function node(id){return {textContent:'',value:'0',children:[],handlers:{},append(...n){this.children.push(...n);},replaceChildren(...n){this.children=n;},addEventListener(k,f){this.handlers[k]=f;},getContext(){return new Proxy({},{get(){return ()=>{};},set(){return true;}});}};}
const document={getElementById(id){return elements[id]??=node(id);},createElement(tag){return node(tag);}};
document.getElementById('showcase-data').textContent=r.payload;
new vm.Script(r.code).runInContext(vm.createContext({document,performance:{now:()=>0},requestAnimationFrame(fn){raf=fn;}}));
function state(){return {a:elements['pose-play'].ariaPressed,b:elements['point-play'].ariaPressed,aClock:elements['pose-clock'].textContent,bClock:elements['point-clock'].textContent,aValue:elements['pose-scrub'].ariaValueText,bValue:elements['point-scrub'].ariaValueText};}
function action(id,event,value){elements[id].value=String(value);elements[id].handlers[event]({target:elements[id]});}
const initial=state();elements['pose-play'].handlers.click();const aOn=state();elements['point-play'].handlers.click();const bothOn=state();
action('pose-scrub','input',60);const scrubA=state();action('feedback-select','change',1);const changeA=state();
action('point-select','change',1);const changeB=state();elements['point-play'].handlers.click();raf(100000);const endB=state();
action('pose-scrub','input',180);elements['pose-play'].handlers.click();const replayA=state();raf(100000);const endA=state();
console.log(JSON.stringify({initial,aOn,bothOn,scrubA,changeA,changeB,endB,replayA,endA}));
'''
    js=tmp_path/'state-check.cjs';data=tmp_path/'request.json'
    js.write_text(runner,encoding='utf-8');data.write_text(json.dumps(dict(code=code,payload=payload)),encoding='utf-8')
    done=subprocess.run([node,str(js),str(data)],capture_output=True,text=True,encoding='utf-8')
    assert done.returncode==0,done.stderr
    d=json.loads(done.stdout)
    assert d['initial']['a']==d['initial']['b']=='false'
    assert d['aOn']['a']=='true' and d['aOn']['b']=='false'
    assert d['bothOn']['a']==d['bothOn']['b']=='true'
    assert d['scrubA']['a']=='false' and d['scrubA']['b']=='true'
    assert d['scrubA']['bClock']==d['bothOn']['bClock']
    assert d['changeA']['a']=='false' and d['changeA']['b']=='true' and 'Frame 0' in d['changeA']['aClock']
    assert d['changeB']['b']=='false' and d['changeB']['aClock']==d['changeA']['aClock']
    assert d['endB']['b']=='false' and 'Frame 180' in d['endB']['bClock']
    assert d['replayA']['a']=='true' and 'Frame 0' in d['replayA']['aClock']
    assert d['endA']['a']=='false' and d['endA']['bClock']==d['endB']['bClock']
    for entry in d.values():
        assert entry['aValue']==entry['aClock'] and entry['bValue']==entry['bClock']
