"""Saved synthetic artifact presentation only; no producers/inference in tests."""
import copy
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from gym_coach import axis_geometry,geometry
from gym_coach.flags import GymError
from gym_coach.reporting.squat_showcase import build_showcase_payload,render_showcase
from gym_coach.reporting.squat_showcase_demo import load_selection,write_showcase
from gym_coach.segmentation import squat,temporal_v2,point_resolution

ROOT=Path(__file__).resolve().parents[1]
PROTOTYPE=ROOT/'outputs/gym_result_feedback/run-e1f46752326a49fea5696cc5ccc4e296/demo-verified'
POINT=ROOT/'outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/demo-final'


@pytest.fixture(scope='module')
def selection(tmp_path_factory):
    workspace=tmp_path_factory.mktemp('showcase-workspace');outputs=workspace/'outputs';outputs.mkdir()
    feedback=[]
    for name in ('synthetic','synthetic-gap'):
        target=outputs/(name+'.json');target.write_bytes((PROTOTYPE/name/'feedback.json').read_bytes())
        feedback.append(dict(label=name,path=target.relative_to(workspace).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    manifest=json.loads((POINT/'manifest.json').read_text(encoding='utf-8'))
    point_dir=outputs/'points';point_dir.mkdir()
    for entry in manifest['cases']:
        for field in ('input_path','result_path'):
            target=point_dir/entry[field];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((POINT/entry[field]).read_bytes())
    point_manifest=point_dir/'manifest.json';point_manifest.write_bytes((POINT/'manifest.json').read_bytes())
    noise=outputs/'noise.json';noise.write_text(json.dumps(dict(review_version='squat-showcase-noise-review-v1',
        independent_sample_noise_evidence_found=False,summary='Scoped test presentation: no independent per-sample uncertainty evidence.')),encoding='utf-8')
    config=dict(selection_version='squat-showcase-sources-v1',feedback=feedback,
        point_manifest=dict(path=point_manifest.relative_to(workspace).as_posix(),sha256=hashlib.sha256(point_manifest.read_bytes()).hexdigest()),
        noise_review=dict(path=noise.relative_to(workspace).as_posix(),sha256=hashlib.sha256(noise.read_bytes()).hexdigest()))
    path=outputs/'selection.json';path.write_text(json.dumps(config),encoding='utf-8')
    return workspace,path,config


@pytest.fixture(scope='module')
def loaded(selection):
    workspace,path,_=selection
    return load_selection(path,workspace=workspace)


@pytest.fixture(scope='module')
def payload(loaded):
    return build_showcase_payload(loaded['feedbacks'],loaded['points'],source_refs=loaded['refs'],noise_review=loaded['noise'])


def test_independent_provenance_even_when_fps_and_frame_count_match(payload):
    a=payload['feedback_cases'][0];b=payload['point_cases'][1]
    assert a['fps']==b['fps']==30. and len(a['frames'])==b['frame_count']==181
    assert a['video_id']!=b['video_id'] and a['temporal_input_hash']!=b['input_hash']
    assert payload['linkage']['mode']=='independent_panels'
    assert payload['linkage']['synchronized'] is False
    assert payload['linkage']['temporal_overlay_applied_to_skeleton'] is False
    assert payload['official_eligibility'] is payload['assessment_enabled'] is False


@pytest.mark.parametrize('case',[0,1])
@pytest.mark.parametrize('index',[0,58,60,180])
def test_exact_saved_skeleton_metric_identity_and_no_legacy_phase(loaded,payload,case,index):
    original=loaded['feedbacks'][case].frames[index];view=payload['feedback_cases'][case]['frames'][index]
    assert view['frame_index']==original.frame_index and view['timestamp_seconds']==original.timestamp_seconds
    assert view['source_frame_hash']==original.source_frame_hash
    assert view['joints']==[j.model_dump(mode='json') for j in original.overlay_joints]
    assert view['metrics']==[m.model_dump(mode='json') for m in original.metrics]
    assert not {'phase','phase_evidence','phase_reasons'}&view.keys()
    assert all(m['temporal_analysis_eligible'] is False for m in view['metrics'])


def test_unavailable_metrics_keep_null_reasons_quality_and_imputation(payload):
    frame=payload['feedback_cases'][1]['frames'][60]
    absent=[m for m in frame['metrics'] if m['status']=='unavailable']
    assert absent and all(m['value'] is None and m['reasons'] and m['flags'] for m in absent)
    assert any(j['status']=='missing' and j['x'] is None for j in frame['joints'])
    assert payload['feedback_cases'][1]['quality']['eligibility']['action']=='degrade'


def test_saved_point_events_not_promoted_and_no_new_skeleton(payload,loaded):
    for view,source in zip(payload['point_cases'],loaded['points']):
        assert view['events']==[e.model_dump(mode='json') for e in source.base_result.events]
        assert view['accepted_point_count']==0 and view['motion_status']=='unavailable'
        assert view['arrival'] is view['departure'] is view['pause_duration_seconds'] is None
        assert not any('joints' in s or 'value' in s for s in view['samples'])
        assert all(c['point_turn_event'] is None for c in view['checks'])


def test_rendering_is_read_only_no_geometry_or_producer_calls(loaded,monkeypatch):
    before=[item.model_dump_json() for item in (*loaded['feedbacks'],*loaded['points'])]
    def forbidden(*args,**kwargs):raise AssertionError('No geometry/producer/inference in renderer')
    for module,name in ((geometry,'joint_angle'),(axis_geometry,'midpoint_axis_angle_vertical'),
        (squat,'segment_squat'),(temporal_v2,'segment_squat_v2'),(point_resolution,'segment_squat_point_resolution')):
        monkeypatch.setattr(module,name,forbidden)
    original=io.open
    def guard(file,*args,**kwargs):
        if 'data' in str(file).replace('\\','/').lower().split('/'):raise AssertionError('No Data access')
        return original(file,*args,**kwargs)
    monkeypatch.setattr(io,'open',guard)
    assert 'SYNTHETIC ONLY' in render_showcase(loaded['feedbacks'],loaded['points'])
    assert before==[item.model_dump_json() for item in (*loaded['feedbacks'],*loaded['points'])]


def test_json_html_escaping_and_offline_no_network(loaded):
    dangerous='</script><img src=x onerror=alert(1)> & \u2028\u2029'
    refs=copy.deepcopy(loaded['refs']);refs[2]['label']=dangerous
    markup=render_showcase(loaded['feedbacks'],loaded['points'],source_refs=refs,noise_review=dict(summary=dangerous))
    assert dangerous not in markup and '\\u003c/script\\u003e' in markup
    assert '\\u2028' in markup and '\\u2029' in markup
    assert 'innerHTML' not in markup and 'fetch(' not in markup and 'XMLHttpRequest' not in markup
    assert "connect-src 'none'" in markup
    encoded=re.search(r'<script id="showcase-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    assert json.loads(encoded)['noise_review']['summary']==dangerous


@pytest.mark.parametrize('path',['javascript:alert(1)','https://example.com/private.json','../source.json','sources/../x.json'])
def test_unsafe_snapshot_links_rejected(loaded,path):
    refs=copy.deepcopy(loaded['refs']);refs[0]['snapshot_path']=path
    with pytest.raises(GymError):render_showcase(loaded['feedbacks'],loaded['points'],source_refs=refs)


@pytest.mark.parametrize('kind',['nan','assessment','not_synthetic','wrong_type'])
def test_reject_invalid_or_non_synthetic_saved_models(loaded,kind):
    saved=loaded['feedbacks'][0]
    if kind=='nan':
        first=saved.frames[0];m=first.metrics[0].model_copy(update={'value':float('nan')})
        saved=saved.model_copy(update={'frames':(first.model_copy(update={'metrics':(m,*first.metrics[1:])}),*saved.frames[1:])})
    elif kind=='assessment':saved=saved.model_copy(update={'assessment':saved.assessment.model_copy(update={'enabled':True})})
    elif kind=='not_synthetic':saved=saved.model_copy(update={'source':saved.source.model_copy(update={'kind':'real_preview'})})
    else:saved=object()
    with pytest.raises(GymError):build_showcase_payload((saved,),loaded['points'])


@pytest.mark.parametrize('bad',['hash','Data','version','duplicate','nonfinite','point_summary','point_traversal'])
def test_selection_rejects_bad_hash_input_location_and_manifest(selection,bad):
    workspace,path,original=selection;config=copy.deepcopy(original)
    if bad=='hash':config['feedback'][0]['sha256']='0'*64
    elif bad=='Data':config['feedback'][0]['path']='Data/DO_NOT_OPEN.json'
    elif bad=='version':config['selection_version']='99'
    elif bad in ('point_summary','point_traversal'):
        original_manifest=workspace/config['point_manifest']['path']
        manifest=json.loads(original_manifest.read_text(encoding='utf-8'))
        if bad=='point_summary':manifest['cases'][0]['frames']+=1
        else:manifest['cases'][0]['result_path']='../../Data/DO_NOT_OPEN.json'
        target=original_manifest.with_name(bad+'.json');target.write_text(json.dumps(manifest),encoding='utf-8')
        config['point_manifest']=dict(path=target.relative_to(workspace).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target=path.with_name(bad+'.json')
    if bad=='duplicate':target.write_text('{"selection_version":"a","selection_version":"b"}',encoding='utf-8')
    elif bad=='nonfinite':target.write_text('{"x":NaN}',encoding='utf-8')
    else:target.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(GymError):load_selection(target,workspace=workspace)


def test_demo_exclusive_hashes_links_json_roundtrip(selection):
    workspace,path,_=selection;output=workspace/'outputs'/'demo-new'
    manifest=write_showcase(path,output,workspace=workspace)
    assert manifest['feedback_cases']==2 and manifest['point_cases']==9
    assert manifest['new_inference'] is manifest['new_geometry'] is manifest['new_detector'] is False
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==expected
    markup=(output/'index.html').read_text(encoding='utf-8')
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            if tag=='a':
                for key,value in attrs:
                    if key=='href':assert (output/value).is_file()
    Links().feed(markup)
    for ref in manifest['source_artifacts']:
        assert (output/ref['snapshot_path']).read_bytes()==(workspace/ref['original_relative_path']).read_bytes()
    before={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):write_showcase(path,output,workspace=workspace)
    assert all(p.read_bytes()==value for p,value in before.items())


def test_local_javascript_independent_controls_saved_values_and_missing_skeleton(loaded,tmp_path):
    node=shutil.which('node')
    if node is None:pytest.skip('Optional local JavaScript QA requires preinstalled Node; no install performed')
    markup=render_showcase(loaded['feedbacks'],loaded['points'])
    payload=re.search(r'<script id="showcase-data" type="application/json">(.*?)</script>',markup,re.S).group(1)
    code=re.findall(r'<script>(.*?)</script>',markup,re.S)[0]
    runner=r'''
const fs=require('node:fs'),vm=require('node:vm'),request=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements={},draws={};let raf;
function element(id){return {id,textContent:'',value:'0',children:[],handlers:{},append(...nodes){this.children.push(...nodes);},replaceChildren(...nodes){this.children=nodes;},addEventListener(name,fn){this.handlers[name]=fn;},getContext(){const canvas=id;return new Proxy({},{get(_,name){return (...args)=>{(draws[canvas]??=[]).push([name,...args]);};},set(){return true;}});}};}
const document={getElementById(id){return elements[id]??=(element(id));},createElement(tag){return element(tag);}};
document.getElementById('showcase-data').textContent=request.payload;
new vm.Script(request.code).runInContext(vm.createContext({document,performance:{now:()=>0},requestAnimationFrame(fn){raf=fn;}}));
function action(id,event,value){elements[id].value=String(value);elements[id].handlers[event]({target:elements[id]});}
action('pose-scrub','input',58);const a58=elements['pose-clock'].textContent,poseCalls=draws['pose-canvas'].length;
action('point-select','change',1);action('point-scrub','input',60);
const independent={clockA:elements['pose-clock'].textContent,clockB:elements['point-clock'].textContent,poseCallsUnchanged:draws['pose-canvas'].length===poseCalls,events:elements.events.textContent,status:elements['point-status'].textContent,motion:elements['motion-status'].textContent,pause:elements['pause-status'].textContent};
action('feedback-select','change',1);const start=draws['pose-canvas'].length;action('pose-scrub','input',60);
const gap={metrics:elements.metrics.children.map(c=>c.children[1].textContent),arcs:draws['pose-canvas'].slice(start).filter(x=>x[0]==='arc').map(x=>x.slice(1,3))};
action('point-select','change',3);action('point-scrub','input',60);const plateau=elements['point-reasons'].textContent;
action('point-select','change',8);action('point-scrub','input',60);const edge={events:elements.events.textContent,reasons:elements['point-reasons'].textContent,motion:elements['motion-status'].textContent};
const bBefore=elements['point-clock'].textContent;elements['pose-play'].handlers.click();raf(1000);
console.log(JSON.stringify({a58,independent,gap,plateau,edge,playbackIndependent:elements['point-clock'].textContent===bBefore,frameAAfterPlayback:elements['pose-clock'].textContent}));
'''
    script=tmp_path/'ui-code-check.cjs';request=tmp_path/'request.json'
    script.write_text(runner,encoding='utf-8');request.write_text(json.dumps(dict(payload=payload,code=code)),encoding='utf-8')
    result=subprocess.run([node,str(script),str(request)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    got=json.loads(result.stdout)
    assert got['independent']['clockA']==got['a58'] and got['independent']['poseCallsUnchanged']
    assert 'Frame 60' in got['independent']['clockB'] and 'bottom_anchor' in got['independent']['events']
    assert 'Unavailable' in got['independent']['status'] and 'Unavailable' in got['independent']['motion']
    assert '(null)' in got['independent']['pause']
    source=loaded['feedbacks'][1].frames[60]
    assert got['gap']['metrics']==['Unavailable' if m.value is None else f'{m.value:.1f}°' for m in source.metrics]
    expected=[[j.x*640,j.y*640] for j in source.overlay_joints if j.status!='missing' and j.x is not None and j.y is not None and 0<=j.x<=1 and 0<=j.y<=1]
    assert got['gap']['arcs']==expected
    assert 'raw_minimum_not_strict' in got['plateau'] and 'no_full_pvp_candidate' in got['edge']['reasons']
    assert 'no_full_pvp_candidate' in got['edge']['motion']
    assert 'Không có event' in got['edge']['events'] and got['playbackIndependent']
    assert 'Frame 90' in got['frameAAfterPlayback']
