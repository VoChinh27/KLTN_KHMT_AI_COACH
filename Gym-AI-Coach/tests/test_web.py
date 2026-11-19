"""Local HTTP/security/controller boundaries. Fake jobs are not inference QA."""
import io
import json
from pathlib import Path
import sys
import subprocess
import threading
from types import SimpleNamespace
from urllib import request, error
import uuid

import pytest

from gym_coach import web
from gym_coach.job_controller import VerifiedRun


class FakeController:
    def __init__(self):
        self.active = True
        self.status = "running"
        self.last_event = None
        self.verified = None
        self.error = None
        self.log = []
        self.started = web.time.monotonic()

    def start(self, video, parent, **options):
        jid = uuid.uuid4().hex
        directory = parent / ("run-" + jid)
        directory.mkdir(parents=True)
        self.job = SimpleNamespace(run_id=jid, directory=directory, output=directory/"result",
            video=video, side=options["side"], exploratory=options["exploratory"])
        return self.job

    def poll(self):
        return []


@pytest.fixture
def app(tmp_path):
    return web.Application(tmp_path/"web có dấu",controller_factory=FakeController,
        inspector=lambda p:dict(width=320,height=240,fps=25,frame_count=9))


def upload(app, name="Video có dấu & <script>.avi"):
    return app.upload(io.BytesIO(b"fake decoder fixture"),20,name)


def payload(uid, **kw):
    data=dict(upload_id=uid, request_id=uuid.uuid4().hex,side="left",mode="strict",acknowledged=False,use_cache=True)
    data.update(kw)
    return data


@pytest.fixture
def http(app):
    server=web.LocalServer(app,0)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    yield app,server
    server.shutdown();server.server_close();thread.join()


def call(app, path, data=None, *, headers=None, method=None):
    hdr={}
    if data is not None:
        hdr.update(Origin=app.origin,**{"X-Gym-CSRF":app.token,"Content-Type":"application/json"})
        if not isinstance(data,bytes): data=json.dumps(data).encode()
    hdr.update(headers or {})
    req=request.Request(app.origin+path,data=data,headers=hdr,method=method)
    try:
        with request.urlopen(req,timeout=5) as response:
            return response.status,response.headers,response.read()
    except error.HTTPError as exc:
        return exc.code,exc.headers,exc.read()


def completed(app,jid):
    c=app.controllers[jid];out=c.job.output;out.mkdir()
    (out/'overlay-h264.mp4').write_bytes(b"0123456789")
    (out/'result.json').write_text('{"value":null,"assessment":false}')
    (out/'START_HERE.html').write_text('<html>fixture only</html>')
    files={p.name:web.identity(p) for p in out.iterdir()}
    web.save(out/'manifest.json',dict(files=files))
    c.active=False;c.status='completed'
    c.verified=VerifiedRun(out,web.identity(out/'manifest.json')['sha256'],
        (('video','overlay-h264.mp4',files['overlay-h264.mp4']['sha256']),),
        (('total_frames',9),('available_frames',0),('unavailable_frames',9),('available_ratio',0.)), 'hit',False)
    return c


def test_upload_internal_names_and_failure_receipt(app):
    record=upload(app,'../../x & <script>.avi');path,_=app.uploads[record['upload_id']]
    assert path.name=='input.avi' and path.is_relative_to(app.session)
    assert record['original_name']=='../../x & <script>.avi'
    app.inspector=lambda p:(_ for _ in ()).throw(ValueError('decoder rejected'))
    with pytest.raises(web.RequestError,match='Không đọc được'):upload(app)
    assert len(list(app.session.rglob('upload-failed.json')))==1
    assert len(app.uploads)==1 and not app.controllers


@pytest.mark.parametrize('length',[0,-1,web.MAX_UPLOAD+1])
def test_upload_size_rejected_before_writing(app,length):
    with pytest.raises(web.RequestError):app.upload(io.BytesIO(),length,'a.avi')
    assert not list((app.session/'uploads').iterdir())


@pytest.mark.parametrize('name',['a.exe','a.html','bad\n.avi','x'*241+'.avi'])
def test_upload_type_name_limit(app,name):
    with pytest.raises(web.RequestError):upload(app,name)


def test_partial_transfer_kept_failed_not_accepted(app):
    with pytest.raises(web.RequestError):app.upload(io.BytesIO(b'few'),20,'a.avi')
    assert not app.uploads
    assert list(app.session.rglob('upload-failed.json'))


def test_strict_defaults_explicit_ack_duplicate_idempotency_refresh(app):
    u=upload(app);p=payload(u['upload_id']);first=app.start(p)
    assert first['mode']=='strict' and first['side']=='left'
    assert app.start(p)['job_id']==first['job_id']
    assert app.state()['jobs'][0]['job_id']==first['job_id']
    with pytest.raises(web.RequestError,match='Đang có'):app.start(payload(u['upload_id']))
    assert len(app.controllers)==1
    app.controllers[first['job_id']].active=False
    with pytest.raises(web.RequestError,match='xác nhận'):app.start(payload(u['upload_id'],mode='exploratory'))
    second=app.start(payload(u['upload_id'],mode='exploratory',acknowledged=True,side='right'))
    assert second['side']=='right' and second['mode']=='exploratory'


def test_concurrent_submit_only_one_job(app):
    uid=upload(app)['upload_id'];barrier=threading.Barrier(3);results=[]
    def submit():
        barrier.wait()
        try:results.append(app.start(payload(uid)))
        except web.RequestError as exc:results.append(exc.reason)
    threads=[threading.Thread(target=submit) for _ in range(2)]
    for t in threads:t.start()
    barrier.wait()
    for t in threads:t.join()
    assert len(app.controllers)==1 and 'job_active' in results


@pytest.mark.parametrize('change',[{'path':'C:/private.avi'},{'side':'both'},{'mode':'official'},
    {'acknowledged':1},{'use_cache':'true'},{'upload_id':'../file'},{'request_id':'invalid'}])
def test_no_arbitrary_paths_or_coerced_options(app,change):
    p=payload(upload(app)['upload_id']);p.update(change)
    with pytest.raises(web.RequestError):app.start(p)
    assert not app.controllers


def test_changed_upload_and_request_conflict(app):
    uid=upload(app)['upload_id'];p=payload(uid);jid=app.start(p)['job_id']
    with pytest.raises(web.RequestError):app.start(dict(p,side='right'))
    app.controllers[jid].active=False
    app.uploads[uid][0].write_bytes(b'changed')
    with pytest.raises(web.RequestError,match='Bytes'):app.start(payload(uid))


def test_zero_coverage_completed_not_failure_and_partial_not_downloadable(app):
    jid=app.start(payload(upload(app)['upload_id']))['job_id']
    with pytest.raises(web.RequestError):app.artifact(jid,'result.json')
    c=completed(app,jid);snapshot=app.snapshot(jid)
    assert snapshot['status']=='completed' and snapshot['coverage']['available_frames']==0
    assert snapshot['error'] is None and snapshot['cache_status']=='hit'
    assert app.artifact(jid,'result.json').read_text().startswith('{"value":null')
    (c.job.output/'result.json').write_text('tampered')
    with pytest.raises(web.RequestError,match='thay đổi'):app.artifact(jid,'result.json')


@pytest.mark.parametrize('name',['../result.json','..','a/b','a\\b','C:private','%2f..','not-listed.json'])
def test_artifact_allowlist(app,name):
    jid=app.start(payload(upload(app)['upload_id']))['job_id'];completed(app,jid)
    with pytest.raises(web.RequestError):app.artifact(jid,name)


def test_manifest_tampering_prevents_success_download(app):
    jid=app.start(payload(upload(app)['upload_id']))['job_id'];c=completed(app,jid)
    (c.job.output/'manifest.json').write_text('{}')
    with pytest.raises(web.RequestError):app.snapshot(jid)
    with pytest.raises(web.RequestError):app.artifact(jid,'manifest.json')


@pytest.mark.parametrize('headers',[{'Host':'evil.test'}, {'Origin':'https://evil.test'},
    {'Sec-Fetch-Site':'cross-site'}, {'Host':'localhost:8765'}])
def test_host_origin_dns_rebinding_protection(http,headers):
    app,_=http;assert call(app,'/api/state',headers=headers)[0]==403


@pytest.mark.parametrize('headers',[{'Origin':'null'},{'X-Gym-CSRF':'wrong'},{'Origin':'https://evil.test'}])
def test_mutations_require_same_origin_and_csrf(http,headers):
    app,_=http;assert call(app,'/api/jobs',{},headers=headers)[0]==403


def test_http_routes_headers_upload_and_range(http):
    app,server=http;assert server.server_address[0]=='127.0.0.1'
    status,h,b=call(app,'/');assert status==200 and b'NOT_VALIDATED' in b
    assert 'frame-ancestors' in h['Content-Security-Policy'] and 'Access-Control-Allow-Origin' not in h
    assert call(app,'/api/state')[0]==200
    status,h,b=call(app,'/api/uploads',b'fake decoder fixture',headers={'Content-Type':'application/octet-stream','X-File-Name':'clip.avi'})
    assert status==201
    uid=json.loads(b)['upload_id'];status,_,b=call(app,'/api/jobs',payload(uid));assert status==202
    jid=json.loads(b)['job_id'];completed(app,jid)
    status,h,b=call(app,f'/artifacts/{jid}/overlay-h264.mp4',headers={'Range':'bytes=3-6'})
    assert status==206 and b==b'3456' and h['Content-Range']=='bytes 3-6/10'
    status,h,b=call(app,f'/artifacts/{jid}/overlay-h264.mp4',headers={'Range':'bytes=-2'})
    assert status==206 and b==b'89'
    status,h,b=call(app,f'/artifacts/{jid}/overlay-h264.mp4',headers={'Range':'bytes=20-'})
    assert status==416 and h['Content-Range']=='bytes */10'
    status,h,b=call(app,f'/artifacts/{jid}/result.json?download=1')
    assert status==200 and 'attachment;' in h['Content-Disposition']
    status,h,b=call(app,f'/artifacts/{jid}/overlay-h264.mp4',method='HEAD')
    assert status==200 and not b and h['Content-Length']=='10'
    assert call(app,f'/artifacts/{jid}/%2e%2e%2fprivate')[0]==403
    assert call(app,'/Data/private.avi')[0]==404
    assert call(app,'/api/open',{})[0]==404


@pytest.mark.parametrize('value',['bytes=0-1,4-5','items=1-3','bytes=5-2','bytes=-0','bytes=-'])
def test_invalid_ranges(value):
    with pytest.raises(web.RequestError):web.range_bounds(value,10)


def test_ui_escaping_defaults_and_no_fake_measurements():
    js=(web.ASSETS/'app.js').read_text(encoding='utf-8');html=(web.ASSETS/'index.html').read_text(encoding='utf-8')
    assert 'innerHTML' not in js and 'textContent' in js and 'replaceChildren' in js
    assert '<option value="strict">' in html and '<input id="ack" type="checkbox" disabled>' in html
    assert 'e.total!==null' in js and 'removeAttribute(\'value\')' in js
    assert 'job.status===\'completed\'&&!job.active&&job.coverage!==null' in js
    assert 'Math.acos' not in js and 'official_eligibility=false' in html
    assert 'assessment=false' in html and 'REFERENCE_PENDING' in html


def test_web_import_no_tk_or_inference_and_cli_help():
    code="import sys;import gym_coach.web;assert not set(('tkinter','gym_coach.desktop','mediapipe','cv2','numpy','scipy','torch'))&set(sys.modules)"
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,'-B','-m','gym_coach.web','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--output-root' in result.stdout and '--host' not in result.stdout


def test_aborted_browser_video_request_does_not_write_second_response():
    handler=object.__new__(web.Handler)
    handler.guard=lambda _:None
    handler.path='/';handler.server=SimpleNamespace(app=None)
    handler.serve_file=lambda *_:(_ for _ in ()).throw(ConnectionAbortedError('browser stopped range request'))
    handler.json_response=lambda *_:pytest.fail('must not respond on an aborted stream')
    handler.dispatch()
    assert handler.close_connection


def test_shutdown_waits_worker_and_does_not_cancel(monkeypatch,tmp_path):
    actions=[]
    c=SimpleNamespace(thread=SimpleNamespace(join=lambda:actions.append('joined')),poll=lambda:actions.append('polled'))
    app=SimpleNamespace(controllers={'fixture':c},origin='http://127.0.0.1:1',session=tmp_path,stopping=False)
    class Server:
        def __init__(self,*args):pass
        def serve_forever(self,**kwargs):raise KeyboardInterrupt
        def server_close(self):actions.append('closed')
    monkeypatch.setattr(web,'Application',lambda _:app);monkeypatch.setattr(web,'LocalServer',Server)
    web.main(['--output-root',str(tmp_path)])
    assert app.stopping and actions==['closed','joined','polled']
