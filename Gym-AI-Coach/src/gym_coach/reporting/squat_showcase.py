"""Offline presentation of saved synthetic artifacts, with independent clocks.

No feature/pose/segmentation producer is called. Legacy phases are not projected
onto a different source. This presentation DTO is not a new domain contract.
"""
import json
import re

from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, GymError, Severity, StructuredFlag
from ..segmentation.models import strict_json
from ..segmentation.point_resolution_models import SquatPointResolutionAudit
from .squat_models import SquatFeedbackResult


def presentation_error(message):
    return GymError(StructuredFlag(code=FlagCode.INVALID_FEEDBACK_INPUT,
        category=ErrorCategory.INPUT_ERROR,severity=Severity.ERROR,stage=Stage.REPORTING,
        source_tier=SourceTier.S3,message=message))


def build_showcase_payload(feedbacks,point_audits,*,source_refs=(),noise_review=None):
    """Revalidate/detach saved records; preserve exact framewise values/identities."""
    try:
        if not feedbacks or not point_audits:
            raise ValueError('Both independent synthetic panels require saved artifacts')
        for ref in source_refs:
            if (not re.fullmatch(r'sources/[A-Za-z0-9_/-]+\.json',ref['snapshot_path'])
                    or '..' in ref['snapshot_path'] or not re.fullmatch(r'[a-f0-9]{64}',ref['sha256'])):
                raise ValueError('Snapshot links must be local JSON paths with byte SHA-256')
        labels={ref['role']:ref['label'] for ref in source_refs}
        feedback_cases=[]
        for item in feedbacks:
            if not isinstance(item,SquatFeedbackResult):
                raise ValueError('Expected saved SquatFeedbackResult')
            saved=SquatFeedbackResult.model_validate_json(strict_json(item),strict=True)
            if saved.source.kind!='synthetic_contract' or any(f.preview_image is not None for f in saved.frames):
                raise ValueError('Showcase accepts synthetic pose/metric artifacts only')
            feedback_cases.append(dict(video_id=saved.source.video_id,fps=saved.source.fps,
                source=saved.source.model_dump(mode='json'),source_hash=saved.source_hash,
                feedback_hash=saved.content_hash(),registry_version=saved.registry_version,registry_hash=saved.registry_hash,
                temporal_input_hash=saved.segmentation.input_hash,segmentation_hash=saved.segmentation_hash,
                profile=saved.segmentation.profile.model_dump(mode='json'),profile_hash=saved.segmentation.profile_hash,
                quality=saved.quality.model_dump(mode='json'),legacy_phase_displayed=False,
                frames=[dict(frame_index=f.frame_index,timestamp_seconds=f.timestamp_seconds,
                    source_frame_hash=f.source_frame_hash,overlay_source=f.overlay_source,
                    coordinate_space=f.coordinate_space,joints=[j.model_dump(mode='json') for j in f.overlay_joints],
                    metrics=[m.model_dump(mode='json') for m in f.metrics]) for f in saved.frames]))
        point_cases=[]
        for index,item in enumerate(point_audits):
            if not isinstance(item,SquatPointResolutionAudit):
                raise ValueError('Expected saved SquatPointResolutionAudit')
            saved=SquatPointResolutionAudit.model_validate_json(strict_json(item),strict=True)
            base=saved.base_result;source=base.legacy.input
            point_cases.append(dict(case_label=labels.get(f'point-{index}',source.video_id+' / '+source.source_hash[:12]),
                video_id=source.video_id,fps=source.fps,frame_count=len(source.samples),
                source_hash=source.source_hash,input_hash=base.legacy.input_hash,evidence_basis=source.evidence_basis,
                audit_hash=saved.content_hash(),producer_version=saved.producer_version,
                policy=saved.policy.model_dump(mode='json'),policy_hash=saved.policy_hash,
                base_policy=base.policy.model_dump(mode='json'),base_policy_hash=base.policy_hash,base_hash=saved.base_hash,
                profile=base.legacy.profile.model_dump(mode='json'),profile_hash=base.legacy.profile_hash,
                candidate_count=base.candidate_count,accepted_point_count=saved.accepted_point_count,
                necessary_condition_count=saved.necessary_condition_count,
                motion_status='unavailable',arrival=None,departure=None,pause_duration_seconds=None,
                events=[e.model_dump(mode='json') for e in base.events],cycles=[c.model_dump(mode='json') for c in base.cycles],
                checks=[c.model_dump(mode='json') for c in saved.checks],
                diagnostics=[r.model_dump(mode='json') for r in (*base.diagnostics,*saved.diagnostics)],
                flags=[f.model_dump(mode='json') for f in base.legacy.flags],
                samples=[dict(frame_index=s.frame_index,timestamp_seconds=s.timestamp_seconds,side_source=s.side_source,
                    joint_provenance=[p.value for p in s.joint_provenance],reasons=[r.value for r in s.reasons]) for s in source.samples]))
        payload=dict(presentation_version='gym-squat-showcase-v1',exercise='BodyWeightSquats',
            experimental_label='EXPERIMENTAL / NOT VALIDATED',official_eligibility=False,assessment_enabled=False,
            official_gate='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',g6_status='IN_PROGRESS',
            linkage=dict(mode='independent_panels',synchronized=False,temporal_overlay_applied_to_skeleton=False,
                reason='Different video/source/input hashes and source capabilities; matching FPS/frame numbers do not establish linkage.'),
            feedback_cases=feedback_cases,point_cases=point_cases,source_refs=list(source_refs),noise_review=noise_review)
        # Detach all nested presentation data too; reject nonfinite metadata.
        return json.loads(json.dumps(payload,ensure_ascii=False,allow_nan=False))
    except (ValueError,TypeError,AttributeError,KeyError) as exc:
        raise presentation_error(f'Cannot present saved synthetic artifacts: {exc}') from exc


def render_showcase(feedbacks,point_audits,*,source_refs=(),noise_review=None,handoff_guide=False):
    payload=build_showcase_payload(feedbacks,point_audits,source_refs=source_refs,noise_review=noise_review)
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    for char,escape in (('&','\\u0026'),('<','\\u003c'),('>','\\u003e'),('\u2028','\\u2028'),('\u2029','\\u2029')):
        encoded=encoded.replace(char,escape)
    return _HTML.replace('__HANDOFF_GUIDE__',_HANDOFF_GUIDE if handoff_guide else '').replace('__SHOWCASE_JSON__',encoded)


_HANDOFF_GUIDE='''<nav class="notice" aria-label="Điều hướng demo"><a href="START_HERE.html">Hướng dẫn mở / bàn giao</a> · <a href="#pose-panel">Đến panel A</a> · <a href="#point-panel">Đến panel B</a></nav>'''


_HTML=r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; font-src 'none'; frame-src 'none'; base-uri 'none'">
<title>Gym Result Feedback · Synthetic showcase</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b1320;color:#ebf2f9;line-height:1.5}
*{box-sizing:border-box}body{margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:clamp(26px,4vw,42px);margin:8px 0}h2{font-size:22px;margin:8px 0}h3{font-size:17px}p{margin:8px 0}
.eyebrow{font-size:12px;letter-spacing:.15em;color:#7de4d8}.badges{display:flex;flex-wrap:wrap;gap:8px}.badge{font-size:12px;border:1px solid #7e672f;color:#ffdb91;background:#30281a;border-radius:5px;padding:5px 10px}
.notice{border-left:3px solid #e2b05e;padding:12px 16px;background:#1c2330;margin:20px 0}.muted,.small{color:#b7c6d8}.small{font-size:12px;overflow-wrap:anywhere}
.panels{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.panel{border:1px solid #314459;background:#121e2e;padding:20px;border-radius:14px;min-width:0}
select,button{max-width:100%;background:#1e3744;color:#edf8ff;border:1px solid #648493;padding:8px;border-radius:6px}select{width:100%}button{cursor:pointer}button:disabled{opacity:.5}input{width:100%;accent-color:#76dbd4}
.controls{display:grid;grid-template-columns:auto minmax(50px,1fr);gap:12px;align-items:center;margin:14px 0}output{font-size:13px;font-variant-numeric:tabular-nums}
canvas{width:100%;display:block;background:#0b1725;border-radius:10px}#pose-canvas{aspect-ratio:1;max-height:430px;object-fit:contain}#event-canvas{height:145px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}.metric,.status{border:1px solid #324256;border-radius:8px;padding:12px}.value{font-size:25px;display:block}.unavailable{color:#ffcf82}.status{margin:12px 0;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:8px 5px;text-align:left;border-bottom:1px solid #314459;overflow-wrap:anywhere}details{margin:14px 0}summary{cursor:pointer}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;color:#b7c6d8}
.footer{margin:20px 0}a{color:#89e2da}li{margin:8px 0}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #ffd88a;outline-offset:3px}
@media(max-width:900px){.panels{grid-template-columns:1fr}main{padding:16px}}@media(max-width:450px){.metrics{grid-template-columns:1fr}}
</style>
<main id="main-content"><div class="eyebrow">GYM AI COACH / SAVED ARTIFACT SHOWCASE</div>
<h1>Gym Result Feedback</h1><p>BodyWeightSquats · Skeleton, số đo và cổng temporal</p>
__HANDOFF_GUIDE__
<div class="badges"><span class="badge">EXPERIMENTAL / NOT VALIDATED</span><span class="badge">SYNTHETIC ONLY · OFFLINE</span><span class="badge">official_eligibility=false</span><span class="badge">assessment=false</span></div>
<p class="small">G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT · G6 IN_PROGRESS</p>
<div class="notice"><strong>Hai panel độc lập, không đồng bộ frame.</strong><p>Panel A là pose/số đo synthetic đã lưu. Panel B là scalar fixture khác nguồn. Cùng FPS hoặc số frame không chứng minh cùng provenance; event temporal không được đặt lên skeleton.</p><p class="small">Không chạy model hoặc detector. Không hiển thị phase v1 từ prototype như phase của nguồn point-resolution.</p></div>
<p id="keyboard-help" class="small">Bàn phím: Tab / Shift+Tab chuyển điều khiển; Enter / Space bật nút; khi slider có focus, dùng phím mũi tên hoặc Home / End. Mỗi panel có clock riêng.</p>
<div class="panels">
<section id="pose-panel" tabindex="-1" class="panel" aria-label="Panel A: saved synthetic pose and metrics">
<div class="eyebrow">A / SAVED SYNTHETIC POSE &amp; METRICS</div><h2>Skeleton và số đo theo frame</h2>
<label>Nguồn A <select id="feedback-select" aria-label="Chọn nguồn pose"></select></label><p id="pose-source" class="small"></p>
<canvas id="pose-canvas" width="640" height="640" aria-label="Skeleton từ raw joints của JSON synthetic"></canvas>
<div class="controls"><button id="pose-play" type="button" aria-pressed="false" aria-controls="pose-canvas pose-scrub">Phát A</button><input id="pose-scrub" aria-label="Tua frame A" aria-describedby="keyboard-help pose-clock" type="range" min="0" max="0" value="0" step="1"></div><output id="pose-clock" for="pose-scrub"></output>
<p class="small">Cyan: observed · Amber: low_visibility · Missing: không vẽ. Tọa độ raw image_normalized_xy. Skeleton này được sinh synthetic ở lượt trước, không phải inference MediaPipe.</p>
<div id="pose-quality" class="status"></div><div id="metrics" class="metrics"></div>
<p class="small">Số đo đọc nguyên từ artifact; chỉ định dạng hiển thị. Unvalidated, không phải chuẩn kỹ thuật. Framewise temporal_analysis_eligible=false.</p>
<details><summary>Provenance của panel A</summary><pre id="pose-provenance"></pre></details></section>
<section id="point-panel" tabindex="-1" class="panel" aria-label="Panel B: independent temporal point-resolution audit">
<div class="eyebrow">B / INDEPENDENT SCALAR FIXTURE</div><h2>PVP events và point-resolution</h2>
<label>Nguồn B <select id="point-select" aria-label="Chọn nguồn temporal"></select></label><p id="point-source" class="small"></p>
<canvas id="event-canvas" width="720" height="145" aria-label="PVP candidate và event được lưu, không phải motion phases"></canvas>
<div class="controls"><button id="point-play" type="button" aria-pressed="false" aria-controls="event-canvas point-scrub">Phát B</button><input id="point-scrub" aria-label="Tua frame B" aria-describedby="keyboard-help point-clock" type="range" min="0" max="0" value="0" step="1"></div><output id="point-clock" for="point-scrub"></output>
<p id="candidate-count" class="small"></p><div id="point-status" class="status unavailable"></div>
<p id="motion-status" class="unavailable">Motion phase: Unavailable</p><p id="pause-status" class="unavailable">Arrival / departure / pause: Unavailable (null)</p>
<h3>Event overlay · riêng nguồn B</h3><pre id="events"></pre><h3>Lý do unavailable</h3><pre id="point-reasons"></pre>
<table><thead><tr><th>PVP frames S → B → E</th><th>Điều kiện raw</th><th>Point-turn</th></tr></thead><tbody id="pvp-rows"></tbody></table>
<details><summary>Input/run diagnostics và original flags</summary><pre id="point-diagnostics"></pre></details>
<details><summary>Profile / policy / provenance của panel B</summary><pre id="point-provenance"></pre></details></section>
</div>
<section class="panel footer"><h2>Evidence và giới hạn</h2><p id="noise-summary"></p>
<p>Visibility/presence, imputation provenance hoặc confidence threshold không tự trở thành uncertainty/noise bound được hiệu chuẩn cho signal sample. Detector work dừng ở abstention.</p>
<p>Không có form score, lời khuyên kỹ thuật hoặc kết luận đúng/sai. Không có validation trên người tập.</p>
<details><summary>Read-only noise evidence review</summary><pre id="noise-review"></pre></details>
<h3>Saved JSON snapshots và manifest</h3><ul id="source-links"></ul><p><a href="manifest.json">Manifest / byte hashes</a> · <a href="presentation.json">Presentation DTO</a> · <a href="noise-review.json">Noise review</a></p></section>
<noscript>Demo cần JavaScript cục bộ để vẽ các joints đã lưu; không dùng mạng. Có thể mở JSON snapshots để xem dữ liệu.</noscript>
</main>
<script id="showcase-data" type="application/json">__SHOWCASE_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('showcase-data').textContent),el=id=>document.getElementById(id);
const names={squat_hip_angle_left_deg:'Góc hông trái',squat_hip_angle_right_deg:'Góc hông phải',squat_knee_angle_left_deg:'Góc gối trái',squat_knee_angle_right_deg:'Góc gối phải',squat_torso_angle_vertical_deg:'Độ nghiêng thân'};
const links=[['nose','left_shoulder'],['nose','right_shoulder'],['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']];
const make=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
let ai=0,bi=0,af=0,bf=0,aplay=false,bplay=false,alast=0,blast=0;
const pc=el('pose-canvas').getContext('2d'),ec=el('event-canvas').getContext('2d');
function syncPlayState(){el('pose-play').ariaPressed=String(aplay);el('point-play').ariaPressed=String(bplay);}
data.feedback_cases.forEach((d,i)=>{const option=make('option',d.video_id+' · '+d.fps+' FPS');option.value=String(i);el('feedback-select').append(option);});
data.point_cases.forEach((d,i)=>{const option=make('option',d.case_label+' · '+d.fps+' FPS · '+d.candidate_count+' PVP · raw '+d.necessary_condition_count);option.value=String(i);el('point-select').append(option);});
function drawPose(f){
 pc.fillStyle='#0b1725';pc.fillRect(0,0,640,640);pc.fillStyle='#b7c6d8';pc.font='17px system-ui';pc.fillText('SAVED SYNTHETIC JOINTS · JSON ONLY',20,32);
 if(!f)return;
 const joints=new Map(f.joints.map(j=>[j.joint,j]));
 const valid=j=>j&&j.status!=='missing'&&Number.isFinite(j.x)&&Number.isFinite(j.y)&&j.x>=0&&j.x<=1&&j.y>=0&&j.y<=1;
 for(const [a,b] of links){const A=joints.get(a),B=joints.get(b);if(!valid(A)||!valid(B))continue;pc.strokeStyle=A.status==='observed'&&B.status==='observed'?'#6fded9':'#ffcc71';pc.lineWidth=5;pc.beginPath();pc.moveTo(A.x*640,A.y*640);pc.lineTo(B.x*640,B.y*640);pc.stroke();}
 for(const j of joints.values()){if(!valid(j))continue;pc.fillStyle=j.status==='observed'?'#6fded9':'#ffcc71';pc.beginPath();pc.arc(j.x*640,j.y*640,6,0,2*Math.PI);pc.fill();}
}
function showA(){
 const d=data.feedback_cases[ai],f=d.frames[af];el('pose-scrub').max=Math.max(0,d.frames.length-1);el('pose-scrub').value=af;el('pose-play').disabled=!d.frames.length;
 syncPlayState();
 el('pose-source').textContent=d.video_id+' · '+d.source.frame_count+' frames · '+d.fps+' FPS · nguồn A độc lập';
 drawPose(f);el('metrics').replaceChildren();if(!f){el('pose-clock').textContent='Unavailable — không có frame';return;}
 el('pose-clock').textContent='A · Frame '+f.frame_index+' · '+f.timestamp_seconds.toFixed(3)+' s';
 el('pose-scrub').ariaValueText=el('pose-clock').textContent;
 el('pose-quality').textContent='Quality sequence: '+d.quality.eligibility.action+' · skeleton source: '+f.overlay_source+' · saved raw frame '+f.source_frame_hash;
 for(const m of f.metrics){const card=make('div','');card.className='metric';card.append(make('div',names[m.feature_id]||m.feature_id));
  const v=make('strong',m.status==='available'&&m.value!==null?m.value.toFixed(1)+'°':'Unavailable');v.className='value'+(m.value===null?' unavailable':'');card.append(v);
  card.append(make('p',m.status+' · quality '+m.quality_action+' · '+m.measurement_validation_status));
  const detail=make('p',m.reasons.length?m.reasons.join(', '):m.joint_inputs.map(j=>j.joint+': '+j.status).join(' · '));detail.className='small';card.append(detail);card.title=m.feature_id+' / '+(m.observation_hash||'NOT_RUN');el('metrics').append(card);}
 el('pose-provenance').textContent=JSON.stringify({source:d.source,source_hash:d.source_hash,feedback_hash:d.feedback_hash,registry_version:d.registry_version,registry_hash:d.registry_hash,temporal_input_hash:d.temporal_input_hash,original_segmentation_hash:d.segmentation_hash,profile:d.profile,profile_hash:d.profile_hash,legacy_phase_displayed:false,current_frame:f.frame_index,source_frame_hash:f.source_frame_hash,metric_links:f.metrics.map(m=>({feature:m.feature_id,observation_hash:m.observation_hash,input_hash:m.input_hash,evidence:m.evidence,temporal_analysis_eligible:m.temporal_analysis_eligible}))},null,2);
}
function showB(){
 const d=data.point_cases[bi],s=d.samples[bf],events=d.events.filter(e=>e.frame_index===bf),byId=new Map(d.events.map(e=>[e.event_id,e]));
 syncPlayState();
 el('point-scrub').max=Math.max(0,d.frame_count-1);el('point-scrub').value=bf;el('point-play').disabled=!d.frame_count;
 el('point-source').textContent=d.video_id+' · '+d.frame_count+' frames · '+d.fps+' FPS · '+d.evidence_basis+' · nguồn B độc lập';
 el('point-clock').textContent=s?'B · Frame '+s.frame_index+' · '+s.timestamp_seconds.toFixed(3)+' s · '+s.side_source:'Unavailable — không có sample';
 el('point-scrub').ariaValueText=el('point-clock').textContent;
 el('candidate-count').textContent=d.candidate_count+' PVP candidates (không phải true rep count) · raw conditions '+d.necessary_condition_count;
 el('point-status').textContent='Point-turn: Unavailable · '+d.accepted_point_count+' accepted point-turn events';
 el('motion-status').textContent='Motion phase: Unavailable · '+(d.candidate_count?'bottom_bounds_unresolved':'no_full_pvp_candidate');
 el('pause-status').textContent='Arrival / departure / pause: Unavailable (null), không phải 0 s';
 const roles=[];for(const c of d.cycles){for(const [id,role] of [[c.start_event_id,'top_start'],[c.end_event_id,'top_end'],[c.bottom.anchor_event_id,'bottom_anchor']]){if(byId.get(id).frame_index===bf)roles.push(c.cycle_id+': '+role);}}
 el('events').textContent=events.length?events.map(e=>e.kind+' @ frame '+e.frame_index).join('\n')+'\n'+roles.join('\n'):'Không có event được lưu tại frame này.';
 const active=d.checks.filter(c=>c.candidate_support.start_frame<=bf&&bf<=c.candidate_support.end_frame);
 const reasons=active.flatMap(c=>c.reasons).concat(d.diagnostics.filter(r=>!r.support||(r.support.start_frame<=bf&&bf<=r.support.end_frame)));
 el('point-reasons').textContent=reasons.length?reasons.map(r=>r.reason+' · '+r.category+' · '+r.stage+' · '+r.source_tier).join('\n'):'Không có phase interval được lưu cho frame này; không suy phase từ event.';
 el('pvp-rows').replaceChildren();d.cycles.forEach((c,i)=>{const row=make('tr',''),check=d.checks[i];row.append(make('td',[c.start_event_id,c.bottom.anchor_event_id,c.end_event_id].map(id=>byId.get(id).frame_index).join(' → ')),make('td',String(check.necessary_conditions_met)),make('td','Unavailable: '+check.reasons.map(r=>r.reason).join(', ')));el('pvp-rows').append(row);});
 ec.fillStyle='#0b1725';ec.fillRect(0,0,720,145);ec.fillStyle='#b7c6d8';ec.font='12px system-ui';ec.fillText('PVP spans / event markers · KHÔNG phải motion phase',15,20);
 const x=i=>25+670*i/Math.max(1,d.frame_count-1);for(const c of d.cycles){ec.strokeStyle='#475e71';ec.lineWidth=7;ec.beginPath();ec.moveTo(x(byId.get(c.start_event_id).frame_index),65);ec.lineTo(x(byId.get(c.end_event_id).frame_index),65);ec.stroke();}
 for(const e of d.events){ec.fillStyle=e.kind==='top'?'#89dfd9':'#f3c986';ec.beginPath();ec.arc(x(e.frame_index),65,5,0,2*Math.PI);ec.fill();}
 ec.strokeStyle='#eef4fc';ec.lineWidth=1;ec.beginPath();ec.moveTo(x(bf),35);ec.lineTo(x(bf),110);ec.stroke();
 el('point-diagnostics').textContent=JSON.stringify({diagnostics:d.diagnostics,original_flags:d.flags},null,2);
 el('point-provenance').textContent=JSON.stringify({producer:d.producer_version,policy:d.policy,policy_hash:d.policy_hash,base_policy:d.base_policy,base_policy_hash:d.base_policy_hash,profile:d.profile,profile_hash:d.profile_hash,video_id:d.video_id,source_hash:d.source_hash,input_hash:d.input_hash,audit_hash:d.audit_hash,base_hash:d.base_hash,quality:'No QualityReport on this pure scalar fixture',accepted_point_count:d.accepted_point_count},null,2);
}
el('feedback-select').addEventListener('change',e=>{ai=Number(e.target.value);af=0;aplay=false;el('pose-play').textContent='Phát A';showA();});
el('point-select').addEventListener('change',e=>{bi=Number(e.target.value);bf=0;bplay=false;el('point-play').textContent='Phát B';showB();});
el('pose-scrub').addEventListener('input',e=>{af=Number(e.target.value);aplay=false;el('pose-play').textContent='Phát A';showA();});
el('point-scrub').addEventListener('input',e=>{bf=Number(e.target.value);bplay=false;el('point-play').textContent='Phát B';showB();});
el('pose-play').addEventListener('click',()=>{aplay=!aplay;alast=performance.now();if(aplay&&af===data.feedback_cases[ai].frames.length-1)af=0;el('pose-play').textContent=aplay?'Dừng A':'Phát A';showA();});
el('point-play').addEventListener('click',()=>{bplay=!bplay;blast=performance.now();if(bplay&&bf===data.point_cases[bi].frame_count-1)bf=0;el('point-play').textContent=bplay?'Dừng B':'Phát B';showB();});
function tick(now){for(const panel of ['A','B']){const on=panel==='A'?aplay:bplay;if(!on)continue;const fps=panel==='A'?data.feedback_cases[ai].fps:data.point_cases[bi].fps,last=panel==='A'?alast:blast;if(now-last<1000/fps)continue;const step=Math.floor((now-last)*fps/1000);if(panel==='A'){af=Math.min(data.feedback_cases[ai].frames.length-1,af+step);alast+=step*1000/fps;showA();if(af===data.feedback_cases[ai].frames.length-1){aplay=false;el('pose-play').textContent='Phát A';}}else{bf=Math.min(data.point_cases[bi].frame_count-1,bf+step);blast+=step*1000/fps;showB();if(bf===data.point_cases[bi].frame_count-1){bplay=false;el('point-play').textContent='Phát B';}}}syncPlayState();requestAnimationFrame(tick);}
el('noise-summary').textContent=data.noise_review?data.noise_review.summary:'Noise review chưa được đính kèm; không suy noise-free từ provenance.';
el('noise-review').textContent=JSON.stringify(data.noise_review,null,2);
for(const ref of data.source_refs){const li=make('li',''),a=make('a',ref.role+' · '+ref.label);a.href=ref.snapshot_path;li.append(a,make('span',' · SHA-256 '+ref.sha256));el('source-links').append(li);}
showA();showB();requestAnimationFrame(tick);
</script></html>'''
