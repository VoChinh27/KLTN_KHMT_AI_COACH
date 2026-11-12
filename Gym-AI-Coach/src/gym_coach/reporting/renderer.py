"""Standalone offline HTML/Canvas. No network, framework or domain calculation."""
import json

from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, GymError, Severity, StructuredFlag
from ..segmentation.models import strict_json
from .squat_models import SquatFeedbackResult


def render_feedback(result: SquatFeedbackResult) -> str:
    """Validate a detached snapshot before embedding; all strings use textContent."""
    try:
        if not isinstance(result, SquatFeedbackResult):
            raise ValueError('Expected SquatFeedbackResult')
        result = SquatFeedbackResult.model_validate_json(strict_json(result), strict=True)
        payload = json.dumps(result.model_dump(mode='json'), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, AttributeError) as exc:
        raise GymError(StructuredFlag(code=FlagCode.FEEDBACK_SERIALIZATION_ERROR,
            category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR, stage=Stage.REPORTING,
            source_tier=SourceTier.S3, message='Cannot render invalid feedback',
            evidence={'reason': str(exc)})) from exc
    # Prevent a data string from terminating the non-executable JSON script tag.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    payload = payload.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return _HTML.replace('__FEEDBACK_PAYLOAD__', payload)


_HTML = r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gym Result Feedback · Squat</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c1422;color:#e9eef6;line-height:1.5}
*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:24px}
h1{font-size:clamp(24px,4vw,34px);margin:4px 0}h2{font-size:18px;margin:0 0 12px}p{margin:8px 0}
small,.muted{color:#b4c2d5}.eyebrow{color:#74d8da;letter-spacing:.12em;font-size:12px}
.badge{display:inline-block;border:1px solid #bd964d;background:#382c18;color:#ffda8a;border-radius:5px;padding:5px 9px;font-size:12px;margin:4px 8px 4px 0}
.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:18px;margin:22px 0}
.panel{background:#121f31;border:1px solid #2d4059;border-radius:12px;padding:18px;min-width:0}
canvas{display:block;width:100%;border-radius:8px;background:#0b1726}#scene{max-height:510px;aspect-ratio:1;object-fit:contain}
.controls{display:flex;gap:12px;align-items:center;margin:14px 0}.controls input{width:100%;accent-color:#71dad8}
button{background:#19404c;border:1px solid #4e8792;color:#f1ffff;padding:8px 15px;border-radius:7px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}button:focus-visible,input:focus-visible{outline:3px solid #ffda8a;outline-offset:4px}
.phase{font-size:30px;font-weight:650;color:#89e3e0}.metric{padding:12px 0;border-bottom:1px solid #2d4059}.metric:last-child{border:0}
.metric header{display:flex;gap:8px;justify-content:space-between;align-items:center}.value{font-size:25px;font-variant-numeric:tabular-nums;white-space:nowrap}
.reason{font-size:12px;color:#ffda8a;overflow-wrap:anywhere}.detail{font-size:12px;color:#b4c2d5;overflow-wrap:anywhere}
.legend{font-size:12px;color:#b4c2d5}.legend span{margin-right:14px}.observed{color:#6fded9}.low{color:#ffcc71}
#graph{height:145px;margin-top:14px}.notice{border-left:3px solid #cb9d52;padding-left:12px;margin:12px 0}
details{margin:18px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#b4c2d5}
#timeline{display:flex;gap:2px;height:28px;margin:10px 0}.segment{min-width:1px;border-radius:3px;background:#2b626b}
.foot{font-size:12px;color:#b4c2d5}.links{display:flex;gap:16px;flex-wrap:wrap;margin-top:16px}a{color:#8fe5e2}
@media(max-width:760px){main{padding:14px}.layout{grid-template-columns:1fr}.panel{padding:14px}#scene{max-height:400px}}
</style>
<main>
<div class="eyebrow">GYM AI COACH / DESCRIPTIVE PROTOTYPE</div>
<h1>Gym Result Feedback</h1><p>BodyWeightSquats · Pha, số đo và trạng thái dữ liệu</p>
<div><span class="badge">EXPERIMENTAL / NOT VALIDATED</span><span class="badge" id="mode"></span><span class="badge">assessment.enabled=false</span></div>
<p class="muted" id="official"></p>
<div class="layout">
 <section class="panel" aria-label="Khung hình và overlay">
  <canvas id="scene" width="640" height="640" aria-label="Skeleton synthetic hoặc preview video gốc"></canvas>
  <div class="controls"><button id="play" type="button">Phát</button><input id="scrub" type="range" min="0" value="0" step="1" aria-label="Chọn frame"><output id="clock"></output></div>
  <div class="legend"><span class="observed">● observed</span><span class="low">● low_visibility</span><span>missing: không vẽ</span></div>
  <p class="detail" id="overlay"></p>
  <canvas id="graph" width="720" height="145" aria-label="Tín hiệu góc gối; gap không nối"></canvas>
  <p class="detail">Đồ thị mô tả góc gối, đơn vị °; không phải vùng kỹ thuật đạt/chưa đạt.</p>
 </section>
 <section class="panel" aria-label="Pha và số đo">
  <h2>Pha chuyển động · thử nghiệm</h2><div class="phase" id="phase"></div><p class="detail" id="phaseEvidence"></p>
  <div id="timeline" aria-label="Timeline chu kỳ ứng viên"></div><p class="detail" id="candidates"></p>
  <h2>Số đo theo frame</h2><div id="metrics"></div>
 </section>
</div>
<section class="panel"><h2>Phản hồi về dữ liệu</h2><div id="messages"></div>
<p class="foot">Không có chuẩn đúng/sai, điểm form hoặc chẩn đoán lỗi kỹ thuật. Nhãn UCF101 không phải ground truth về form.</p></section>
<details class="panel"><summary>Nguồn, policy và provenance</summary><pre id="audit"></pre></details>
<p class="foot">Timeline và profile S3 chưa validated. Chu kỳ chạm biên, gap hoặc đổi bên không được ghép thành rep. Official Stage 5.7 vẫn blocked.</p>
</main>
<script id="feedback-data" type="application/json">__FEEDBACK_PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('feedback-data').textContent);
const el=id=>document.getElementById(id), scene=el('scene'), ctx=scene.getContext('2d'), graph=el('graph'), g=graph.getContext('2d');
const raw=data.source.kind==='real_preview';let current=0,playing=false,last=0,pendingImage=null;
const names={squat_hip_angle_left_deg:'Góc hông trái',squat_hip_angle_right_deg:'Góc hông phải',squat_knee_angle_left_deg:'Góc gối trái',squat_knee_angle_right_deg:'Góc gối phải',squat_torso_angle_vertical_deg:'Độ nghiêng thân'};
const links=[['nose','left_shoulder'],['nose','right_shoulder'],['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']];
function text(tag,value,cls){const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n;}
el('mode').textContent=raw?'REAL VIDEO · BLOCKED':'SYNTHETIC FIXTURE';
el('official').textContent='Official gate: '+data.official_gate_status+' · '+(raw?'Chỉ preview; pose, quality, feature, temporal chưa chạy.':'Số đo và pha chỉ kiểm tra luồng synthetic.');
el('scrub').max=Math.max(0,data.frames.length-1);el('scrub').disabled=data.frames.length===0;
// Real preview has three sampled frames, not a continuous playback claim.
el('play').disabled=raw||data.frames.length===0;
for(const m of data.messages)el('messages').append(text('p',m.text,'notice'));
for(const flag of data.flags)el('messages').append(text('p',flag.code+': '+flag.message,'reason'));
el('audit').textContent=JSON.stringify({source:data.source,source_hash:data.source_hash,registry_version:data.registry_version,registry_hash:data.registry_hash,profile:data.segmentation.profile,profile_hash:data.segmentation.profile_hash,algorithm:data.segmentation.algorithm_version,segmentation_hash:data.segmentation_hash,quality:data.quality?{sequence:data.quality.eligibility,metrics:data.quality.metric_eligibility}:'NOT_RUN',flags:data.segmentation.flags,framewise_temporal_analysis_eligible:false,official_eligibility:false},null,2);
el('candidates').textContent=data.segmentation.status==='experimental_candidates'?data.segmentation.candidate_count+' chu kỳ ứng viên synthetic; chưa là rep đã kiểm chứng.':'Chu kỳ unavailable: '+data.segmentation.flags.map(f=>f.code).filter((x,i,a)=>a.indexOf(x)===i).join(', ');
for(const rep of data.segmentation.repetitions){const n=text('div','','segment');n.style.flex=String(rep.end_frame-rep.start_frame);n.title='PVP '+rep.start_frame+' → '+rep.bottom_frame+' → '+rep.end_frame;el('timeline').append(n);}
function background(label){ctx.fillStyle='#0b1726';ctx.fillRect(0,0,640,640);ctx.strokeStyle='#172a3d';for(let n=64;n<640;n+=64){ctx.beginPath();ctx.moveTo(n,0);ctx.lineTo(n,640);ctx.moveTo(0,n);ctx.lineTo(640,n);ctx.stroke();}ctx.fillStyle='#b4c2d5';ctx.font='17px system-ui';ctx.fillText(label,24,35);}
function drawOverlay(frame){const joints=new Map(frame.overlay_joints.map(j=>[j.joint,j]));const valid=j=>j&&j.status!=='missing'&&Number.isFinite(j.x)&&Number.isFinite(j.y)&&j.x>=0&&j.x<=1&&j.y>=0&&j.y<=1;
 for(const [a,b] of links){const A=joints.get(a),B=joints.get(b);if(!valid(A)||!valid(B))continue;ctx.strokeStyle=A.status==='observed'&&B.status==='observed'?'#6fded9':'#ffcc71';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(A.x*640,A.y*640);ctx.lineTo(B.x*640,B.y*640);ctx.stroke();}
 for(const j of joints.values()){if(!valid(j))continue;ctx.fillStyle=j.status==='observed'?'#6fded9':'#ffcc71';ctx.beginPath();ctx.arc(j.x*640,j.y*640,6,0,2*Math.PI);ctx.fill();}}
function drawGraph(frame){g.clearRect(0,0,720,145);g.fillStyle='#0b1726';g.fillRect(0,0,720,145);g.font='12px system-ui';g.fillStyle='#b4c2d5';g.fillText('180°',4,15);g.fillText('0°',4,135);
 const inp=data.segmentation.input;if(!inp){g.fillText('Tín hiệu temporal unavailable / NOT_RUN',60,75);return;}
 g.strokeStyle='#6fded9';g.lineWidth=2;g.beginPath();let open=false,previousSide=null;
 inp.samples.forEach((s,i)=>{if(s.value===null){open=false;previousSide=null;return;}const x=45+660*i/Math.max(1,inp.samples.length-1),y=130-115*s.value/180;if(!open||previousSide!==s.side_source)g.moveTo(x,y);else g.lineTo(x,y);open=true;previousSide=s.side_source;});g.stroke();
 const x=45+660*frame.frame_index/Math.max(1,data.source.frame_count-1);g.strokeStyle='#ffda8a';g.beginPath();g.moveTo(x,10);g.lineTo(x,135);g.stroke();}
function show(index){current=index;el('scrub').value=index;const frame=data.frames[index];if(!frame){background('Không có frame');el('phase').textContent='Unavailable';return;}
 el('clock').textContent=frame.timestamp_seconds.toFixed(2)+'s';el('phase').textContent=frame.phase;
 el('phaseEvidence').textContent='Frame '+frame.frame_index+' · '+(frame.phase_evidence?'Evidence '+frame.phase_evidence.start_frame+'–'+frame.phase_evidence.end_frame:frame.phase_reasons.join(', '));
 el('overlay').textContent=raw?'Ba frame preview rời rạc · không có skeleton suy đoán.':'Overlay: raw image_normalized_xy · không dùng tọa độ torso-normalized để vẽ. Skeleton tạo synthetic; không chạy MediaPipe.';
 background(raw?'Preview ảnh gốc · BLOCKED':'SYNTHETIC · native 2D skeleton');
 pendingImage=null;if(frame.preview_image){const image=new Image();pendingImage=image;image.onload=()=>{if(pendingImage!==image)return;background('');const scale=Math.min(640/image.width,640/image.height);ctx.drawImage(image,(640-image.width*scale)/2,(640-image.height*scale)/2,image.width*scale,image.height*scale);};image.onerror=()=>{if(pendingImage===image)background('Preview image unavailable');};image.src=frame.preview_image;}else drawOverlay(frame);
 el('metrics').replaceChildren();for(const m of frame.metrics){const card=text('div','','metric'),header=text('header','');header.append(text('span',names[m.feature_id]),text('strong',m.value===null?'—':m.value.toFixed(1)+'°','value'));card.append(header,text('div',m.status+' · quality '+m.quality_action+' · unvalidated','detail'));if(m.reasons.length)card.append(text('div',m.reasons.join(', '),'reason'));else card.append(text('div',m.joint_inputs.map(j=>j.joint+': '+j.status).join(' · '),'detail'));card.title=m.feature_id+' / '+(m.observation_hash||'NOT_RUN');el('metrics').append(card);}drawGraph(frame);}
el('scrub').addEventListener('input',e=>{playing=false;el('play').textContent='Phát';show(Number(e.target.value));});
el('play').addEventListener('click',()=>{playing=!playing;if(playing&&current===data.frames.length-1)show(0);last=performance.now();el('play').textContent=playing?'Dừng':'Phát';});
function tick(now){if(playing&&now-last>=1000/data.source.fps){const step=Math.floor((now-last)*data.source.fps/1000);last+=step*1000/data.source.fps;show(Math.min(data.frames.length-1,current+step));if(current===data.frames.length-1){playing=false;el('play').textContent='Phát';}}requestAnimationFrame(tick);}show(0);requestAnimationFrame(tick);
</script></html>'''
