"""Small offline v2 renderer: events and unavailable motion, no angle display."""
import html

from ..segmentation.models import strict_json
from ..segmentation.temporal_v2_models import SquatTemporalResultV2


def render_temporal_v2(result: SquatTemporalResultV2) -> str:
    if not isinstance(result, SquatTemporalResultV2):
        raise ValueError('Renderer requires explicit temporal v2, not implicit v1 migration')
    result = SquatTemporalResultV2.model_validate_json(strict_json(result), strict=True)
    payload = strict_json(result).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    source = result.legacy.input
    rows = []
    for cycle in result.cycles:
        pause = cycle.bottom.pause
        description = f'{pause.value.duration_seconds:.6f} s ([))' if pause.value else 'Unavailable — ' + ', '.join(r.reason for r in pause.reasons)
        rows.append('<tr><td>'+html.escape(cycle.cycle_id)+'</td><td>'+html.escape(cycle.bottom.mode)+
            '</td><td>'+html.escape(cycle.motion.status)+'</td><td>'+html.escape(description)+'</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="4">Unavailable — no confirmed full cycle; see reasons.</td></tr>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gym temporal v2 — synthetic</title><style>
body{font:16px system-ui;margin:2rem auto;max-width:1050px;padding:0 1rem;background:#101820;color:#eef4f6}
section{background:#1c2a35;padding:1rem;border-radius:12px;margin:1rem 0} .warning{color:#ffd17d}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:.6rem;border-bottom:1px solid #53616a}
input{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:.6rem}
</style><h1>Gym Result Feedback · temporal v2</h1>
<p class="warning">EXPERIMENTAL / NOT VALIDATED · SYNTHETIC ONLY</p>
<p>official_eligibility=false · assessment.enabled=false · G5.7 BLOCKED · G6 IN_PROGRESS</p>
<p>No athlete, angle calculation, form score or technical feedback. Events are separate from motion phases.</p>
<section><p>Source: '''+html.escape(source.video_id)+'''</p>
<p>Candidate cycles: '''+str(result.candidate_count)+''' (not a verified repetition count)</p>
<label>Frame / time <input id="scrub" type="range" min="0" max="'''+str(max(0,len(source.samples)-1))+'''" value="0" step="1"></label>
<p id="position"></p><h2>Motion phase</h2><p id="motion">Unavailable</p>
<h2>Event overlay / cycle roles</h2><pre id="events"></pre><h2>Reasons</h2><pre id="reasons"></pre></section>
<section><h2>Bottom components / Pause</h2><p>Unknown duration stays null; a bottom anchor does not establish arrival or departure.</p>
<table><thead><tr><th>Candidate</th><th>Bottom mode</th><th>Motion status</th><th>Pause</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></section>
<section><h2>Provenance</h2><pre id="provenance"></pre></section>
<script id="temporal-data" type="application/json">'''+payload+'''</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('temporal-data').textContent);
const byId=new Map(data.events.map(e=>[e.event_id,e]));
const samples=data.legacy.input.samples;
const put=(id,text)=>{document.getElementById(id).textContent=text;};
function show(){
 const i=Number(document.getElementById('scrub').value), sample=samples[i];
 if(!sample){put('motion','Unavailable — empty input');return;}
 put('position',`Frame ${i} · ${sample.timestamp_seconds.toFixed(6)} s · ${sample.side_source}`);
 let motion=null, status='unavailable', reasons=[], roles=[];
 const events=data.events.filter(e=>e.frame_index===i);
 for(const c of data.cycles){
  const refs=[[c.start_event_id,'top_start'],[c.end_event_id,'top_end'],[c.bottom.anchor_event_id,'bottom_anchor']];
  for(const comp of [c.bottom.arrival,c.bottom.departure]){if(comp.event)refs.push([comp.event.event_id,comp.event.kind]);}
  for(const [id,role] of refs){if(byId.get(id).frame_index===i)roles.push(`${c.cycle_id}: ${role}`);}
  if(byId.get(c.start_event_id).frame_index<=i && i<byId.get(c.end_event_id).frame_index){
   status=c.motion.status;reasons=c.motion.reasons;
   for(const span of c.motion.intervals){if(byId.get(span.start_event_id).frame_index<=i && i<byId.get(span.end_event_id).frame_index)motion=span.phase;}
  }
 }
 if(!motion && !reasons.length)reasons=data.diagnostics.filter(r=>!r.support||(r.support.start_frame<=i&&i<=r.support.end_frame));
 put('motion',motion||`Unavailable (${status})`);
 put('events',events.length?events.map(e=>e.kind).join(' + ')+String.fromCharCode(10)+roles.join(String.fromCharCode(10)):'No event at this frame');
 put('reasons',reasons.length?reasons.map(r=>`${r.reason} · ${r.category} · ${r.stage} · ${r.source_tier}`).join(String.fromCharCode(10)):(motion?'None':'outside_candidate'));
}
put('provenance',JSON.stringify({schema:data.segmentation_schema_version,contract:data.contract_version,
 producer:data.producer_version,policy:data.policy,policy_hash:data.policy_hash,input_hash:data.legacy.input_hash,
 source_hash:data.legacy.input.source_hash,profile_hash:data.legacy.profile_hash,legacy_hash:data.legacy_hash,
 validation:data.validation_status,official:data.official_eligibility,assessment:data.assessment},null,2));
document.getElementById('scrub').addEventListener('input',show);show();
</script></html>'''
