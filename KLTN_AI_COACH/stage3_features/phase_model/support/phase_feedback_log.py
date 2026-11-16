"""Ghi log JSONL các chỉnh sửa từ auto proposal sang nhãn đã review."""
from __future__ import annotations
import json,os
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Dict
from core.phase_schema import PHASE_ORDER,PhaseSet,load_phases

def _mapping(value: Any) -> Dict[str,Any]:
    if isinstance(value,(str,Path)): value=load_phases(value)
    if isinstance(value,PhaseSet):
        rep=value.reps[0];return {'video':value.video,'fps':value.fps,'total_frames':value.total_frames,'marked_by':value.marked_by,'auto_meta':value.auto_meta,'auto_phases':value.auto_phases,'phases':{p:rep.phases[p].to_dict() for p in PHASE_ORDER}}
    if isinstance(value,dict):return value
    raise TypeError('auto/final phải là PhaseSet, dict hoặc đường dẫn .phases.json')
def _edge_map(data):
    phases=data.get('phases',{});out={}
    for name in PHASE_ORDER:
        w=phases.get(name,{})
        if w and not w.get('absent') and 'start_frame' in w:out[f'start:{name}']=int(w['start_frame'])
    present=[p for p in PHASE_ORDER if f'start:{p}' in out]
    if present:out['end:last']=int(phases[present[-1]]['end_frame'])
    return out
def log_review(auto:Any,final:Any,path:Any='data/phase_feedback/log.jsonl')->Dict[str,Any]:
    a,f=_mapping(auto),_mapping(final);ae,fe=_edge_map(a),_edge_map(f);errors={k:(fe.get(k)-ae.get(k) if k in ae and k in fe else None) for k in sorted(set(ae)|set(fe))}
    edits=(f.get('review') or {}).get('edits',[]);record={'timestamp':datetime.now(timezone.utc).isoformat(timespec='seconds'),'video':f.get('video') or a.get('video'),'model_version':(a.get('auto_meta') or {}).get('model_version'),'phase_source':(a.get('auto_meta') or {}).get('phase_source'),'boundary_conf':(a.get('auto_meta') or {}).get('boundary_conf'),'boundary_error_frames':errors,'n_boundaries_edited':sum(v not in (0,None) for v in errors.values()),'n_edit_events':len(edits),'seconds_spent':(f.get('review') or {}).get('seconds_spent'),'marked_by':f.get('marked_by')}
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);line=(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o666)
    try:os.write(fd,line)
    finally:os.close(fd)
    return record
