"""Một cửa chọn nguồn ranh giới: manual/reviewed → auto → model → heuristic."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any,Callable,Dict,Mapping,Optional,Sequence,Tuple
import numpy as np
from core.phase_schema import PHASE_ORDER,PhaseSet,Rep,save_phases,load_phases
logger=logging.getLogger(__name__)

def _windows(rep_or_map: Any) -> Optional[Dict[str,Tuple[int,int]]]:
    if rep_or_map is None:return None
    if isinstance(rep_or_map,(list,tuple)) and len(rep_or_map)==5:
        edges=[int(x) for x in rep_or_map]
        if any(b<a for a,b in zip(edges,edges[1:])):raise ValueError('heuristic boundaries are not monotonic')
        return {name:(edges[i],edges[i+1]-1) for i,name in enumerate(PHASE_ORDER) if edges[i+1]>edges[i]}
    if hasattr(rep_or_map,'phases') and not isinstance(rep_or_map,Mapping):
        rep_or_map=rep_or_map.phases
    if not isinstance(rep_or_map,Mapping):return None
    raw=rep_or_map.get('phases',rep_or_map)
    if not isinstance(raw,Mapping):return None
    out={}
    for name in PHASE_ORDER:
        value=raw.get(name)
        if value is None:continue
        if hasattr(value,'absent'):
            if value.absent:continue
            start,end=value.start_frame,value.end_frame
        elif isinstance(value,Mapping):
            if value.get('absent'):continue
            start=value.get('start_frame',value.get('start'));end=value.get('end_frame',value.get('end'))
        else:
            try:start,end=value
            except (TypeError,ValueError):continue
        if start is None or end is None:continue
        start,end=int(start),int(end)
        if start<0 or end<start:raise ValueError(f'Pha {name} có cửa sổ không hợp lệ: {(start,end)}')
        out[name]=(start,end)
    return out or None

def _pose_array(poses: Any) -> np.ndarray:
    arr=np.asarray(poses) if isinstance(poses,np.ndarray) else None
    if arr is not None and arr.ndim==3:
        if arr.shape[1:] not in ((17,2),(17,3)):raise ValueError(f'pose array sai shape: {arr.shape}')
        if arr.shape[2]==2:arr=np.concatenate((arr,np.isfinite(arr).all(axis=2,keepdims=True).astype(float)),axis=2)
        return arr.astype(np.float32)
    rows=[]
    for pose in poses:
        xy=np.asarray(pose.xy,dtype=np.float32);conf=np.asarray(pose.conf,dtype=np.float32).reshape(-1,1)
        if xy.shape!=(17,2) or len(conf)!=17:raise ValueError('Mỗi pose phải có 17 keypoint COCO')
        rows.append(np.concatenate((xy,conf),axis=1))
    if not rows:raise ValueError('pose sequence rỗng')
    return np.stack(rows)

def _read_set(path:Path):
    ps=load_phases(path)
    if ps.validate():raise ValueError('; '.join(ps.validate()))
    rep=ps.reps[0]
    return ps,{n:(int(w.start_frame),int(w.end_frame)) for n,w in rep.present()}

def _persist_auto(stem:str,fps:float,total_frames:int,windows:Mapping[str,Tuple[int,int]],
                  source:str,meta:Optional[dict],auto_dir:Path)->None:
    """Persist a proposal for the review UI without overwriting an existing one."""
    target=auto_dir/f'{stem}.phases.json'
    if target.exists():return
    auto_dir.mkdir(parents=True,exist_ok=True)
    rep=Rep.from_windows({name:windows.get(name) for name in PHASE_ORDER})
    auto_phases={name:(rep.phases[name].to_dict()) for name in PHASE_ORDER}
    ps=PhaseSet(video=f'{stem}.mp4',fps=float(fps),total_frames=int(total_frames),reps=[rep],
                marked_by='auto',auto_meta={**(meta or {}),'phase_source':source},auto_phases=auto_phases)
    errors=ps.validate()
    if errors:raise ValueError('refusing to persist invalid auto proposal: '+'; '.join(errors))
    save_phases(target,ps)

def _confidence(meta:Optional[dict]) -> Tuple[float,list]:
    meta=meta or {};vals=[float(v) for v in (meta.get('boundary_conf') or []) if v is not None and np.isfinite(float(v))]
    return (min(vals) if vals else 0.0,list(meta.get('flags') or []))

def resolve_phases(video_id: str, poses: Any, fps: float, cfg: Mapping[str,Any], *,
                   manual_windows: Optional[Mapping[str,Tuple[int,int]]]=None,
                   manual_source: Optional[str]=None,
                   heuristic_detector: Optional[Callable[...,Any]]=None
                   ) -> Tuple[Optional[Dict[str,Tuple[int,int]]],str,bool]:
    """Trả ``(windows, phase_source, needs_review)``; tự động mặc định phải review.

    `auto` chưa duyệt chỉ được dùng để chấm nếu cấu hình bật rõ
    `allow_auto_unreviewed` và mọi confidence/quality gate đều đạt.
    """
    if not np.isfinite(fps) or fps<=0:raise ValueError('fps phải hữu hạn và > 0')
    cfg=dict(cfg);segment_cfg=cfg.get('phase_segmenter') or {};stem=Path(str(video_id)).name
    if stem.lower().endswith('.phases.json'):stem=stem[:-len('.phases.json')]
    else:stem=Path(stem).stem
    # Manual source from the main phases directory: verify its marker status.
    if manual_windows:
        source=str(manual_source or 'manual:explicit')
        source_path=source.split(':',1)[1] if source.startswith('manual:') else None
        marker='manual';meta={}
        if source_path:
            candidate=Path(source_path)
            if candidate.exists():
                try:
                    ps,_=_read_set(candidate);marker=ps.marked_by;meta=ps.auto_meta or {}
                except Exception as exc:logger.warning('Cannot inspect phase marker %s: %s',candidate,exc)
        picked=_windows(dict(manual_windows))
        if marker in ('manual','auto_reviewed'):
            prefix='manual' if marker=='manual' else 'auto_reviewed'
            return picked,f'{prefix}:{source_path or source}',False
        if marker=='auto':
            conf,flags=_confidence(meta)
            allowed=bool(segment_cfg.get('allow_auto_unreviewed',cfg.get('allow_auto_unreviewed',False)))
            minimum=float(segment_cfg.get('min_auto_confidence',0.85))
            needs=not (allowed and conf>=minimum and not flags)
            return picked,f'auto-unreviewed:{conf:.3f}',needs

    # Approved labels can be read directly by the resolver when caller did not load them.
    phases_dir=Path(segment_cfg.get('manual_phases_dir',cfg.get('phases_dir','data/phases')))
    manual_path=phases_dir/f'{stem}.phases.json'
    if manual_path.exists():
        try:
            ps,win=_read_set(manual_path)
            if ps.marked_by in ('manual','auto_reviewed'):
                label='manual' if ps.marked_by=='manual' else 'auto_reviewed'
                return win,f'{label}:{manual_path}',False
            conf,flags=_confidence(ps.auto_meta);allowed=bool(segment_cfg.get('allow_auto_unreviewed',False));minimum=float(segment_cfg.get('min_auto_confidence',0.85))
            return win,f'auto-unreviewed:{conf:.3f}',not(allowed and conf>=minimum and not flags)
        except Exception as exc:logger.warning('Ignoring invalid phase label %s: %s',manual_path,exc)

    # Existing auto proposal gets priority over making a new one.
    auto_dir=Path(segment_cfg.get('phases_auto_dir',cfg.get('phases_auto_dir','data/phases_auto')))
    auto_path=auto_dir/f'{stem}.phases.json'
    if auto_path.exists():
        try:
            ps,win=_read_set(auto_path)
            if ps.marked_by=='auto_reviewed':return win,f'auto_reviewed:{auto_path}',False
            if ps.marked_by=='manual':return win,f'manual:{auto_path}',False
            conf,flags=_confidence(ps.auto_meta)
            allowed=bool(segment_cfg.get('allow_auto_unreviewed',False));minimum=float(segment_cfg.get('min_auto_confidence',0.85))
            return win,f'auto-unreviewed:{conf:.3f}',not(allowed and conf>=minimum and not flags)
        except Exception as exc:logger.warning('Ignoring invalid auto proposal %s: %s',auto_path,exc)

    # Model proposal.
    model_path=segment_cfg.get('model_path',cfg.get('phase_model_dir','models/phase_segmenter'))
    try:
        from stage3_features.phase_model.inference.phase_segmenter import PhaseSegmenter
        pose_array=_pose_array(poses)
        proposal=PhaseSegmenter.from_registry(model_path,segment_cfg.get('model_version')).propose(
            pose_array,float(fps),is_coach=stem.lower().startswith(('coach_','hlv')),
            front_side_hint=cfg.get('front_side_hint'),video_id=stem)
        win=_windows(proposal.windows);meta=proposal.auto_meta()
        if win:_persist_auto(stem,fps,len(pose_array),win,f'model:{proposal.model_version}',meta,auto_dir)
        conf,flags=_confidence(meta);allowed=bool(segment_cfg.get('allow_auto_unreviewed',False));minimum=float(segment_cfg.get('min_auto_confidence',0.85))
        return win,f'auto-unreviewed:{conf:.3f}',not(allowed and conf>=minimum and not flags)
    except (FileNotFoundError,ImportError):
        logger.info('No registered phase model; proceeding to heuristic fallback.')
    except Exception as exc:logger.warning('Phase model proposal failed; using heuristic fallback: %s',exc)

    # Legacy heuristic is supplied by the integrating pipeline, avoiding a hard
    # dependency on FeatureExtractor internals in this resolver.
    detector=heuristic_detector or cfg.get('heuristic_detector')
    if callable(detector):
        try:
            result=detector(poses,float(fps));win=_windows(result)
            if win:
                total=len(_pose_array(poses))
                _persist_auto(stem,fps,total,win,'heuristic',{'flags':['heuristic_requires_review'],'boundary_conf':[]},auto_dir)
                return win,'heuristic',True
        except Exception as exc:logger.warning('Heuristic phase detection failed: %s',exc)
    return None,'none_needs_review',True
