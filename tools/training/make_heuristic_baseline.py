"""Sinh dá»± Ä‘oÃ¡n baseline tá»« heuristic cÅ©, lÆ°u Ä‘á»™c láº­p khá»i nhÃ£n tháº­t."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,json
from pathlib import Path
from typing import Any,Dict,Optional,Tuple
from stage3_features.phase_model.core.phase_dataset import load_all
from core.phase_schema import PhaseSet,Rep,PhaseWindow,PHASE_ORDER,save_phases

def _coerce_boundaries(raw: Any, n_frames: int) -> Rep:
    if isinstance(raw, Rep): return raw
    if isinstance(raw, dict) and 'boundaries' in raw: raw=raw['boundaries']
    if isinstance(raw, (list,tuple)) and len(raw)==5 and all(isinstance(x,(int,float)) for x in raw):
        b=[int(round(x)) for x in raw]; rep=Rep()
        for i,name in enumerate(PHASE_ORDER):
            if b[i+1]>b[i]: rep.phases[name]=PhaseWindow(b[i],b[i+1]-1)
        return rep
    if isinstance(raw, dict):
        rep=Rep()
        for name in PHASE_ORDER:
            val=raw.get(name)
            if val is None: continue
            if isinstance(val,dict):
                if val.get('absent'): continue
                a=val.get('start_frame',val.get('start'));z=val.get('end_frame',val.get('end'))
            else: a,z=val
            if a is not None and z is not None: rep.phases[name]=PhaseWindow(int(a),int(z))
        return rep
    raise TypeError('Heuristic pháº£i tráº£ Rep, 5 biÃªn half-open hoáº·c dict phase -> (start,end).')

def detect_heuristic(poses, fps: float, front_side_hint: Optional[str]=None) -> Rep:
    """Adapter vá»›i API cÅ©; khÃ´ng táº¡o má»‘c giáº£ náº¿u khÃ´ng tÃ¬m Ä‘Æ°á»£c káº¿t quáº£."""
    from stage3_features.feature_extractor import FeatureExtractor
    extractor=FeatureExtractor()
    normalizer=getattr(extractor,'normalizer',None)
    if normalizer is None or not hasattr(normalizer,'normalize_sequence'):
        raise RuntimeError('FeatureExtractor hiá»‡n táº¡i khÃ´ng cung cáº¥p normalizer.normalize_sequence')
    normalized=normalizer.normalize_sequence(poses,front_side_hint=front_side_hint)
    labels=None
    for name in ('_label_events','label_events','_detect_events'):
        fn=getattr(extractor,name,None)
        if callable(fn):
            try: labels=fn(poses,normalized,fps);break
            except TypeError: continue
    fn=getattr(extractor,'_detect_sub_phase_boundaries',None)
    if not callable(fn): raise RuntimeError('KhÃ´ng tÃ¬m tháº¥y FeatureExtractor._detect_sub_phase_boundaries')
    raw=fn(poses,normalized,fps,labels)
    return _coerce_boundaries(raw,len(poses))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phases-dir',type=Path,default=Path('data/phases'));ap.add_argument('--pose-cache-dir',type=Path,default=Path('data/pose_cache'));ap.add_argument('--out-dir',type=Path,default=Path('data/phases_baseline'));ap.add_argument('--group',choices=('all','coach','trainee'),default='all');args=ap.parse_args()
    clips=load_all(args.phases_dir,args.pose_cache_dir,group=args.group);args.out_dir.mkdir(parents=True,exist_ok=True);errors=[]
    for clip in clips:
        try:
            rep=detect_heuristic(clip.poses,clip.fps,clip.front_side_override)
            ps=PhaseSet(video=f'{clip.stem}.mp4',fps=clip.fps,total_frames=clip.total_frames,reps=[rep],marked_by='auto',auto_meta={'phase_source':'heuristic','baseline_only':True})
            save_phases(args.out_dir/f'{clip.stem}.phases.json',ps)
        except Exception as exc: errors.append({'stem':clip.stem,'error':f'{type(exc).__name__}: {exc}'})
    (args.out_dir/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'written={len(clips)-len(errors)}/{len(clips)}; errors={len(errors)}; out={args.out_dir}')
if __name__=='__main__':main()

