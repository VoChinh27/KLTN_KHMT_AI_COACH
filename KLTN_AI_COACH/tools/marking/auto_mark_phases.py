"""Sinh proposal .phases.json tá»« pose cache báº±ng model Ä‘Ã£ Ä‘Äƒng kÃ½."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from pathlib import Path
import numpy as np
from core.phase_schema import PhaseSet,Rep,PhaseWindow,save_phases
from stage3_features.phase_model.core.phase_dataset import read_pose_cache
from stage3_features.phase_model.inference.phase_segmenter import PhaseSegmenter

def to_array(poses):
    return np.stack([np.concatenate((p.xy,np.asarray(p.conf).reshape(-1,1)),axis=1) for p in poses]).astype(np.float32)
def is_coach(stem): return stem.lower().startswith(('coach_','hlv'))
def mark_one(segmenter,stem,cache_dir,out_dir,overwrite=False):
    src=cache_dir/f'{stem}.json';dst=out_dir/f'{stem}.phases.json'
    if dst.exists() and not overwrite: raise FileExistsError(f'{dst} exists; use --overwrite to replace it')
    poses,fps=read_pose_cache(src)
    if not poses or not np.isfinite(fps) or fps<=0: raise ValueError(f'pose/fps khÃ´ng há»£p lá»‡: {src}')
    arr=to_array(poses);proposal=segmenter.propose(arr,fps,is_coach(stem),video_id=f'{stem}.mp4')
    rep=Rep()
    for name,win in proposal.windows.items():
        rep.phases[name]=PhaseWindow.missing() if win is None else PhaseWindow(int(win[0]),int(win[1]))
    ps=PhaseSet(video=f'{stem}.mp4',fps=float(fps),total_frames=len(poses),reps=[rep],marked_by='auto',auto_meta={**proposal.auto_meta(),'phase_source':f'model:{proposal.model_version}'},auto_phases={name:({'absent':True} if win is None else {'start_frame':int(win[0]),'end_frame':int(win[1])}) for name,win in proposal.windows.items()})
    warnings=ps.validate()
    if warnings: raise ValueError('proposal khÃ´ng há»£p lá»‡: '+'; '.join(warnings))
    save_phases(dst,ps);return dst,proposal

def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--video',help='Ä‘Æ°á»ng dáº«n video hoáº·c stem; cáº§n pose cache cÃ¹ng stem');g.add_argument('--folder',type=Path,help='folder video/clip; dÃ¹ng stem Ä‘á»ƒ tra pose cache');ap.add_argument('--pose-cache-dir',type=Path,default=Path('data/pose_cache'));ap.add_argument('--out-dir',type=Path,default=Path('data/phases_auto'));ap.add_argument('--model-dir',type=Path,default=Path('models/phase_segmenter'));ap.add_argument('--version');ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
    stems=[Path(args.video).stem] if args.video else sorted({p.stem for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4','.mov','.avi','.mkv')})
    if not stems: raise SystemExit('KhÃ´ng tÃ¬m tháº¥y video/clip.')
    segmenter=PhaseSegmenter.from_registry(args.model_dir,args.version);args.out_dir.mkdir(parents=True,exist_ok=True);failed=0
    for stem in stems:
        try:
            path,proposal=mark_one(segmenter,stem,args.pose_cache_dir,args.out_dir,args.overwrite)
            print(f'{stem}: {path} | version={proposal.model_version} | confidence={proposal.boundary_conf.tolist()} | flags={proposal.flags}')
        except Exception as exc:failed+=1;print(f'ERROR {stem}: {type(exc).__name__}: {exc}')
    if failed: raise SystemExit(1)
if __name__=='__main__':main()

