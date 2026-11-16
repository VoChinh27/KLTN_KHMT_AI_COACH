"""Kiá»ƒm kÃª nhÃ£n pha vÃ  pose cache; xuáº¥t prior thá»i lÆ°á»£ng thá»±c nghiá»‡m."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,csv,json
from pathlib import Path
import numpy as np
from core.phase_schema import PHASE_ORDER, load_phases
from stage3_features.phase_model.core.phase_dataset import list_clips, read_pose_cache, group_of
from stage3_features.phase_model.core.phase_decoder import DurationPriors

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phases-dir',type=Path,default=Path('data/phases'));ap.add_argument('--pose-cache-dir',type=Path,default=Path('data/pose_cache'));ap.add_argument('--report-dir',type=Path,default=Path('reports/phase_audit'));ap.add_argument('--priors-out',type=Path,default=Path('config/phase_duration_priors.json'));args=ap.parse_args()
    matched,missing_pose,missing_labels=list_clips(args.phases_dir,args.pose_cache_dir);phase_files=sorted(args.phases_dir.glob('*.phases.json'));durations={p:[] for p in PHASE_ORDER};rows=[];errors=[]
    for path in phase_files:
        stem=path.name[:-len('.phases.json')]
        try:
            ps=load_phases(path);errs=ps.validate()
            if errs: errors.extend(f'{path.name}: {e}' for e in errs)
            warnings=ps.warnings()
            pose_path=args.pose_cache_dir/f'{stem}.json';nan_ratio=float('nan');pose_fps=float('nan');pose_count=0
            if pose_path.exists():
                poses,pose_fps=read_pose_cache(pose_path);pose_count=len(poses);nan_ratio=float(np.mean([not np.isfinite(x.xy).all() for x in poses])) if poses else 1.0
            rep=ps.reps[0]
            for name,w in rep.present():
                seconds=w.n_frames/ps.fps;durations[name].append(seconds)
                rows.append({'stem':stem,'group':group_of(stem),'phase':name,'start_frame':w.start_frame,'end_frame':w.end_frame,'duration_frames':w.n_frames,'duration_s':seconds,'fps':ps.fps,'pose_frames':pose_count,'pose_fps':pose_fps,'pose_nan_ratio':nan_ratio,'warnings':' | '.join(warnings)})
            if warnings: print(f'WARNING {stem}: '+'; '.join(warnings))
        except Exception as exc: errors.append(f'{path.name}: {type(exc).__name__}: {exc}')
    priors=DurationPriors.fit(durations)
    args.priors_out.parent.mkdir(parents=True,exist_ok=True);priors.save(args.priors_out)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    with (args.report_dir/'phase_durations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['stem','group','phase','start_frame','end_frame','duration_frames','duration_s','fps','pose_frames','pose_fps','pose_nan_ratio','warnings'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={'phase_files':len(phase_files),'matched_pose':len(matched),'missing_pose':missing_pose,'pose_without_labels':missing_labels,'validation_errors':errors,'duration_summary':{p:{'n':len(v),'p50_s':float(np.median(v)) if v else None,'p90_s':float(np.percentile(v,90)) if v else None} for p,v in durations.items()},'duration_priors':priors.to_dict()}
    (args.report_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('phase_files','matched_pose','missing_pose','pose_without_labels')},ensure_ascii=False,indent=2));print(f'priors={args.priors_out}; report={args.report_dir}')
    if errors: print(f'validation_errors={len(errors)} (xem summary.json)')
if __name__=='__main__':main()

