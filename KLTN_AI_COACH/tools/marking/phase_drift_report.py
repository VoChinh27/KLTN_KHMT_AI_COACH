"""BÃ¡o cÃ¡o drift Ä‘á»™ tin cáº­y/sai sá»‘ tá»« phase_feedback/log.jsonl."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,csv,json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

def load_rows(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for no,line in enumerate(f,1):
            if not line.strip():continue
            try:rows.append(json.loads(line))
            except json.JSONDecodeError as exc:raise ValueError(f'{path}:{no}: JSONL lá»—i: {exc}')
    return rows
def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        ts=str(r.get('timestamp',''));period=ts[:7] if len(ts)>=7 else 'unknown'
        conf=r.get('boundary_conf') or []
        level='unknown' if not conf else ('low' if float(np.nanmean(conf))<.60 else ('medium' if float(np.nanmean(conf))<.85 else 'high'))
        errs=r.get('boundary_error_frames') or {}
        vals=[abs(float(v)) for v in errs.values() if v is not None]
        groups[(period,level)].append((vals,int(r.get('n_boundaries_edited') or 0),float(np.mean(conf)) if conf else float('nan')))
    out=[]
    for (period,level),items in sorted(groups.items()):
        vals=[x for row,_,_ in items for x in row];edited=[n for _,n,_ in items];cs=[c for _,_,c in items if np.isfinite(c)]
        out.append({'period':period,'confidence_band':level,'n_clips':len(items),'mae_abs_frames':float(np.mean(vals)) if vals else None,'p90_abs_frames':float(np.percentile(vals,90)) if vals else None,'edited_boundaries_per_clip':float(np.mean(edited)) if edited else None,'mean_confidence':float(np.mean(cs)) if cs else None})
    return out
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=Path('data/phase_feedback/log.jsonl'));ap.add_argument('--out-dir',type=Path,default=Path('reports/phase_drift'));args=ap.parse_args();rows=load_rows(args.input);summary=summarize(rows);args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'phase_drift.json').write_text(json.dumps({'n_records':len(rows),'groups':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.out_dir/'phase_drift.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=list(summary[0]) if summary else ['period','confidence_band','n_clips','mae_abs_frames','p90_abs_frames','edited_boundaries_per_clip','mean_confidence'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(summary)
    for row in summary:print(row)
    print(f'records={len(rows)}; report={args.out_dir}')
if __name__=='__main__':main()


