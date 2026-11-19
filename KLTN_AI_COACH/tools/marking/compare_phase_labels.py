"""So sÃ¡nh hai lÆ°á»£t gÃ¡n nhÃ£n Ä‘á»™c láº­p theo stem vÃ  tá»«ng ranh giá»›i."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, csv, re
from pathlib import Path
import numpy as np
from core.phase_schema import PHASE_ORDER, load_phases

BOUNDARIES=tuple(f"start:{p}" for p in PHASE_ORDER)+("end:last",)
def subject_of(stem: str) -> str:
    m=re.match(r"(?i)^(?:coach|trainee)_([^_]+)",stem)
    return m.group(1) if m else stem

def values(path: Path):
    ps=load_phases(path); rep=ps.reps[0]; present=rep.present(); out={k:None for k in BOUNDARIES}
    for name,w in present: out[f"start:{name}"]=w.start_frame
    if present: out["end:last"]=present[-1][1].end_frame
    return ps,out

def compare(dir_a: Path, dir_b: Path):
    a={p.name[:-len('.phases.json')]:p for p in dir_a.glob('*.phases.json')}
    b={p.name[:-len('.phases.json')]:p for p in dir_b.glob('*.phases.json')}
    rows=[]
    for stem in sorted(a.keys()|b.keys()):
        if stem not in a or stem not in b:
            rows.append(dict(stem=stem,subject=subject_of(stem),boundary='*',frame_a='',frame_b='',error='',status='missing_file_a' if stem not in a else 'missing_file_b'))
            continue
        _,va=values(a[stem]); _,vb=values(b[stem])
        for boundary in BOUNDARIES:
            x,y=va[boundary],vb[boundary]
            status='matched' if x is not None and y is not None else ('both_absent' if x is None and y is None else ('missing_a' if x is None else 'missing_b'))
            rows.append(dict(stem=stem,subject=subject_of(stem),boundary=boundary,frame_a='' if x is None else x,frame_b='' if y is None else y,error='' if x is None or y is None else y-x,status=status))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels-a',type=Path,required=True);p.add_argument('--labels-b',type=Path,required=True);p.add_argument('--out-csv',type=Path,required=True);args=p.parse_args()
    rows=compare(args.labels_a,args.labels_b);args.out_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.out_csv.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['stem','subject','boundary','frame_a','frame_b','error','status']);w.writeheader();w.writerows(rows)
    print(f"clips={len({r['stem'] for r in rows})}; rows={len(rows)}; csv={args.out_csv}")
    for b in BOUNDARIES:
        e=np.asarray([int(r['error']) for r in rows if r['boundary']==b and r['error']!=''],float)
        print(f"{b}: n={len(e)} median_abs={np.median(np.abs(e)) if len(e) else float('nan'):.2f} p90_abs={np.percentile(np.abs(e),90) if len(e) else float('nan'):.2f} max_abs={np.max(np.abs(e)) if len(e) else float('nan'):.2f}")
if __name__=='__main__': main()

