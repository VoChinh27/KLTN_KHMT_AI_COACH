import sys, glob, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import numpy as np, pandas as pd
from badminton3d.config import load_config
from badminton3d.analyze import build_context, load_annotation, analyze_context
from badminton3d.pose_io import load_pose
from badminton3d.reference import build_reference
cfg=load_config(); FE=["elbow_arm","wrist_height","trunk_lean","knee_mean","shoulder_arm"]
C={}
for f in sorted(glob.glob("Input/*_serve_*.mp4")+glob.glob("Input/_luu_tru/*_serve_*.mp4")):
    stem=os.path.basename(f)[:-4]; ann=load_annotation(f); pp=f"Output/{stem}/pose.csv" if os.path.exists(f"Output/{stem}/pose.csv") else f"Output/_luu_tru/{stem}/pose.csv"
    C[stem]=build_context(load_pose(pp),cfg,"serve",ann.get("racket_hand") or "auto",ann.get("serve_windows_s"))
ref=build_reference("serve",{n:C[n].units for n in ["coach_serve_02","coach_serve_07"]},cfg); tol={k:ref["phases"]["danh_cau"][k]["tol"] for k in FE}
def seq(c,rep):
    fps=c.pose.fps; a=max(0,rep.swing_start-int(0.2*fps)); b=min(rep.end,c.pose.n-1)
    X=np.stack([np.interp(np.arange(a,b+1),np.arange(c.pose.n)[~np.isnan(c.feats[k]["value"])],c.feats[k]["value"][~np.isnan(c.feats[k]["value"])])/tol[k] for k in FE],1)
    return X,(b-a+1)/fps
def dtw(A,B,band=0.5):
    n,m=len(A),len(B); D=np.full((n+1,m+1),np.inf); D[0,0]=0; w=max(int(band*max(n,m)),abs(n-m))
    for i in range(1,n+1):
        for j in range(max(1,i-w),min(m,i+w)+1):
            d=np.linalg.norm(A[i-1]-B[j-1]); D[i,j]=d+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    # đường đi để lấy hệ số nhịp
    i,j,path=n,m,[]
    while i>0 and j>0:
        path.append((i-1,j-1)); k=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]]); i,j=[(i-1,j-1),(i-1,j),(i,j-1)][k]
    return D[n,m]/(n+m),path[::-1]
rows=[]
for s,c in C.items():
    if not c.serve_seg.reps: continue
    rep=c.serve_seg.reps[0]; X,dur=seq(c,rep); out={"video":s,"swing_s":round(dur,2)}
    ds=[];dsh=[]
    for r in ("coach_serve_02","coach_serve_07"):
        if s==r: continue
        Y,dr=seq(C[r],C[r].serve_seg.reps[0]); d,_=dtw(X,Y); ds.append(d)
        d2,_=dtw(X-X.mean(0),Y-Y.mean(0)); dsh.append(d2)
    out["dtw_abs"]=round(min(ds),2); out["dtw_shape"]=round(min(dsh),2)
    rows.append(out)
d=pd.DataFrame(rows)
# điểm hiện tại
sc={}
for s,c in C.items():
    if not c.units: continue
    r=build_reference("serve",{n:C[n].units for n in ["coach_serve_02","coach_serve_07"] if n!=s},cfg) if s in("coach_serve_02","coach_serve_07") else ref
    x=analyze_context(c,cfg,r,s)["score"]; sc[s]=round(x["overall"],1) if x else None
d["diem"]=d.video.map(sc); pd.set_option("display.width",200); print(d.sort_values("dtw_abs").to_string(index=False))
t=d[d.video.str.startswith("trainee")]; c=d[d.video.str.startswith("coach")]
print("\nSpearman(diem, dtw_abs) =",round(d[["diem","dtw_abs"]].corr(method="spearman").iloc[0,1],2)," | shape:",round(d[["diem","dtw_shape"]].corr(method="spearman").iloc[0,1],2))
