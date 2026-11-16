"""UI Ä‘á»™c láº­p Ä‘á»ƒ xem/Ä‘iá»u chá»‰nh proposal tá»± Ä‘á»™ng, khÃ´ng sá»­a mark_phases.py.

Space: pause/play; â†/â†’: tua; Tab: chá»n biÃªn; ,/. : lÃ¹i/tiáº¿n biÃªn 1 frame;
V rá»“i 1..4: báº­t/táº¯t pha váº¯ng máº·t; A: cháº¥p nháº­n proposal; S: lÆ°u Ä‘Ã£ review;
Esc: thoÃ¡t khÃ´ng lÆ°u.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,time
from pathlib import Path
import cv2
from core.phase_schema import PHASE_ORDER,PhaseSet,Rep,PhaseWindow,load_phases,save_phases

class AutoPhaseReview:
    def __init__(self,video:Path,proposal_path:Path,output:Path,feedback_log:Path):
        self.video=video;self.proposal_path=proposal_path;self.output=output;self.feedback_log=feedback_log;self.ps=load_phases(proposal_path)
        self.rep=self.ps.reps[0];self.original={p:(None if self.rep.phases[p].absent else (self.rep.phases[p].start_frame,self.rep.phases[p].end_frame)) for p in PHASE_ORDER}
        self.conf=list((self.ps.auto_meta or {}).get('boundary_conf') or []);self.nframes=self.ps.total_frames;self.fps=self.ps.fps;self.selected=0;self.frame=0;self.playing=False;self.absent_mode=False;self.started=time.monotonic();self.edits=[];self.accepted_all=False
    def _record(self,phase,before,after):
        if before!=after:self.edits.append({'phase':phase,'from':before,'to':after,'at_frame':self.frame})
    def _window(self,p):
        w=self.rep.phases[p];return None if w.absent else (w.start_frame,w.end_frame)
    def _set(self,p,window):
        old=self._window(p);self.rep.phases[p]=PhaseWindow.missing() if window is None else PhaseWindow(int(window[0]),int(window[1]));self._record(p,old,window)
    def _move_boundary(self,delta):
        idx=self.selected
        present=[(i,p,self._window(p)) for i,p in enumerate(PHASE_ORDER) if self._window(p) is not None]
        if not present:return
        if idx==0:
            _,p,(a,b)=present[0];na=max(0,min(b,a+delta));self._set(p,(na,b));return
        if idx==4:
            _,p,(a,b)=present[-1];nb=max(a,min(self.nframes-1,b+delta));self._set(p,(a,nb));return
        left=next((x for x in reversed(present) if x[0]<idx),None)
        right=next((x for x in present if x[0]>=idx),None)
        if left is None or right is None:return
        _,lp,(la,lb)=left;_,rp,(ra,rb)=right
        boundary=max(la+1,min(rb,ra+delta))
        self._set(lp,(la,boundary-1));self._set(rp,(boundary,rb))
    def _draw(self,frame):
        h,w=frame.shape[:2];cv2.putText(frame,f'{self.video.name}  frame {self.frame}/{self.nframes-1}  {self.fps:.2f} fps',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        bar_y=h-52;cv2.line(frame,(12,bar_y),(w-12,bar_y),(180,180,180),2)
        for i,p in enumerate(PHASE_ORDER):
            win=self._window(p);text=f'{i+1}:{p} '+('ABSENT' if win is None else f'{win[0]}â€“{win[1]}')
            if win:
                color=(0,220,0) if i!=self.selected%4 else (0,255,255)
                for edge in (win[0],win[1]):
                    x=12+int((w-24)*edge/max(1,self.nframes-1));cv2.line(frame,(x,bar_y-10),(x,bar_y+10),color,2)
            c0=self.conf[i] if i<len(self.conf) else None;c1=self.conf[i+1] if i+1<len(self.conf) else None
            f0='n/a' if c0 is None else f'{float(c0):.2f}';f1='n/a' if c1 is None else f'{float(c1):.2f}'
            cv2.putText(frame,f'{text} conf[start={f0}, end={f1}]',(12,bar_y+20+i*18),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,255) if i==self.selected%4 else (255,255,255),1)
        cv2.putText(frame,'Tab:boundary  , .:move  V+1..4:absent  A:accept  S:save  Esc:quit',(12,h-10),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        return frame
    def save(self):
        self.ps.marked_by='auto_reviewed';self.ps.review={'edits':self.edits,'seconds_spent':round(time.monotonic()-self.started,2),'accepted_all':self.accepted_all,'source_file':str(self.proposal_path)}
        self.ps.reps=[self.rep];errs=self.ps.validate()
        if errs:raise ValueError('NhÃ£n review khÃ´ng há»£p lá»‡: '+'; '.join(errs))
        save_phases(self.output,self.ps)
        from stage3_features.phase_model.support.phase_feedback_log import log_review
        auto={'video':self.ps.video,'fps':self.ps.fps,'total_frames':self.ps.total_frames,'marked_by':'auto','auto_meta':self.ps.auto_meta,'phases':self.ps.auto_phases or {}}
        log_review(auto,self.ps,self.feedback_log)
        print(f'Saved: {self.output}')
    def run(self):
        cap=cv2.VideoCapture(str(self.video))
        if not cap.isOpened():raise OSError(f'KhÃ´ng má»Ÿ Ä‘Æ°á»£c video: {self.video}')
        try:
            while True:
                if self.playing:
                    ok,frame=cap.read()
                    if not ok:self.playing=False;continue
                    self.frame=min(self.nframes-1,int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame);ok,frame=cap.read()
                    if not ok:raise OSError(f'KhÃ´ng Ä‘á»c Ä‘Æ°á»£c frame {self.frame}')
                cv2.imshow('Auto phase review',self._draw(frame));key=cv2.waitKey(max(1,int(1000/self.fps)) if self.playing else 30)&0xff
                if key==27:break
                if key==ord(' '):self.playing=not self.playing
                elif key in (81,ord('a')):self.frame=max(0,self.frame-max(1,int(self.fps)))
                elif key in (83,ord('d')):self.frame=min(self.nframes-1,self.frame+max(1,int(self.fps)))
                elif key==9:self.selected=(self.selected+1)%5
                elif key==ord(','):self._move_boundary(-1)
                elif key==ord('.'):self._move_boundary(1)
                elif key in (ord('v'),ord('V')):self.absent_mode=True
                elif self.absent_mode and key in map(ord,'1234'):
                    i=key-ord('1');p=PHASE_ORDER[i];current=self._window(p)
                    if current is not None:self._set(p,None)
                    else:
                        span=max(1,int(round(0.2*self.fps)));start=max(0,min(self.frame,self.nframes-1));end=min(self.nframes-1,start+span-1);self._set(p,(start,end))
                    self.absent_mode=False
                elif key==ord('A'):
                    for p,window in self.original.items():self._set(p,window)
                    self.edits=[];self.accepted_all=True
                elif key in (ord('S'),ord('s')):self.save();break
                elif key==ord(' '):continue
                elif 0<=key<4:self.selected=key
        finally:cap.release();cv2.destroyAllWindows()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--video',type=Path,required=True);ap.add_argument('--proposal',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--feedback-log',type=Path,default=Path('data/phase_feedback/log.jsonl'));args=ap.parse_args()
    out=args.output or Path('data/phases')/(args.proposal.stem.replace('.phases','')+'.phases.json')
    AutoPhaseReview(args.video,args.proposal,out,args.feedback_log).run()
if __name__=='__main__':main()

