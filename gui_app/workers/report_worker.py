import cv2

from gui_app.services import report_service as R
from gui_app.workers.base_worker import BaseWorker


class ReportWorker(BaseWorker):
    """Ảnh khung xương mỗi pha + TXT. (PDF dựng ở luồng chính vì dùng QPrinter/QTextDocument.)
    Trả dict {'txt': Path, 'snaps': {pha: path}}."""

    def __init__(self, ctx, renderer, out_dir, want_txt=True, want_img=True):
        super().__init__()
        self.ctx, self.renderer, self.out = ctx, renderer, out_dir
        self.txt, self.img = want_txt, want_img

    def work(self):
        c, out = self.ctx, {"snaps": {}}
        self.out.mkdir(parents=True, exist_ok=True)
        if self.img:
            cap = cv2.VideoCapture(str(c.input_video))
            (self.out / "snapshots").mkdir(exist_ok=True)
            for ph, win in c.phase_set.reps[0].present():
                mid = (win.start_frame + win.end_frame) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, frame = cap.read()
                if ok:
                    p = self.out / "snapshots" / f"{ph}.png"
                    cv2.imwrite(str(p), self.renderer.render(frame, mid))
                    out["snaps"][ph] = str(p)
            cap.release()
        if self.txt:
            out["txt"] = self.out / "report.txt"
            out["txt"].write_text(R.build_txt(c), encoding="utf-8")
        return out
