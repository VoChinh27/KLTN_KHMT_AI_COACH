from gui_app.services.event_builder import build_events
from gui_app.workers.base_worker import BaseWorker


class ScoreWorker(BaseWorker):
    def __init__(self, svc, ctx, include_warn=False):
        super().__init__()
        self.svc, self.ctx, self.include_warn = svc, ctx, include_warn

    def work(self):
        c = self.ctx
        self.progress.emit(-1, "Đang chấm điểm…")
        res = self.svc.score(c.frames, c.fps, c.input_video.stem, str(c.phases_path))
        windows = {n: (w.start_frame, w.end_frame) for n, w in c.phase_set.reps[0].present()}
        events, quality = build_events(res, windows, self.include_warn)
        return res, events, quality
