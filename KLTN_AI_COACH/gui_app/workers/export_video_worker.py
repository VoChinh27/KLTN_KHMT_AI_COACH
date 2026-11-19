import cv2

from gui_app import constants as C
from gui_app.workers.base_worker import BaseWorker


class ExportVideoWorker(BaseWorker):
    """Ghi overlay.mp4 (mp4v). Lặp khung ~2s tại điểm lỗi để tái hiện hiệu ứng dừng."""

    def __init__(self, ctx, renderer, out_path):
        super().__init__()
        self.ctx, self.renderer, self.out = ctx, renderer, str(out_path)

    def work(self):
        c = self.ctx
        cap = cv2.VideoCapture(str(c.input_video))
        w = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*"mp4v"), c.fps, (c.width, c.height))
        if not w.isOpened():
            raise IOError(f"Không ghi được: {self.out}")
        pause_at = {p.frame for p in c.pauses}
        repeat = int(C.PAUSE_REPEAT_SECONDS * c.fps)
        i = 0
        try:
            while not self.cancelled:
                ok, frame = cap.read()
                if not ok:
                    break
                out = self.renderer.render(frame, i)
                w.write(out)
                if i in pause_at:
                    for _ in range(repeat):
                        w.write(out)
                i += 1
                if i % 15 == 0:
                    self.progress.emit(int(100 * i / max(c.total_frames, 1)), "Đang ghi video…")
        finally:
            cap.release(); w.release()
        return self.out
