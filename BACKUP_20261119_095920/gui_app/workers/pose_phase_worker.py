from pathlib import Path

from gui_app.workers.base_worker import BaseWorker


class PosePhaseWorker(BaseWorker):
    """Trích pose (lưu cache) -> PhaseSegmenter.propose().
    Trả (frames, proposal|None, lỗi_mô_hình|""). Thiếu/lỗi mô hình pha KHÔNG làm hỏng việc trích pose:
    GUI chuyển sang chế độ đánh dấu thủ công."""

    def __init__(self, svc, ctx):
        super().__init__()
        self.svc, self.ctx = svc, ctx

    def work(self):
        from render_overlay import save_pose_cache
        c = self.ctx
        if not c.frames:                                           # đã có cache trong phiên thì dùng lại
            self.progress.emit(-1, "Đang trích xuất khung xương…")  # không có % (generator bị list())
            c.frames = self.svc.extract_poses(str(c.input_video), c.width, c.height)
            c.pose_path = Path(c.session_dir) / "pose.json"
            save_pose_cache(str(c.pose_path), c.frames)
        if self.cancelled:
            return None
        self.progress.emit(-1, "Đang phân pha…")
        try:
            proposal = self.svc.propose_phases(c.frames, c.fps, c.input_video.stem)
            return c.frames, proposal, ""
        except FileNotFoundError as e:                              # chưa có mô hình (D10)
            return c.frames, None, f"no_model: {e}"
        except Exception as e:                                      # mô hình lỗi / lệch đặc trưng
            return c.frames, None, f"model_error: {e}"
