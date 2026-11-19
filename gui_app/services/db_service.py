"""SQLite: students / sessions / session_phases / session_findings."""
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS students(
  id INTEGER PRIMARY KEY, student_code TEXT UNIQUE NOT NULL,
  full_name TEXT NOT NULL, created_at TEXT);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, session_uid TEXT UNIQUE,
  student_id INTEGER REFERENCES students(id),
  sport TEXT, recorded_at TEXT, scored_at TEXT,
  input_file_name TEXT, input_dir TEXT, input_video_path TEXT,
  output_video_path TEXT, config_yaml_path TEXT, phases_json_path TEXT,
  pose_cache_path TEXT, report_txt_path TEXT, report_pdf_path TEXT,
  overall_score REAL, score_confidence TEXT, phase_source TEXT);
CREATE TABLE IF NOT EXISTS session_phases(
  id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES sessions(id),
  phase TEXT, start_frame INTEGER, end_frame INTEGER,
  score REAL, is_scored INTEGER, skip_reason TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS session_findings(
  id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES sessions(id),
  phase TEXT, feature_key TEXT, display_name TEXT, status TEXT,
  student_value REAL, reference_center REAL, score REAL,
  error_type TEXT, message TEXT, pause_frame INTEGER, snapshot_path TEXT);
"""
PATH_COLS = {"output_video_path", "report_txt_path", "report_pdf_path"}


def _nan_to_none(v):
    return None if v is None or v != v else float(v)


class DbService:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    # ---- học viên ----
    def find_student(self, code: str):
        with self._conn() as c:
            return c.execute("SELECT * FROM students WHERE student_code=?", (code,)).fetchone()

    def list_students(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT student_code, full_name FROM students ORDER BY student_code")]

    # ---- phiên ----
    def save_session(self, ctx) -> int:
        r = ctx.result
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            s = c.execute("SELECT id FROM students WHERE student_code=?", (ctx.student_code,)).fetchone()
            sid = s["id"] if s else c.execute(
                "INSERT INTO students(student_code, full_name, created_at) VALUES(?,?,?)",
                (ctx.student_code, ctx.student_name, now)).lastrowid
            sess = c.execute(
                "INSERT INTO sessions(session_uid, student_id, sport, recorded_at, scored_at, input_file_name,"
                " input_dir, input_video_path, config_yaml_path, phases_json_path, pose_cache_path,"
                " overall_score, score_confidence, phase_source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ctx.session_dir.name, sid, ctx.sport, ctx.recorded_at.isoformat(timespec="seconds"), now,
                 ctx.source_video.name if ctx.source_video else "", str(ctx.session_dir),
                 str(ctx.input_video), str(ctx.yaml_path), str(ctx.phases_path), str(ctx.pose_path),
                 _nan_to_none(r.overall_score), r.score_confidence, r.phase_source)).lastrowid
            pause_frame = {e.phase: e.frame for e in ctx.pauses}
            rep = ctx.phase_set.reps[0] if ctx.phase_set else None
            for ph in r.sub_phase_order:
                sp = r.sub_phase_results.get(ph)
                if sp is None:
                    continue
                fb = sp.result.feedback if sp.has_score() else None
                win = rep.phases.get(ph) if rep else None
                c.execute("INSERT INTO session_phases(session_id, phase, start_frame, end_frame, score,"
                          " is_scored, skip_reason, summary) VALUES(?,?,?,?,?,?,?,?)",
                          (sess, ph, getattr(win, "start_frame", None), getattr(win, "end_frame", None),
                           _nan_to_none(fb.overall_score) if fb else None, int(sp.has_score()),
                           sp.skip_reason, fb.summary_line if fb else None))
                for it in (fb.items if fb else []):
                    et = getattr(it.error_type, "value", it.error_type)
                    c.execute("INSERT INTO session_findings(session_id, phase, feature_key, display_name,"
                              " status, student_value, reference_center, score, error_type, message,"
                              " pause_frame) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (sess, ph, it.feature_key, it.display_name, it.status_symbol,
                               _nan_to_none(it.student_value), _nan_to_none(it.reference_center),
                               _nan_to_none(it.score_0_100), et, it.message, pause_frame.get(ph)))
            return sess

    def update_paths(self, session_id: int, **paths) -> None:
        paths = {k: str(v) for k, v in paths.items() if k in PATH_COLS and v}
        if not paths:
            return
        sets = ", ".join(f"{k}=?" for k in paths)
        with self._conn() as c:
            c.execute(f"UPDATE sessions SET {sets} WHERE id=?", (*paths.values(), session_id))

    def list_sessions(self, student: str = "", sport: str = "", date_from: str = "", date_to: str = ""):
        q = ("SELECT s.id, s.scored_at, st.student_code, st.full_name, s.sport, s.overall_score,"
             " s.score_confidence, s.input_video_path, s.output_video_path, s.report_txt_path,"
             " s.report_pdf_path FROM sessions s JOIN students st ON st.id=s.student_id"
             " WHERE (?='' OR st.student_code LIKE '%'||?||'%' OR st.full_name LIKE '%'||?||'%')"
             " AND (?='' OR s.sport=?) AND (?='' OR s.scored_at>=?) AND (?='' OR s.scored_at<=?)"
             " ORDER BY s.scored_at DESC")
        a = (student, student, student, sport, sport, date_from, date_from, date_to, date_to)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, a)]

    def phases_of(self, session_id: int):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM session_phases WHERE session_id=? ORDER BY id", (session_id,))]

    def findings_of(self, session_id: int):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM session_findings WHERE session_id=? ORDER BY id", (session_id,))]

    def delete_session(self, session_id: int) -> None:
        with self._conn() as c:
            for t in ("session_findings", "session_phases"):
                c.execute(f"DELETE FROM {t} WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
