"""Local-only web launcher. No network upload beyond this loopback service."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

from .job_controller import JobController, local_path

VERSION = "squat-local-web-v1"
MAX_UPLOAD = 250 * 1024 * 1024  # storage/transport limit; not a quality threshold
ASSETS = Path(__file__).with_name("web_assets")
ID = re.compile(r"[a-f0-9]{32}\Z")
SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


class RequestError(ValueError):
    def __init__(self, status, reason, message):
        self.status, self.reason = status, reason
        super().__init__(message)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def inspect_upload(path):
    from .video import OpenCVVideoReader
    reader = OpenCVVideoReader(path)
    frames = reader.frames()
    try:
        next(frames)  # Decoder validation; full timeline checked by pipeline later.
    finally:
        frames.close()
    return reader.metadata.model_dump(mode="json")


class Application:
    """One active analysis, idempotent submissions, no browser-supplied paths."""
    def __init__(self, root, *, controller_factory=JobController, inspector=inspect_upload, max_upload=MAX_UPLOAD):
        self.root = local_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Each service lifetime gets a new directory; restart never replays a job.
        self.session = self.root / ("session-" + uuid.uuid4().hex)
        self.session.mkdir()
        (self.session / "uploads").mkdir()
        self.controllers = {}
        self.uploads = {}
        self.requests = {}
        self.lock = threading.RLock()
        self.upload_lock = threading.Lock()
        self.controller_factory, self.inspector = controller_factory, inspector
        self.max_upload = max_upload
        self.token = secrets.token_urlsafe(32)
        self.origin = None
        self.stopping = False

    def upload(self, stream, length, filename):
        if not 0 < length <= self.max_upload:
            raise RequestError(413, "upload_size", "Video phải từ 1 byte đến 250 MiB.")
        # Display name only. It is never used to construct paths or commands.
        if not isinstance(filename, str) or len(filename) > 240 or any(ord(c) < 32 for c in filename):
            raise RequestError(400, "filename", "Tên video không hợp lệ.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUFFIXES:
            raise RequestError(400, "file_type", "Chọn AVI, MP4, MOV, MKV hoặc WebM.")
        if not self.upload_lock.acquire(blocking=False):
            raise RequestError(409, "upload_busy", "Đang tiếp nhận một video khác.")
        uid = uuid.uuid4().hex
        directory = self.session / "uploads" / uid
        directory.mkdir()
        path = directory / ("input" + suffix)
        record = dict(upload_id=uid, original_name=filename, status="receiving", version=VERSION)
        try:
            with path.open("xb") as output:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload chưa đủ bytes đã khai báo")
                    output.write(chunk)
                    remaining -= len(chunk)
            record.update(identity(path))
            record.update(metadata=self.inspector(path), status="ready", decoded_validation="metadata_and_first_frame_only")
            save(directory / "upload.json", record)
            with self.lock:
                self.uploads[uid] = (path, record)
            return record
        except Exception as exc:
            record.update(status="failed", reason="decoder_or_transfer_error", message=str(exc), retained_path=str(directory))
            save(directory / "upload-failed.json", record)
            raise RequestError(422, "decoder_or_transfer_error", "Không đọc được video; không tạo job. " + str(exc)) from exc
        finally:
            self.upload_lock.release()

    def start(self, payload):
        required = {"upload_id", "request_id", "side", "mode", "acknowledged", "use_cache"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise RequestError(400, "job_fields", "Job chỉ nhận upload ID, bên, chế độ và các lựa chọn đã khai báo.")
        if (not isinstance(payload["upload_id"], str) or not ID.fullmatch(payload["upload_id"]) or
                not isinstance(payload["request_id"], str) or not ID.fullmatch(payload["request_id"]) or
                payload["side"] not in ("left", "right") or payload["mode"] not in ("strict", "exploratory") or
                type(payload["acknowledged"]) is not bool or type(payload["use_cache"]) is not bool):
            raise RequestError(400, "job_values", "Lựa chọn job không hợp lệ.")
        if payload["mode"] == "exploratory" and not payload["acknowledged"]:
            raise RequestError(400, "assumption_not_acknowledged", "Cần xác nhận giả định side-view chưa kiểm chứng.")
        with self.lock:
            if self.stopping:
                raise RequestError(409, "server_stopping", "Server đang dừng; không nhận job mới.")
            rid = payload["request_id"]
            if rid in self.requests:
                previous, jid = self.requests[rid]
                if previous != payload:
                    raise RequestError(409, "request_reused", "Request ID đã gắn với lựa chọn khác.")
                return self.snapshot(jid)
            for controller in self.controllers.values():
                controller.poll()
                if controller.active:
                    raise RequestError(409, "job_active", "Đang có một job phân tích. Có thể quay lại theo dõi sau khi refresh.")
            entry = self.uploads.get(payload["upload_id"])
            if entry is None:
                raise RequestError(404, "unknown_upload", "Upload không thuộc phiên server này.")
            path, record = entry
            if identity(local_path(path)) != {k: record[k] for k in ("sha256", "size_bytes")}:
                raise RequestError(409, "upload_changed", "Bytes video đã đổi sau khi tiếp nhận.")
            controller = self.controller_factory()
            job = controller.start(path, self.session / "jobs", side=payload["side"],
                exploratory=payload["mode"] == "exploratory", acknowledged=payload["acknowledged"],
                use_cache=payload["use_cache"])
            self.controllers[job.run_id] = controller
            self.requests[rid] = (dict(payload), job.run_id)
            save(job.directory / "web-request.json", dict(version=VERSION, request=payload, upload=record,
                server_origin=self.origin, source_path=str(path), run_id=job.run_id))
            return self.snapshot(job.run_id)

    def snapshot(self, jid):
        with self.lock:
            controller = self.controllers.get(jid)
            if controller is None:
                raise RequestError(404, "unknown_job", "Job không thuộc phiên server này.")
            controller.poll()
            event = controller.last_event
            data = dict(job_id=jid, status=controller.status, active=controller.active,
                side=controller.job.side, mode="exploratory" if controller.job.exploratory else "strict",
                elapsed_seconds=max(0., time.monotonic() - controller.started),
                event=event.model_dump(mode="json") if event else None,
                log=list(controller.log), error=controller.error, evidence_dir=str(controller.job.directory),
                coverage=None, artifacts=[], cache_status=getattr(controller,"cache_status",None))
            if controller.verified is not None and controller.status == "completed" and not controller.active:
                verified = controller.verified
                data.update(coverage=dict(verified.coverage), cache_status=verified.cache_status,
                    inference_run=verified.inference_run)
                manifest = self.checked_manifest(controller)
                data["artifacts"] = sorted(set(manifest["files"]) | {"manifest.json"})
                data["video_file"] = next(n for k, n, _ in verified.targets if k == "video")
            return data

    def state(self):
        with self.lock:
            return dict(version=VERSION, csrf_token=self.token, max_upload_bytes=self.max_upload,
                uploads=[r for _, r in self.uploads.values()], jobs=[self.snapshot(j) for j in self.controllers],
                session_dir=str(self.session), cancellation_supported=False,
                validation_status="NOT_VALIDATED", official_eligibility=False, assessment=False,
                reference_status="REFERENCE_PENDING", g57="BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT", g6="IN_PROGRESS")

    def checked_manifest(self, controller):
        verified = controller.verified
        if controller.active or controller.status != "completed" or verified is None:
            raise RequestError(409, "not_verified", "Artifacts chưa được xác minh hoàn tất.")
        directory = local_path(verified.directory)
        if not directory.is_relative_to(self.session):
            raise RequestError(403, "outside_application", "Artifacts nằm ngoài vùng ứng dụng.")
        manifest_path = local_path(directory / "manifest.json")
        if identity(manifest_path)["sha256"] != verified.manifest_sha256:
            raise RequestError(409, "artifact_changed", "Manifest đã thay đổi sau xác minh.")
        return json.loads(manifest_path.read_bytes())

    def artifact(self, jid, name):
        if not ID.fullmatch(jid) or Path(name).name != name or re.search(r"[\\/:\x00]", name) or name in (".", ".."):
            raise RequestError(403, "artifact_path", "Đường dẫn artifact không được phép.")
        with self.lock:
            controller = self.controllers.get(jid)
            if controller is None:
                raise RequestError(404, "unknown_job", "Job không tồn tại.")
            manifest = self.checked_manifest(controller)
            if name != "manifest.json" and name not in manifest["files"]:
                raise RequestError(404, "not_allowlisted", "File không nằm trong manifest đã xác minh.")
            path = local_path(controller.verified.directory / name)
            if path.parent != controller.verified.directory:
                raise RequestError(403, "artifact_path", "Artifact ra ngoài thư mục job.")
            if name != "manifest.json" and identity(path) != manifest["files"][name]:
                raise RequestError(409, "artifact_changed", "Artifact đã thay đổi sau xác minh.")
            return path


def range_bounds(header, size):
    """Single HTTP byte range for browser MP4 seeking, inclusive endpoints."""
    if not header:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or not size:
        raise RequestError(416, "range", "Byte range không hợp lệ.")
    a, b = match.groups()
    if not a:
        if int(b) <= 0:
            raise RequestError(416, "range", "Suffix range phải dương.")
        start, end = max(0, size - int(b)), size - 1
    else:
        start, end = int(a), min(int(b), size - 1) if b else size - 1
    if start >= size or end < start:
        raise RequestError(416, "range", "Byte range ngoài file.")
    return start, end, True


class Handler(BaseHTTPRequestHandler):
    server_version = "GymLocal/1"
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def log_message(self, fmt, *args):
        # Do not log tokens, original filenames, raw requests or commands.
        pass

    def guard(self, mutation=False):
        app = self.server.app
        if self.client_address[0] != "127.0.0.1" or self.headers.get_all("Host") != [app.origin.removeprefix("http://")]:
            raise RequestError(403, "host", "Host local không hợp lệ.")
        origin = self.headers.get("Origin")
        if (origin is not None and origin != app.origin) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise RequestError(403, "origin", "Origin không được phép.")
        if mutation and (origin != app.origin or not secrets.compare_digest(self.headers.get("X-Gym-CSRF", ""), app.token)):
            raise RequestError(403, "csrf", "Thiếu xác nhận request cùng ứng dụng local.")

    def send_common(self, status, content_type, length, *, extra=None):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; media-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def json_response(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_common(status, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def serve_file(self, path, *, attachment=False):
        size = path.stat().st_size
        try:
            start, end, partial = range_bounds(self.headers.get("Range"), size)
        except RequestError:
            self.send_common(416, "text/plain", 0, extra={"Content-Range": f"bytes */{size}"})
            return
        extra = {"Accept-Ranges": "bytes"}
        if partial:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        if attachment:
            extra["Content-Disposition"] = 'attachment; filename="' + path.name + '"'
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".html", ".js", ".css", ".txt"):
            content_type += "; charset=utf-8"
        self.send_common(206 if partial else 200, content_type, max(0, end-start+1), extra=extra)
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end-start+1
            while remaining > 0:
                chunk = stream.read(min(256*1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def dispatch(self, mutation=False):
        try:
            self.guard(mutation)
            url = urlsplit(self.path)
            path = unquote(url.path)
            app = self.server.app
            if mutation:
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                    raise RequestError(411, "length", "Yêu cầu Content-Length duy nhất.")
                try:
                    length = int(self.headers["Content-Length"])
                except ValueError as exc:
                    raise RequestError(400, "length", "Content-Length không hợp lệ.") from exc
                if path == "/api/uploads":
                    if self.headers.get("Content-Type") != "application/octet-stream":
                        raise RequestError(415, "content_type", "Upload dùng application/octet-stream.")
                    result = app.upload(self.rfile, length, unquote(self.headers.get("X-File-Name", "")))
                    self.json_response(201, result)
                elif path == "/api/jobs":
                    if not 0 < length <= 4096 or self.headers.get("Content-Type") != "application/json":
                        raise RequestError(400, "job_body", "Job JSON không hợp lệ hoặc quá dài.")
                    result = app.start(json.loads(self.rfile.read(length)))
                    self.json_response(202, result)
                else:
                    raise RequestError(404, "route", "Endpoint không tồn tại.")
            elif path == "/api/state":
                self.json_response(200, app.state())
            elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
                self.json_response(200, app.snapshot(path.rsplit("/", 1)[1]))
            elif path.startswith("/artifacts/"):
                parts = path.split("/")
                if len(parts) != 4:
                    raise RequestError(403, "artifact_path", "Đường dẫn artifact không hợp lệ.")
                self.serve_file(app.artifact(parts[2], parts[3]), attachment=url.query == "download=1")
            elif path in ("/", "/app.js", "/app.css"):
                self.serve_file(ASSETS / ("index.html" if path == "/" else path[1:]))
            else:
                raise RequestError(404, "route", "Endpoint không tồn tại.")
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            self.close_connection = True  # never parse unconsumed request body as a new request
            error = exc if isinstance(exc, RequestError) else RequestError(400, "request_error", str(exc))
            self.json_response(error.status, dict(error=error.reason, message=str(error)))

    def do_GET(self):
        self.dispatch()

    def do_HEAD(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch(True)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, app, port=8765):
        # No bind-host parameter: deployment outside loopback is unsupported.
        super().__init__(("127.0.0.1", port), Handler)
        self.app = app
        app.origin = f"http://127.0.0.1:{self.server_port}"


def main(argv=None):
    # Windows redirected terminals can default to cp1252; Vietnamese output is UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/squat_web"))
    args = parser.parse_args(argv)
    app = Application(args.output_root)
    server = LocalServer(app, args.port)
    print("Gym local WEB: " + app.origin, flush=True)
    print("Ctrl+C: ngừng nhận request và đợi job hiện hành hoàn tất; không hỗ trợ hủy.", flush=True)
    print("Evidence: " + str(app.session), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Đang dừng an toàn. Vui lòng đợi job; không đóng cưỡng bức terminal.", flush=True)
    finally:
        app.stopping = True
        server.server_close()
        for controller in app.controllers.values():
            if controller.thread is not None:
                controller.thread.join()
            controller.poll()


if __name__ == "__main__":
    main()
