"""Read-only real-video presentation. Numeric values come only from saved JSON."""
from functools import lru_cache
import html
import json
import os
from pathlib import Path

from .real_descriptive_demo import sha, validate_result, write_json, scoped, DATA_SCOPE
from ..pose.models import PoseFrame
from ..preprocessing.models import content_hash as pose_hash
from ..video import OpenCVVideoReader

EDGES = (("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"))
OBSERVED = (99, 226, 210)
LOW = (255, 190, 77)
SIZE = (1280, 720)
ROI = (24, 128, 640, 480)


@lru_cache(maxsize=12)
def font(size):
    from PIL import ImageFont
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (fonts / "segoeui.ttf", fonts / "arial.ttf", Path("DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    raise RuntimeError("A Unicode TrueType font is required; no font is downloaded")


def overlay_text(row, side, sequence, *, localized_unavailable=False):
    obs = row["observation"]
    reason = ", ".join(obs["reasons"]) if obs["reasons"] else ""
    if localized_unavailable and obs["status"] != "available":
        metric_reasons = obs["availability"]["input"]["quality"]["metric_eligibility"][obs["feature_id"]]["reasons"]
        translations = {
            "partial_joint_coverage": "Gate toàn clip: thiếu độ phủ khớp.",
            "low_pose_visibility": "Gate toàn clip: độ hiển thị thấp.",
            "excessive_missingness": "Gate toàn clip: thiếu dữ liệu khớp.",
            "viewpoint_unknown": "Góc quay chưa xác định theo policy.",
            "viewpoint_mismatch": "Góc quay ngoài policy của số đo.",
            "insufficient_sequence_length": "Clip quá ngắn theo quality policy.",
            "out_of_frame": "Khớp nằm ngoài khung hình.",
            "degenerate_geometry": "Hình học suy biến; không có số đo.",
            "feature_joint_unusable": "Khớp của frame không đủ điều kiện đo.",
            "feature_provenance_rejected": "Provenance joint không đạt policy.",
        }
        reason = next((translations[c] for c in metric_reasons + obs["reasons"] if c in translations),
                      "Dữ liệu không đạt policy đo góc.")
    return dict(side="Gối trái" if side == "left" else "Gối phải",
        angle=f"{obs['value']:.1f}°" if obs["status"] == "available" else
              ("Không đủ điều kiện đo" if localized_unavailable else "Unavailable"),
        status=obs["status"], reason=reason,
        frame=f"Frame {row['frame_index']} · {row['timestamp_seconds']:.2f} s",
        sequence=sequence["action"].upper(), metric=row["quality_action"].upper())


def drawable(joint):
    return (joint["status"] in ("observed", "low_visibility")
            and joint["x"] is not None and joint["y"] is not None
            and 0 <= joint["x"] <= 1 and 0 <= joint["y"] <= 1)


def paint_frame(bgr, row, side, sequence, *, localized_unavailable=False,
                presentation_mode=None, preserve_aspect=False):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    scene = Image.new("RGB", SIZE, (12, 21, 34))
    original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # 320x240 ->640x480, isotropic resize only. No source crop/rotation/retiming.
    image_roi = ROI
    if preserve_aspect:
        scale = min(ROI[2] / original.width, ROI[3] / original.height)
        width, height = max(1, round(original.width * scale)), max(1, round(original.height * scale))
        image_roi = (ROI[0] + (ROI[2]-width)//2, ROI[1] + (ROI[3]-height)//2, width, height)
    original = original.resize(image_roi[2:], Image.Resampling.LANCZOS)
    scene.paste(original, image_roi[:2])
    draw = ImageDraw.Draw(scene)
    text = overlay_text(row, side, sequence, localized_unavailable=localized_unavailable)
    muted = (180, 196, 213); white = (238, 246, 252)
    def label(x, y, value, size=20, color=white):
        draw.text((x, y), value, font=font(size), fill=color)
    label(24, 16, "BodyWeightSquats · mô tả một bên", 29)
    label(24, 58, "Strict — giữ viewpoint nguồn chưa xác định" if presentation_mode == "strict" else
          "Exploratory — góc quay chưa xác nhận", 24, LOW)
    draw.rounded_rectangle((688, 128, 1256, 608), radius=16, fill=(20, 35, 53))
    label(712, 145, text["side"] + " · cố định toàn clip", 24)
    label(712, 190, text["angle"], 62 if text["status"] == "available" else 38)
    label(712, 275, "Đơn vị: degrees · " + text["status"], 19, muted)
    if text["reason"]:
        # Reason codes are data diagnostics, never technical errors.
        for i, line in enumerate(text["reason"].split(", ")[:3]):
            label(712, 308 + i * 24, line[:49], 16, LOW)
    label(712, 390, "Metric: " + text["metric"] + " · " + (presentation_mode or "exploratory"), 21, OBSERVED)
    label(712, 424, "Sequence: " + text["sequence"] + " · giữ nguyên", 21, LOW)
    label(712, 463, "Phase / rep / point-turn / pause:", 20, muted)
    label(712, 495, "Unavailable — chưa có evidence riêng", 19, muted)
    label(712, 552, "official_eligibility=false · assessment=false", 17, muted)
    joints = {j["joint"]: j for j in row["raw_joints"]}
    def xy(j):
        return (image_roi[0] + round(j["x"] * (image_roi[2] - 1)), image_roi[1] + round(j["y"] * (image_roi[3] - 1)))
    for a, b in EDGES:
        A, B = joints[a], joints[b]
        if drawable(A) and drawable(B):
            color = OBSERVED if A["status"] == B["status"] == "observed" else LOW
            draw.line((xy(A), xy(B)), fill=color, width=3)
    for joint in joints.values():
        if not drawable(joint):
            continue
        x, y = xy(joint)
        color = OBSERVED if joint["status"] == "observed" else LOW
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    label(24, 100, "● observed", 17, OBSERVED)
    label(178, 100, "● low visibility", 17, LOW)
    label(370, 100, "missing / ngoài ảnh: không vẽ", 17, muted)
    label(24, 622, text["frame"] + " · " + side + " knee", 22)
    label(24, 657, "Góc projected2D: processed · Skeleton: raw · REFERENCE_PENDING" if presentation_mode else
          "Góc: processed coordinates · Skeleton: raw pose · camera_view=UNKNOWN", 19, muted)
    label(24, 689, "NOT_VALIDATED · G5.7 blocked · G6 IN_PROGRESS · Không chấm kỹ thuật", 16, LOW)
    return cv2.cvtColor(np.asarray(scene), cv2.COLOR_RGB2BGR)


def load_saved(result_path, raw_path):
    result = validate_result(json.loads(Path(result_path).read_text(encoding="utf-8")))
    expected = result["selection"]["reused_artifacts"]["raw-pose-unknown.jsonl"]["sha256"]
    if sha(raw_path) != expected:
        raise ValueError("Raw skeleton source byte hash mismatch")
    raw = [PoseFrame.model_validate_json(line, strict=True) for line in Path(raw_path).read_bytes().splitlines()]
    if len(raw) != len(result["frames"]):
        raise ValueError("Raw skeleton frame count mismatch")
    for pose, row in zip(raw, result["frames"]):
        if (pose.viewpoint.value != "unknown" or pose.frame_index != row["frame_index"]
                or pose.video_id != row["video_id"] or pose.timestamp_seconds != row["timestamp_seconds"]
                or pose_hash(pose) != row["raw_pose_hash"]
                or [j.model_dump(mode="json") for j in pose.joints] != row["raw_joints"]):
            raise ValueError("Skeleton/frame/source linkage mismatch")
    return result


def verify_video(path, count, fps, *, progress=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Output video cannot be decoded")
    measured_fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != SIZE:
                raise ValueError("Output video resolution mismatch")
            decoded += 1
            if progress is not None:progress.count("video_validation",decoded,count,"decoded_frames")
    finally:
        cap.release()
    if decoded != count or abs(measured_fps - fps) > 1e-9:
        raise ValueError("Output frame count/FPS mismatch")
    return dict(status="PASSED", decoded_frames=decoded, fps=measured_fps, width=SIZE[0], height=SIZE[1],
                browser_qa="NOT_RUN", visual_qa="NOT_RUN")


def opening_html(result, video_name):
    esc = html.escape
    c = result["coverage"]; s = result["selection"]
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Squat · Real descriptive demo</title><style>body{{margin:24px auto;max-width:1120px;padding:0 18px;background:#0c1522;color:#eef6fc;font:17px/1.6 system-ui}}video{{width:100%;background:#000}}a{{color:#72e2d2}}a:focus-visible{{outline:3px solid #ffbe4d}}.notice{{color:#ffbe4d}}table{{border-collapse:collapse}}td,th{{border:1px solid #536378;padding:8px 20px}}code{{overflow-wrap:anywhere}}</style>
<h1>BodyWeightSquats · mô tả một bên</h1><p class="notice">Exploratory — góc quay chưa xác nhận · NOT_VALIDATED</p>
<video controls preload="metadata"><source src="{esc(video_name, quote=True)}" type="video/mp4">Mở video bằng trình phát desktop có sẵn.</video>
<p><a href="{esc(video_name, quote=True)}">Mở / lưu video overlay</a> · <a href="original.avi">Video gốc (AVI)</a> · <a href="HUONG_DAN_MO_DEMO.txt">Hướng dẫn</a></p>
<p>Giữ nguyên thư mục. MP4 được kiểm tra giải mã bằng OpenCV; browser QA NOT_RUN.
Nếu trình duyệt không hỗ trợ codec hoặc chặn file local, mở video bằng ứng dụng desktop có sẵn; không tắt bảo vệ hoặc dựng server.</p>
<p>Clip: <code>{esc(s['video_id'])}</code> · bên <b>{esc(s['side'])}</b>, cố định toàn clip.
Sequence: <b>{esc(s['sequence']['action'])}</b>; metric: <b>{esc(s['metric']['action'])}</b> (không nâng sequence).</p>
<table><tr><th>Tổng frame</th><th>Có góc</th><th>Unavailable</th><th>Coverage dữ liệu</th></tr>
<tr><td>{c['total_frames']}</td><td>{c['available_frames']}</td><td>{c['unavailable_frames']}</td><td>{c['available_ratio']:.0%}</td></tr></table>
<p>Coverage là mức sẵn có, không phải độ chính xác. Góc từ processed coordinates;
skeleton từ raw pose cùng video/frame. Observed màu xanh ngọc, low visibility màu cam;
missing/ngoài ảnh không vẽ. Màu không biểu thị kỹ thuật đúng/sai.</p>
<p>Source camera_view=UNKNOWN. Assumed-side: visually_assessed_unverified, documented=false.
Phase, repetition, point-turn, pause và technical feedback: Unavailable. Không form score.</p>
<p>official_eligibility=false · assessment=false · G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT · G6 IN_PROGRESS</p>
<p><a href="result.json">Kết quả từng frame</a> · <a href="coverage.json">Coverage</a> · <a href="selection.json">Selection receipt</a> · <a href="manifest.json">Hashes</a> · <a href="video-verification.json">Decode receipt</a></p></html>'''


def render_video_bundle(root, output, *, localized_unavailable=False):
    """Render saved values and raw joints only; no preprocessing or geometry calls."""
    import cv2
    result = load_saved(output / "result.json", output / "raw-pose-unknown.jsonl")
    source = scoped(root, result["selection"]["path"], DATA_SCOPE)
    if sha(source) != result["selection"]["sha256"]:
        raise ValueError("Source video byte hash mismatch")
    reader = OpenCVVideoReader(source)
    if reader.metadata.model_dump(mode="json") != result["metadata"]:
        raise ValueError("Decoder metadata differs from saved frame timeline")
    target, codec, attempts, indices = encode_saved_frames(reader, output, result["frames"],
        result["selection"]["side"], result["selection"]["sequence"], localized_unavailable=localized_unavailable)
    receipt = verify_video(target, len(result["frames"]), reader.metadata.fps)
    receipt.update(video_file=target.name, codec=codec, codec_attempts=attempts, frame_map=indices,
                   source_sha256=sha(source), overlay_source="saved result.json + exact raw pose snapshot",
                   geometry_calculated_by_renderer=False, source_resize="Isotropic320x240->640x480; no crop/flip; whole clip")
    write_json(output / "video-verification.json", receipt)
    with (output / "START_HERE.html").open("x", encoding="utf-8") as stream:
        stream.write(opening_html(result, target.name))
    guide = ("BODYWEIGHTSQUATS — EXPLORATORY / NOT_VALIDATED\n"
        "Mở START_HERE.html hoặc video " + target.name + " bằng trình phát desktop có sẵn.\n"
        "Giữ nguyên thư mục với JSON sidecars; không cần Python/mạng/server để xem.\n"
        "Nếu codec/file local bị chặn, không tắt bảo vệ hoặc đi vòng security policy.\n"
        "Bên đo cố định; góc lấy từ processed coordinates, skeleton từ raw pose.\n"
        "Observed xanh ngọc; low visibility cam; missing/ngoài ảnh không vẽ.\n"
        "Màu và số đo không nói kỹ thuật đúng/sai. Coverage không phải accuracy.\n"
        "Source camera_view=UNKNOWN, assumed-side visually_assessed_unverified.\n"
        "Unavailable/null là thiếu evidence hoặc gate không đạt, không phải giá trị0.\n"
        "Không rep/phase/point-turn/pause/score/feedback kỹ thuật.\n"
        "official_eligibility=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS.\n"
        "browser_qa=NOT_RUN; xem visual QA frames riêng trong báo cáo milestone.\n")
    with (output / "HUONG_DAN_MO_DEMO.txt").open("x", encoding="utf-8-sig") as stream:
        stream.write(guide)


def encode_saved_frames(reader, output, rows, side, sequence, *, progress=None, **paint_options):
    """Shared encoder for legacy demo and single-video orchestration; saved-only."""
    import cv2
    attempts = []; writer = None
    for codec, name in (("avc1", "overlay-h264.mp4"), ("mp4v", "overlay-mpeg4.mp4")):
        target = output / name
        if target.exists():
            raise FileExistsError(target)
        candidate = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), reader.metadata.fps, SIZE)
        opened = candidate.isOpened()
        attempts.append(dict(codec=codec, path=name, opened=opened))
        if opened:
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError("No installed MP4 encoder available; see codec attempts")
    indices = []
    try:
        for frame in reader.frames():
            row = rows[frame.frame_index]
            if row["timestamp_seconds"] != frame.timestamp_seconds:
                raise ValueError("Video/frame timestamp mismatch")
            writer.write(paint_frame(frame.image, row, side, sequence, **paint_options))
            indices.append(dict(video_id=row["video_id"], output_frame=frame.frame_index,
                                source_frame=frame.frame_index, timestamp_seconds=frame.timestamp_seconds))
            if progress is not None:progress.count("render",len(indices),len(rows),"output_frames")
    finally:
        writer.release()
    return target, codec, attempts, indices


def single_video_html(result, video_name):
    esc = html.escape
    c = result["coverage"]
    mode = "Strict — không giả định viewpoint" if result["mode"] == "strict" else "Exploratory — side do caller giả định, chưa xác nhận"
    reasons = sorted({reason for row in result["frames"] for reason in row["observation"]["reasons"]})
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Gym · BodyWeightSquats</title><style>body{{max-width:1120px;margin:24px auto;padding:0 20px;background:#0c1522;color:#eef6fc;font:17px/1.6 system-ui}}video{{width:100%;background:black}}a{{color:#72e2d2}}a:focus-visible{{outline:3px solid #ffbe4d}}.notice{{color:#ffbe4d}}code{{overflow-wrap:anywhere}}</style>
<h1>BodyWeightSquats · gối {'trái' if result['side']=='left' else 'phải'} cố định</h1>
<p class="notice">{esc(mode)} · NOT_VALIDATED · REFERENCE_PENDING</p>
<video controls preload="metadata"><source src="{esc(video_name,quote=True)}" type="video/mp4">Trình duyệt chưa phát được video local.</video>
<p>Clip: <code>{esc(result['source']['video_id'])}</code>. Bài tập do caller khai báo; không model nhận dạng bài tập.</p>
<p>{c['available_frames']} / {c['total_frames']} frame có số đo; {c['unavailable_frames']} Không đủ điều kiện đo.
Sequence: <b>{esc(result['sequence']['action'])}</b>; metric toàn clip: <b>{esc(result['metric']['action'])}</b>.</p>
<p>Reasons: <code>{esc(', '.join(reasons) or 'none')}</code>. Null không phải góc0°; gate không được nâng theo từng frame.</p>
<p>Góc projected2D ước lượng từ processed coordinates; skeleton từ raw pose cùng frame/time.
Màu xanh ngọc observed / cam low visibility là trạng thái dữ liệu, không phải kỹ thuật đúng/sai.
Missing/ngoài ảnh không vẽ. Chưa kiểm chứng accuracy bằng tham chiếu độc lập; coverage không là accuracy.</p>
<p>camera_view=UNKNOWN; source viewpoint=unknown; documented=false. Assumption:
<code>{esc(json.dumps(result['assumption'],ensure_ascii=False))}</code>.</p>
<p>Phase, rep, point-turn, pause, score và technical feedback: Unavailable. Không chấm kỹ thuật.</p>
<p class="notice">official_eligibility=false · assessment=false · G5.7 blocked · G6 IN_PROGRESS · REFERENCE_PENDING</p>
<p><a href="{esc(video_name,quote=True)}">Video overlay</a> · <a href="{esc(result['source']['original_file'],quote=True)}">Video gốc</a> · <a href="HUONG_DAN_MO_DEMO.txt">Hướng dẫn</a></p>
<p><a href="frames.jsonl">Kết quả từng frame</a> · <a href="result.json">Kết quả đầy đủ</a> · <a href="quality-source.json">Quality nguồn</a> · <a href="quality-processing.json">Quality processing</a> · <a href="processing-metadata.json">Giả định</a> · <a href="receipt.json">Execution/timings</a> · <a href="manifest.json">Hashes</a> · <a href="video-verification.json">Decode receipt</a></p>
<p>Giữ nguyên cả thư mục. Browser/playback QA NOT_RUN; không cần mạng/server.
Nếu local file/codec bị chặn, dùng trình phát desktop có sẵn; không tắt bảo vệ hoặc đi vòng security policy.</p></html>'''


def render_single_video_bundle(output, *, progress=None):
    """Reuse paint/encoder/decode; no geometry or policy inference in renderer."""
    from .single_video import load_saved
    if progress is not None:progress.emit("render","started")
    result=load_saved(output)
    source=output/result["source"]["original_file"]
    if sha(source)!=result["source"]["video_sha256"]:raise ValueError("Copied original bytes differ")
    reader=OpenCVVideoReader(source)
    for name in ("fps","width","height","frame_count","timestamp_policy","decoder"):
        if str(getattr(reader.metadata,name))!=str(result["metadata"][name]):raise ValueError("Renderer metadata differs: "+name)
    target,codec,attempts,indices=encode_saved_frames(reader,output,result["frames"],result["side"],result["sequence"],
        localized_unavailable=True,presentation_mode=result["mode"],preserve_aspect=True,progress=progress)
    if progress is not None:
        progress.emit("render","completed",completed=len(indices),total=len(result["frames"]),unit="output_frames")
        progress.emit("video_validation","started",completed=0,total=len(result["frames"]),unit="decoded_frames")
    receipt=verify_video(target,len(result["frames"]),reader.metadata.fps,progress=progress)
    if progress is not None:progress.emit("video_validation","completed",completed=receipt["decoded_frames"],total=len(result["frames"]),unit="decoded_frames")
    receipt.update(video_file=target.name,codec=codec,codec_attempts=attempts,frame_map=indices,
        source_sha256=sha(source),geometry_calculated_by_renderer=False,
        source_resize="isotropic fit into640x480 with letterboxing; native raw skeleton coordinates; whole clip")
    write_json(output/"video-verification.json",receipt)
    with (output/"START_HERE.html").open("x",encoding="utf-8") as stream:stream.write(single_video_html(result,target.name))
    with (output/"HUONG_DAN_MO_DEMO.txt").open("x",encoding="utf-8-sig") as stream:
        stream.write("BODYWEIGHTSQUATS — "+result["mode"].upper()+" / NOT_VALIDATED / REFERENCE_PENDING\n"
            "Mở START_HERE.html hoặc "+target.name+". Giữ nguyên cả thư mục/JSON sidecars.\n"
            "Không cần mạng/server. Nếu file/codec bị chặn, không vượt security policy.\n"
            "Bên gối cố định. Góc projected2D từ processed; skeleton raw, màu biểu thị dữ liệu.\n"
            "Không đủ điều kiện đo/null không là0° hoặc lỗi kỹ thuật của người tập.\n"
            "Strict không giả định viewpoint; exploratory side chỉ là giả định CLI unverified.\n"
            "Chưa kiểm chứng accuracy bằng nhãn độc lập. Không rep/phase/score/tư vấn.\n"
            "official_eligibility=false; assessment=false; G5.7 blocked; G6 IN_PROGRESS.\n"
            "Chỉ coi run hoàn tất khi manifest.json và receipt.json status=completed.\n"
            "Browser/playback QA NOT_RUN.\n")
