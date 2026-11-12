"""Package an existing saved-synthetic showcase for local handoff only.

No source selection, noise audit, pipeline, detector, or network is run.
python -m gym_coach.reporting.showcase_handoff --bundle <old-dir> --output-dir <new-dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from ..segmentation.point_resolution_models import read_point_resolution_json
from .squat_models import SquatFeedbackResult
from .squat_showcase import build_showcase_payload,presentation_error,render_showcase
from .squat_showcase_demo import _json


GUIDE_TEXT='''GYM RESULT FEEDBACK — BODYWEIGHTSQUATS
EXPERIMENTAL / NOT VALIDATED · SAVED SYNTHETIC ARTIFACTS

MỞ DEMO
1. Giữ nguyên cả thư mục bàn giao. Nếu nhận file ZIP, giải nén toàn bộ trước.
2. Mở START_HERE.html rồi chọn “Mở demo”, hoặc mở index.html bằng trình duyệt
   desktop có sẵn (ví dụ Edge/Chrome). Có thể dùng Ctrl+O để chọn file local.
3. Không cần Python, Internet, server, cài dependency hoặc thay cài đặt bảo mật.
   Nếu chính sách máy chặn file local, dừng; không tắt bảo vệ hoặc dựng server.

GIỮ CẠNH NHAU
- START_HERE.html: hướng dẫn và link mở demo.
- index.html: giao diện; CSS/JavaScript/dữ liệu đã nhúng.
- manifest.json, presentation.json, noise-review.json: sidecar/hash/provenance.
- selection.json, upstream-manifest.json: nguồn lựa chọn và receipt bundle gốc.
- sources/ và toàn bộ JSON bên trong: snapshot nguồn, cần cho các link evidence.
- HUONG_DAN_MO_DEMO.txt: bản hướng dẫn này.
Không đổi tên/tách file. Riêng index.html vẫn chứa dữ liệu trình bày, nhưng
các link JSON sẽ hỏng nếu bỏ sidecars. Nên sao chép nguyên thư mục.

TRÌNH BÀY NHANH
- Panel A: chọn nguồn pose, Phát A hoặc tua frame; thử synthetic-gap ở frame60
  để thấy missing joint và metric Unavailable. Số đo chỉ đọc từ JSON đã lưu.
- Panel B: chọn smooth/plateau/clip_edge, tua hoặc Phát B để xem PVP events và
  reasons. Panel này là scalar fixture khác nguồn, không có skeleton.
- Hai panel có clock và scrubber độc lập. Không gắn metric A với event B,
  kể cả khi cùng FPS/frame number. Có liên kết “Đến panel A/B” trên đầu trang.
- Tab/Shift+Tab đổi điều khiển; Enter/Space bật nút. Khi slider có focus,
  dùng phím mũi tên, Home/End. Select dùng phím mũi tên. Không có phím tắt toàn cục.

Ý NGHĨA TRẠNG THÁI
- Unavailable/null: chưa đủ evidence hoặc quality để có kết quả, không phải0,
  không chứng minh người tập sai, không khẳng định không có pause/repetition.
- EXPERIMENTAL / NOT VALIDATED: prototype synthetic chưa validated trên người.
- PVP là cycle/event candidate; raw-condition pass không phải accepted point-turn.
- Point-turn/phase/arrival/departure/pause đang unavailable, accepted point=0.
  Không được diễn giải pause bằng0s hoặc tự thêm phase.

VÌ SAO KHÔNG CHẤM KỸ THUẬT
Dữ liệu là synthetic; phép đo/temporal chưa validated, official gate G5.7 còn
blocked và chưa có bộ quy tắc kỹ thuật đã được xác nhận để chấm. Demo không
có form score, lời sửa kỹ thuật hoặc kết luận đúng/sai. assessment=false,
official_eligibility=false, G6 IN_PROGRESS. Không inference/detector mới.

QA
browser_qa=NOT_RUN trong môi trường công cụ hiện tại; không có screenshot hoặc
tuyên bố đã visual QA desktop. Kiểm tra offline/JavaScript không thay browser QA.
Noise review chỉ là evidence cũ được giữ nguyên; không audit lại trong handoff.
'''


START_HTML='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Mở demo · Gym Result Feedback</title><style>
body{font:17px/1.6 system-ui;margin:2rem auto;max-width:850px;padding:0 20px;background:#101b2a;color:#edf5fc}
a{color:#91e4dc}a:focus-visible{outline:3px solid #ffd789;outline-offset:5px}.open{display:inline-block;background:#224d56;color:#fff;padding:14px 24px;border-radius:8px;font-weight:700}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui;color:#c2d1df}.warning{color:#ffd789}
</style><main><h1>Gym Result Feedback</h1><p>BodyWeightSquats · bàn giao demo offline</p>
<p class="warning">EXPERIMENTAL / NOT VALIDATED · SYNTHETIC ONLY</p>
<p><a class="open" href="index.html">Mở demo</a></p><p>Mở file local bằng trình duyệt desktop có sẵn. Không cần server hoặc Internet.</p>
<p><a href="HUONG_DAN_MO_DEMO.txt">Hướng dẫn dạng văn bản</a> · <a href="manifest.json">Manifest / hashes</a></p>
<pre>__GUIDE__</pre></main></html>'''


def _relative(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not value or any(p.lower()=='data' for p in path.parts):
        raise ValueError('Bundle paths must stay local and cannot refer to Data')
    return path


def read_saved_bundle(bundle_dir,*,workspace=None):
    workspace=Path(workspace or Path.cwd()).resolve()
    bundle=Path(bundle_dir)
    bundle=(workspace/bundle).resolve() if not bundle.is_absolute() else bundle.resolve()
    try:
        if not bundle.is_relative_to((workspace/'outputs').resolve()):
            raise ValueError('Only a saved bundle beneath workspace outputs/ is allowed')
        manifest_raw=(bundle/'manifest.json').read_bytes();manifest=_json(manifest_raw)
        if (manifest.get('showcase_version')!='gym-squat-showcase-v1' or manifest.get('synthetic_only') is not True
                or manifest.get('official_eligibility') is not False or manifest.get('assessment_enabled') is not False
                or manifest.get('validation_status')!='NOT_VALIDATED'
                or manifest.get('g57')!='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT' or manifest.get('g6')!='IN_PROGRESS'
                or manifest.get('linkage',{}).get('mode')!='independent_panels'
                or manifest.get('linkage',{}).get('synchronized') is not False
                or manifest.get('linkage',{}).get('temporal_overlay_applied_to_skeleton') is not False):
            raise ValueError('Expected the saved synthetic showcase with disabled gates')
        files={}
        for name,expected in manifest['files'].items():
            path=(bundle/_relative(name)).resolve()
            if not path.is_relative_to(bundle):raise ValueError('Bundle symlink/path escapes input directory')
            raw=path.read_bytes()
            if not re.fullmatch(r'[a-f0-9]{64}',expected) or hashlib.sha256(raw).hexdigest()!=expected:
                raise ValueError(f'Bundle hash mismatch: {name}')
            files[name]=raw
        refs=manifest['source_artifacts'];by_role={r['role']:r for r in refs}
        if len(by_role)!=len(refs):raise ValueError('Duplicate bundle source roles')
        def snapshot(role):
            ref=by_role[role];name=ref['snapshot_path'];_relative(name)
            if hashlib.sha256(files[name]).hexdigest()!=ref['sha256']:raise ValueError('Source snapshot mismatch')
            return files[name]
        feedbacks=tuple(SquatFeedbackResult.model_validate_json(snapshot(f'feedback-{i}'),strict=True)
                        for i in range(manifest['feedback_cases']))
        points=tuple(read_point_resolution_json(snapshot(f'point-{i}').decode('utf-8'))
                     for i in range(manifest['point_cases']))
        # Frozen review is copied as metadata; no schema/search/sample noise audit.
        noise=_json(files['noise-review.json'])
        original_payload=_json(files['presentation.json'])
        rebuilt=build_showcase_payload(feedbacks,points,source_refs=refs,noise_review=noise)
        if rebuilt!=original_payload:
            raise ValueError('Presentation must exactly match saved synthetic snapshots/provenance')
        return dict(bundle=bundle,manifest=manifest,manifest_raw=manifest_raw,files=files,
            feedbacks=feedbacks,points=points,refs=refs,noise=noise)
    except (ValueError,TypeError,KeyError,AttributeError,OSError) as exc:
        raise presentation_error(f'Cannot hand off bundle: {exc}') from exc


def write_handoff(bundle_dir,output_dir,*,workspace=None):
    import html
    workspace=Path(workspace or Path.cwd()).resolve();output=Path(output_dir)
    output=(workspace/output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to((workspace/'outputs').resolve()):
        raise presentation_error('Handoff output must be a new workspace outputs/ directory')
    if output.exists():raise FileExistsError(f'Refusing existing handoff output: {output}')
    loaded=read_saved_bundle(bundle_dir,workspace=workspace)
    page=render_showcase(loaded['feedbacks'],loaded['points'],source_refs=loaded['refs'],
        noise_review=loaded['noise'],handoff_guide=True)
    output.mkdir(parents=True,exist_ok=False)
    def write(name,raw):
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    copied={}
    for name,raw in loaded['files'].items():
        if name=='index.html':continue
        write(name,raw);copied[name]=hashlib.sha256(raw).hexdigest()
    write('index.html',page.encode('utf-8'))
    write('upstream-manifest.json',loaded['manifest_raw'])
    write('START_HERE.html',START_HTML.replace('__GUIDE__',html.escape(GUIDE_TEXT)).encode('utf-8'))
    write('HUONG_DAN_MO_DEMO.txt',GUIDE_TEXT.encode('utf-8-sig'))
    files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in output.rglob('*') if p.is_file()}
    result=dict(handoff_version='gym-squat-demo-handoff-v1',synthetic_only=True,
        official_eligibility=False,assessment_enabled=False,validation_status='NOT_VALIDATED',
        g57='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT',g6='IN_PROGRESS',browser_qa='NOT_RUN',
        screenshots=[],noise_audit_repeated=False,new_detector=False,new_inference=False,
        upstream_bundle=loaded['bundle'].relative_to(workspace).as_posix(),
        upstream_manifest_sha256=hashlib.sha256(loaded['manifest_raw']).hexdigest(),
        linkage=loaded['manifest']['linkage'],source_artifacts=loaded['refs'],
        copied_payload_files=copied,files=files)
    write('manifest.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    result=write_handoff(args.bundle,args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir),start_file='START_HERE.html',browser_qa=result['browser_qa'],
        status='NOT_VALIDATED',noise_audit_repeated=False),ensure_ascii=False))


if __name__=='__main__':main()
