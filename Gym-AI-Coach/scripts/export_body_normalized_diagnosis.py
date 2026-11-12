"""Export saved diagnostic tables and a targeted pose overlay; no model execution.

The frame interval is selected from the recorded largest coordinate steps, not
from a new quality threshold. Images are not annotations or pose ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

from gym_coach.counting.inner_cv.manifest import unseal
from gym_coach.counting.readiness import file_info, write_json


JOINTS = ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle')


def linked(root, link):
    p = (root / link['path']).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Source outside workspace')
    if file_info(p)['sha256'] != link['sha256']:
        raise ValueError('Source hash mismatch: ' + link['path'])
    return p


def pixel_xy(row, joint, height, width):
    # Both raw axes are in image-height units, with image center as origin.
    return (row[7 * joint] * height + width / 2,
            row[7 * joint + 1] * height + height / 2)


def markdown_table(headers, rows):
    def cell(x):
        if isinstance(x, float):
            return f'{x:.6f}'
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(cell(x) for x in r) + ' |' for r in rows])


def tables(a):
    text = ['# Bảng chẩn đoán fold 1',
            'AI weak reference; đơn vị lỗi là lần squat. Signed error = prediction − target; ΔAE = body − raw. Fold=1 cho mọi dòng held-out. Không làm tròn prediction trước tính lỗi.']
    for seed, paired in a['paired'].items():
        rows = []
        for r in paired['records']:
            line = [r['video_id'], r['group_id'], r['target'], r['duration_seconds']]
            for method in ('raw58', 'body_normalized', 'duration'):
                line += [r['predictions'][method], r['signed_errors'][method], r['absolute_errors'][method]]
            rows.append(line + [r['delta_AE']])
        text += ['## Seed ' + seed, markdown_table(
            ['Video', 'Group', 'AI count', 'Duration s', 'Raw pred', 'Raw SE', 'Raw AE',
             'Body pred', 'Body SE', 'Body AE', 'Duration pred', 'Duration SE', 'Duration AE', 'ΔAE'], rows),
            'Phân rã nhóm (đóng góp vào ΔMAE = tổng ΔAE nhóm / 13; có thể âm):',
            '```json\n' + json.dumps(paired['groups'], indent=2, ensure_ascii=False) + '\n```']
    rows = []
    for v, r in sorted(a['metadata'].items()):
        p = r['pose']
        rows.append([v, r['side'], r['target'], r['duration_seconds'],
                     f"{p['complete_frames']}/{p['frames']}", p['scale'],
                     p['longest_gap_frames'], p['restarts_after_gap'], p['torso']['population_cv'],
                     p['hip']['y_range'], p['steps']['raw_max_joint_step']['maximum'],
                     p['steps']['normalized_max_joint_step']['maximum']])
    text += ['## Đủ 20 train và 13 held-out của fold 1',
             'Scale, hip-y range và raw step theo chiều cao ảnh; normalized step theo đơn vị torso. Chỉ xét cạnh giữa hai frame complete kề nhau; không nối qua gap.',
             markdown_table(['Video', 'Side', 'AI count', 'Duration s', 'Complete/N', 'Scale',
                             'Longest gap', 'Restarts', 'Torso CV', 'Hip y range', 'Max raw step', 'Max norm step'], rows),
             '## Phân bố và chồng lấp', '```json\n' + json.dumps(a['train_vs_held'], indent=2, ensure_ascii=False) + '\n```']
    return '\n\n'.join(text) + '\n'


def export(root, run, output):
    import cv2
    from PIL import Image, ImageDraw
    a = unseal(json.loads((run / 'analysis.json').read_text(encoding='utf-8')))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'TABLES.md').write_text(tables(a), encoding='utf-8')
    video = 'v_BodyWeightSquats_g19_c02.avi'
    r = a['metadata'][video]
    source = linked(root, r['source'])
    cache_path = linked(root, r['input_artifact'])
    cache = json.loads(cache_path.read_text(encoding='utf-8'))
    if cache['source'] != r['source']:
        raise ValueError('Cache/source identity mismatch')
    # Largest-step focus: previous frame of earliest ranked step minus one,
    # through latest ranked step plus one. No anatomy/quality threshold added.
    top = r['pose']['largest_steps']
    lo = max(0, min(x['previous_frame'] for x in top) - 1)
    hi = min(r['source']['frame_count'] - 1, max(x['frame_index'] for x in top) + 1)
    board = Image.new('RGB', (960, 30 + (hi - lo + 1) * 276), '#101820')
    draw = ImageDraw.Draw(board)
    draw.text((4, 7), 'Original | saved raw pose: L cyan / R orange | named-joint coordinates; no new inference', fill='white')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot decode video')
    decoded = 0; records = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if lo <= decoded <= hi:
            y = 30 + (decoded - lo) * 276
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            board.paste(im, (0, y)); board.paste(im, (320, y))
            row = cache['values'][decoded]; joints = []
            for j, name in enumerate(JOINTS):
                x, z = pixel_xy(row, j, 240, 320)
                color = '#51e4ff' if j % 2 == 0 else '#ffac49'
                draw.ellipse((x+317, z+y-3, x+323, z+y+3), fill=color)
                draw.text((x+324, z+y-7), name[0] + name.split('_')[1][0], fill=color)
                joints.append(dict(joint=name, raw_x=row[7*j], raw_y=row[7*j+1],
                                   pixel_x=x, pixel_y=z, visibility=row[7*j+2],
                                   usable=row[7*j+3], observed=row[7*j+4],
                                   low_visibility=row[7*j+5], missing=row[7*j+6]))
                draw.text((646, y+8+25*j), f'{name}: ({x:.1f}, {z:.1f}) v={row[7*j+2]:.3f}', fill=color)
            draw.text((4, y+244), f'frame {decoded} / {decoded/25:.2f}s; complete={cache["complete"][decoded]}', fill='white')
            records.append(dict(frame_index=decoded, timestamp_seconds=decoded/25,
                                complete=cache['complete'][decoded], joints=joints))
        decoded += 1
    cap.release()
    if decoded != r['source']['frame_count']:
        raise ValueError('Decode frame count mismatch')
    dest = output / 'g19-c02-named-joints-088-094.jpg'
    board.save(dest, quality=95)
    write_json(output / 'targeted-pose.json', dict(
        source=r['source'], input_artifact=r['input_artifact'], frame_selection='top3 raw steps plus adjacent context',
        frames=records, original_images=decoded, image=file_info(dest),
        view_status='EXTRACTED_NOT_YET_VIEWED', model_inference=0, new_labels=0))
    write_json(output / 'export-receipt.json', dict(
        script=dict(path=Path(__file__).relative_to(root).as_posix(), **file_info(Path(__file__))),
        analysis=dict(path=(run/'analysis.json').relative_to(root).as_posix(), **file_info(run/'analysis.json')),
        files=[dict(path=p.name, **file_info(p)) for p in output.iterdir() if p.is_file()],
        pose_inference=0, model_inference=0, training=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    export(Path.cwd().resolve(), args.run_dir.resolve(), args.output_dir.resolve())
