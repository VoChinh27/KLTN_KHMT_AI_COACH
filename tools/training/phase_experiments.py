"""Thí nghiệm đánh giá: learning curve, kiểm tra tập HLV được chọn, mô tả clip.

Chế độ:
  --mode curve     : train với k người ngẫu nhiên, chấm trên người còn lại (lặp nhiều lần)
  --mode subset    : train trên danh sách người trong --subset-file, chấm trên phần còn lại,
                     so với các tập ngẫu nhiên cùng kích thước
  --mode describe  : in bảng thông tin từng clip để chọn HLV theo tiêu chí khách quan
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import argparse, random
import numpy as np
import train_phase_model as T
from stage3_features.phase_model.core.phase_dataset import load_all
from stage3_features.phase_model.core.phase_features import FeatureConfig
from stage3_features.phase_model.core.phase_decoder import DurationPriors, PhaseDecoder
from stage3_features.phase_model.eval.phase_metrics import per_clip_within_tol


def evaluate(train, test, a, cfg):
    clf, _, _ = T.fit_classifier(train, a.backend, a.seed, False, cfg)
    priors = DurationPriors.fit(T.durations(train))
    dec = PhaseDecoder(priors=priors, skip_penalty=a.skip_penalty, emission_scale=a.emission_scale)
    truth, pred = [], []
    for c in test:
        X, _ = T.clip_features(c, cfg)
        pred.append(dec.decode(clf.predict_proba(X), c.fps).to_rep())
        truth.append(c.rep)
    return float(np.mean(per_clip_within_tol(truth, pred, a.tolerance_frames)))


def by_people(clips, people):
    s = set(people)
    return [c for c in clips if c.person in s]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--mode', choices=('curve', 'subset', 'describe'), required=True)
    ap.add_argument('--phases-dir', type=Path, default=Path('data/phases'))
    ap.add_argument('--pose-cache-dir', type=Path, default=Path('data/pose_cache'))
    ap.add_argument('--group', choices=('all', 'coach', 'trainee'), default='coach')
    ap.add_argument('--backend', choices=('lightgbm', 'logreg'), default='lightgbm')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tolerance-frames', type=int, default=3)
    ap.add_argument('--skip-penalty', type=float, default=4.0)
    ap.add_argument('--emission-scale', type=float, default=0.5)
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 8, 12, 16])
    ap.add_argument('--n-test', type=int, default=6, help='số người giữ lại làm test cố định mỗi lần lặp (mode curve)')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--subset-file', type=Path, help='mỗi dòng 1 tên người, ví dụ coach:HLV01')
    a = ap.parse_args()

    clips = load_all(a.phases_dir, a.pose_cache_dir, group=a.group)
    clips = [c for c in clips if c.phaseset.marked_by in ('manual', 'auto_reviewed')]
    people = sorted({c.person for c in clips})
    cfg = FeatureConfig()
    T.log(f'{len(clips)} clips / {len(people)} people; backend={a.backend}')

    if a.mode == 'describe':
        print(f'{"person":<18}{"fps":>5}{"frames":>8}{"conf":>7}{"src":>15}  độ dài pha (giây)')
        for c in sorted(clips, key=lambda x: x.person):
            conf = float(np.mean([np.mean(np.asarray(p.conf)) for p in c.poses]))
            durs = ' '.join(f'{n}={w.n_frames / c.fps:.2f}' for n, w in c.rep.present())
            print(f'{c.person:<18}{c.fps:>5.0f}{len(c.poses):>8}{conf:>7.2f}{c.phaseset.marked_by:>15}  {durs}')
        return

    if a.mode == 'curve':
        res = {k: [] for k in a.ks}
        for r in range(a.repeats):
            rng = random.Random(a.seed + r)
            sh = people[:]
            rng.shuffle(sh)
            test_p, pool = sh[:a.n_test], sh[a.n_test:]
            test = by_people(clips, test_p)
            for k in a.ks:
                if k > len(pool):
                    continue
                train = by_people(clips, pool[:k])  # tập lồng nhau: k nhỏ nằm trong k lớn
                acc = evaluate(train, test, a, cfg)
                res[k].append(acc)
                T.log(f'repeat {r + 1}/{a.repeats} k={k} acc={acc:.3f}')
        print(f'\n=== Learning curve (test {a.n_test} người/lần, tol={a.tolerance_frames} frame) ===')
        for k in a.ks:
            v = res[k]
            if v:
                print(f'k={k:<3} mean={np.mean(v):.3f}  std={np.std(v):.3f}  n={len(v)}')
        print('\nĐọc: nếu mean còn tăng rõ ở k lớn nhất -> thêm data sẽ giúp.\n'
              'Nếu gần như phẳng (chênh < std) -> nút thắt là feature/nhãn.')
        return

    if a.mode == 'subset':
        if not a.subset_file or not a.subset_file.exists():
            raise SystemExit('Cần --subset-file')
        chosen = [l.strip() for l in a.subset_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        unknown = [p for p in chosen if p not in people]
        if unknown:
            raise SystemExit(f'Không tìm thấy: {unknown}\nTên hợp lệ: {people}')
        train = by_people(clips, chosen)
        test = [c for c in clips if c.person not in set(chosen)]
        acc = evaluate(train, test, a, cfg)
        print(f'\nTập chọn ({len(chosen)} người) -> test {len({c.person for c in test})} người còn lại: acc={acc:.3f}')
        rand = []
        for r in range(a.repeats):
            rng = random.Random(a.seed + r)
            pick = rng.sample(people, len(chosen))
            tr = by_people(clips, pick)
            te = [c for c in clips if c.person not in set(pick)]
            rand.append(evaluate(tr, te, a, cfg))
            T.log(f'random {r + 1}/{a.repeats} acc={rand[-1]:.3f}')
        print(f'Ngẫu nhiên cùng kích thước: mean={np.mean(rand):.3f} std={np.std(rand):.3f} (n={len(rand)})')


if __name__ == '__main__':
    main()
