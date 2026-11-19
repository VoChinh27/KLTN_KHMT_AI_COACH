"""Chẩn đoán OOF từng clip: biên thật vs dự đoán, độ tự tin của classifier, accuracy theo frame."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import argparse, csv
import numpy as np
import train_phase_model as T
from stage3_features.phase_model.core.phase_dataset import load_all
from stage3_features.phase_model.core.phase_features import FeatureConfig
from stage3_features.phase_model.core.phase_decoder import DurationPriors, PhaseDecoder
from stage3_features.phase_model.eval.phase_metrics import rep_boundaries
from stage3_features.phase_model.core.phase_labels import IGNORE_LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phases-dir', type=Path, default=Path('data/phases'))
    ap.add_argument('--pose-cache-dir', type=Path, default=Path('data/pose_cache'))
    ap.add_argument('--group', choices=('all', 'coach', 'trainee'), default='coach')
    ap.add_argument('--backend', choices=('lightgbm', 'logreg'), default='lightgbm')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-penalty', type=float, default=4.0)
    ap.add_argument('--emission-scale', type=float, default=0.5)
    ap.add_argument('--out', type=Path, default=Path('reports/phase_diag.csv'))
    args = ap.parse_args()

    clips = load_all(args.phases_dir, args.pose_cache_dir, group=args.group)
    clips = [c for c in clips if c.phaseset.marked_by in ('manual', 'auto_reviewed')]
    people = sorted({c.person for c in clips})
    cfg = FeatureConfig()
    rows = []
    bnames = None

    for k, held in enumerate(people, 1):
        T.log(f'fold {k}/{len(people)} held={held}')
        train = [c for c in clips if c.person != held]
        test = [c for c in clips if c.person == held]
        clf, _, _ = T.fit_classifier(train, args.backend, args.seed, False, cfg)
        priors = DurationPriors.fit(T.durations(train))
        decoder = PhaseDecoder(priors=priors, skip_penalty=args.skip_penalty,
                               emission_scale=args.emission_scale)
        for c in test:
            X, _ = T.clip_features(c, cfg)
            probs = np.asarray(clf.predict_proba(X))
            rep = decoder.decode(probs, c.fps).to_rep()
            tb = rep_boundaries(c.rep)
            pb = rep_boundaries(rep)
            if bnames is None:
                bnames = list(tb.keys())
            labels = c.pose_labels()
            valid = labels != IGNORE_LABEL
            # Giả định cột xác suất bắt đầu từ nhãn nhỏ nhất (xấp xỉ, chỉ để tham khảo)
            off = labels[valid].min()
            frame_acc = float((probs.argmax(1)[valid] == (labels[valid] - off)).mean())
            row = {'stem': c.stem, 'n_frames': len(labels), 'fps': c.fps,
                   'peak_prob': float(probs.max(1).mean()), 'frame_acc': frame_acc}
            for b in tb:
                t, p = tb[b], pb.get(b)
                row[f'{b}|truth'] = t
                row[f'{b}|pred'] = p
                row[f'{b}|diff'] = (None if (t is None or p is None) else p - t)
            rows.append(row)

    print('\n=== Từng clip (diff = pred - truth, frame; MISS = decoder không ra biên) ===')
    for r in rows:
        parts = []
        for b in bnames:
            d = r[f'{b}|diff']
            parts.append(f'{b}:{"MISS" if d is None else f"{d:+d}"}')
        print(f'{r["stem"]:<22} n={r["n_frames"]:<5} peak={r["peak_prob"]:.2f} '
              f'facc={r["frame_acc"]:.2f}  ' + '  '.join(parts))

    print('\n=== Tổng kết ===')
    print(f'mean peak_prob = {np.mean([r["peak_prob"] for r in rows]):.3f}  '
          f'(gần 1.0 nghĩa là classifier quá cực đoan -> emission-scale gần như vô tác dụng)')
    print(f'mean frame_acc = {np.mean([r["frame_acc"] for r in rows]):.3f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'\nĐã ghi {args.out}')


if __name__ == '__main__':
    main()
