"""Assemble two independent passes for review; never adjudicate automatically."""
import argparse
import json
from pathlib import Path

from .evaluation import checked, disagreement
from .models import AnnotationBundle, AnnotationPass, VideoIdentity


def assemble(manifest, manifest_sha256, pass_a, pass_b):
    if manifest['version'] != 'squat-count-dataset-v1':
        raise ValueError('Unknown dataset manifest version')
    sources = {r['source']['video_id']: checked(VideoIdentity, r['source']) for r in manifest['videos']}
    if len(sources) != len(manifest['videos']): raise ValueError('Duplicate video in cohort')
    indexes = []
    for data in (pass_a, pass_b):
        if data['version'] != 'squat-count-pass-collection-v1' or data['dataset_manifest_sha256'] != manifest_sha256:
            raise ValueError('Pass collection version/manifest linkage mismatch')
        rows = [checked(AnnotationPass, r) for r in data['annotations']]
        index = {r.source.video_id: r for r in rows}
        if len(index) != len(rows) or set(index) != set(sources):
            raise ValueError('Every manifest video needs one pass record, including pending/unavailable')
        for key, row in index.items():
            if row.source != sources[key] or row.protocol_sha256 != manifest['protocol_sha256']:
                raise ValueError('Pass source/protocol identity mismatch')
        indexes.append(index)
    bundles = [AnnotationBundle(data_kind=manifest['data_kind'], pass_a=indexes[0][key], pass_b=indexes[1][key])
               for key in sources]
    return (dict(version='squat-count-adjudication-collection-v1', dataset_manifest_sha256=manifest_sha256,
                 bundles=[b.model_dump(mode='json') for b in bundles]),
            [dict(video_id=b.pass_a.source.video_id, **disagreement(b)) for b in bundles])


def main(argv=None):
    from .readiness import file_info, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'pass-a', 'pass-b', 'output-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(argv)
    collection, review = assemble(json.loads(args.manifest.read_bytes()), file_info(args.manifest)['sha256'],
        json.loads(args.pass_a.read_bytes()), json.loads(args.pass_b.read_bytes()))
    out = args.output_dir.resolve()
    if not out.is_relative_to(Path.cwd().resolve()/'outputs'):
        raise ValueError('Output must be new child directory under outputs')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'adjudication.pending.json', collection)
    write_json(out/'disagreements.json', review)
    write_json(out/'receipt.json', dict(inputs={str(p): file_info(p) for p in (args.manifest, args.pass_a, args.pass_b)},
        outputs={p.name: file_info(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(videos=len(review), auto_adjudicated=0)))


if __name__ == '__main__': main()
