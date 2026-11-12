"""Deterministic group/duplicate components; never call exposed groups pristine."""
import hashlib
import math
from .models import VideoIdentity, canonical_hash
from .evaluation import checked

SEED = 'squat-count-groups-20261024-v1'


def propose_split(sources, *, exposed_ids, related_sources=(), seed=SEED):
    sources = sorted((checked(VideoIdentity, s) for s in sources), key=lambda s: s.path)
    by_id = {s.video_id: s for s in sources}
    if len(by_id) != len(sources) or set(exposed_ids)-set(by_id):
        raise ValueError('Duplicate source or unknown exposure ID')
    parent = {k: k for k in by_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(max(a, b))] = find(min(a, b))
    for field in ('group_id', 'sha256'):
        seen = {}
        for s in sources:
            key = getattr(s, field)
            if key in seen: union(s.video_id, seen[key])
            seen[key] = s.video_id
    for edge in related_sources:
        if not edge.get('evidence_ref') or edge['a'] not in by_id or edge['b'] not in by_id:
            raise ValueError('Verified source relationship requires two known IDs and evidence')
        union(edge['a'], edge['b'])
    components = {}
    for s in sources: components.setdefault(find(s.video_id), []).append(s.video_id)
    groups = [sorted(members) for members in components.values()]
    groups.sort(key=lambda members: hashlib.sha256((seed+'|'+ '|'.join(members)).encode()).hexdigest())
    n = len(groups)
    # Small cohorts are not silently represented as a train/validation/test study.
    ntrain, nvalidation = (math.floor(.6*n), math.floor(.2*n)) if n >= 3 else (n, 0)
    if n >= 3: nvalidation = max(1, nvalidation); ntrain = max(1, min(ntrain, n-nvalidation-1))
    assignments = []
    for i, group in enumerate(groups):
        role = 'train' if i < ntrain else 'validation' if i < ntrain+nvalidation else 'development_test'
        for key in group:
            s = by_id[key]
            assignments.append(dict(video_id=key, path=s.path, sha256=s.sha256, group_id=s.group_id,
                component_id=canonical_hash(group), role=role, previously_exposed=key in exposed_ids))
    for field in ('sha256', 'group_id', 'component_id'):
        seen = {}
        for row in assignments:
            if row[field] in seen and seen[row[field]] != row['role']:
                raise AssertionError('Leakage in deterministic partition')
            seen[row[field]] = row['role']
    return dict(version='squat-count-split-proposal-v1', seed=seed,
        algorithm='SHA256(seed|sorted component video IDs), 60/20/20 component count; gXX/bytes/verified related-source union',
        assignments=sorted(assignments, key=lambda row: row['path']), component_count=n,
        related_sources=list(related_sources), subject_identity_available=False,
        pristine_holdout_established=False,
        reason='All evaluated exposure evidence retained; no independent unseen-subject guarantee. development_test is not a pristine holdout.',
        fit_policy='Fit normalization and feature selection on train only; validation chooses config; test never tunes.',
        training_authorized=False, reselection_by_performance_allowed=False)
