"""Separate weak AI references. Never accepted as human AnnotationPass/Bundle."""
from datetime import datetime
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .models import Model, Hash, Text, VideoIdentity, Span, Partial, Uncertain, canonical_hash, check_timeline
from .readiness import file_info, write_json


class SheetEvidence(Model):
    path: Text
    sha256: Hash
    frame_indices: tuple[int, ...]
    tool: Literal['view_image'] = 'view_image'

    @model_validator(mode='after')
    def ordered(self):
        if not self.frame_indices or tuple(sorted(set(self.frame_indices))) != self.frame_indices:
            raise ValueError('Viewed frame indices must be nonempty, unique and ordered')
        return self


class CycleHint(Model):
    """Uncertainty envelopes, not exhaustive event labels or pause intervals."""
    start_range: Span
    bottom_range: Span
    end_range: Span
    reason: Text

    @model_validator(mode='after')
    def ordered(self):
        if not (self.start_range.end.frame_index < self.bottom_range.start.frame_index
                <= self.bottom_range.end.frame_index < self.end_range.start.frame_index):
            raise ValueError('Start/bottom/end uncertainty envelopes must be ordered')
        return self


class ViewInterval(Model):
    interval: Span
    viewpoint: Literal['side','front','oblique','unknown']
    reason: Text


class AIViewpoint(Model):
    schema_version: Literal['squat-ai-viewpoint/0.1.0'] = 'squat-ai-viewpoint/0.1.0'
    protocol_version: Literal['squat-ai-visual-review-v1'] = 'squat-ai-visual-review-v1'
    protocol_sha256: Hash
    rubric_sha256: Hash
    source: VideoIdentity
    recorded_at: datetime
    author_kind: Literal['ai'] = 'ai'
    method: Literal['ai_visual_contact_sheet_review'] = 'ai_visual_contact_sheet_review'
    evidence_status: Literal['visually_assessed_unverified'] = 'visually_assessed_unverified'
    source_camera_view: Literal['UNKNOWN'] = 'UNKNOWN'
    documented: Literal[False] = False
    human_validated: Literal[False] = False
    independent_ground_truth: Literal[False] = False
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    assessment: Literal[False] = False
    evidence: tuple[SheetEvidence,...]
    reviewed_intervals: tuple[ViewInterval,...]
    declared_whole_clip_viewpoint: Literal['side','front','oblique','unknown']
    subject_description: Text
    camera_changes: bool
    camera_direction_notes: Text
    framing_notes: Text
    occlusion_notes: Text
    uncertainty_notes: Text

    @model_validator(mode='after')
    def integrity(self):
        if self.recorded_at.utcoffset() is None or self.source.fps is None:
            raise ValueError('Actual timezone and complete source required')
        support=[i for e in self.evidence for i in e.frame_indices]
        if sorted(support)!=list(range(self.source.frame_count)):
            raise ValueError('Whole viewpoint needs full displayed evidence, without duplicates')
        expected=0
        for item in self.reviewed_intervals:
            span=item.interval
            if span.start.frame_index!=expected:
                raise ValueError('View intervals must cover source without gaps/overlaps')
            for m in (span.start,span.end):
                if not math.isclose(m.timestamp_seconds,m.frame_index/self.source.fps,rel_tol=0,abs_tol=1e-9):
                    raise ValueError('View interval timeline mismatch')
            expected=span.end.frame_index+1
        if expected!=self.source.frame_count:
            raise ValueError('View intervals must cover whole source')
        views={item.viewpoint for item in self.reviewed_intervals}
        whole=next(iter(views)) if len(views)==1 else 'unknown'
        if self.declared_whole_clip_viewpoint!=whole:
            raise ValueError('Mixed/unknown view cannot become whole-clip side')
        return self


class AIAnnotation(Model):
    schema_version: Literal['squat-ai-labels/0.1.0'] = 'squat-ai-labels/0.1.0'
    protocol_version: Literal['squat-ai-visual-review-v1'] = 'squat-ai-visual-review-v1'
    protocol_sha256: Hash
    semantics_protocol_sha256: Hash
    source: VideoIdentity
    pass_id: Literal['A', 'B']
    author_kind: Literal['ai'] = 'ai'
    actor: Literal['Codex assistant; same conversation'] = 'Codex assistant; same conversation'
    method: Literal['ai_visual_contact_sheet_review'] = 'ai_visual_contact_sheet_review'
    recorded_at: datetime
    blinded_to_other_pass: Literal[False] = False
    independent_raters: Literal[False] = False
    human_validated: Literal[False] = False
    independent_ground_truth: Literal[False] = False
    evidence: tuple[SheetEvidence, ...]
    subject_description: Text
    whole_clip_count: int | None = Field(ge=0)
    observed_complete_count: int | None = Field(ge=0)
    count_reason: Text | None
    boundaries_status: Literal['unavailable'] = 'unavailable'
    boundaries_reason: Text
    cycle_hints: tuple[CycleHint, ...] = ()
    partials: tuple[Partial, ...] = ()
    uncertain: tuple[Uncertain, ...] = ()
    observations: Text
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    assessment: Literal[False] = False

    @model_validator(mode='after')
    def integrity(self):
        if self.recorded_at.utcoffset() is None or self.source.fps is None or not self.evidence:
            raise ValueError('Actual timezone/source timeline/viewed evidence required')
        indices = [i for e in self.evidence for i in e.frame_indices]
        if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= self.source.frame_count:
            raise ValueError('Evidence frame support duplicated or outside source')
        if self.whole_clip_count is None:
            if not self.count_reason:
                raise ValueError('Unavailable count requires a reason, never substitute zero')
        elif (self.count_reason is not None or self.observed_complete_count != self.whole_clip_count
              or set(indices) != set(range(self.source.frame_count)) or any(u.affects_count for u in self.uncertain)):
            raise ValueError('Whole count needs full viewed coverage and resolved count evidence')
        if self.cycle_hints and (self.observed_complete_count is None or len(self.cycle_hints) > self.observed_complete_count):
            raise ValueError('Hints cannot exceed observed complete cycles')
        check_timeline(self.source, (), self.partials, self.uncertain)
        for hint in self.cycle_hints:
            for span in (hint.start_range, hint.bottom_range, hint.end_range):
                for m in (span.start, span.end):
                    if m.frame_index >= self.source.frame_count or not math.isclose(m.timestamp_seconds, m.frame_index/self.source.fps, abs_tol=1e-9, rel_tol=0):
                        raise ValueError('Hint not on exact source timeline')
            if any(max(hint.start_range.start.frame_index, x.interval.start.frame_index) <= min(hint.end_range.end.frame_index, x.interval.end.frame_index) for x in (*self.partials, *self.uncertain)):
                raise ValueError('Hint crosses partial/uncertain support')
        if any(a.end_range.end.frame_index > b.start_range.start.frame_index for a,b in zip(self.cycle_hints,self.cycle_hints[1:])):
            raise ValueError('Hints overlap or are unordered')
        return self


def verify_evidence(annotation, root):
    """Verify bytes AND sheet/source/sample linkage; this cannot prove label accuracy."""
    import json
    root=Path(root).resolve()
    for relative, expected in [(annotation.source.path, annotation.source.sha256)] + [(e.path,e.sha256) for e in annotation.evidence]:
        resolved=(root/relative).resolve()
        if not resolved.is_relative_to(root) or file_info(resolved)['sha256'] != expected:
            raise ValueError('Source/evidence hash linkage mismatch')
    for evidence in annotation.evidence:
        sheet=(root/evidence.path).resolve()
        manifest=sheet.parent/'manifest.json'
        if not manifest.resolve().is_relative_to(root):
            raise ValueError('Sheet manifest outside workspace')
        data=json.loads(manifest.read_bytes())
        if (data['version'] != 'squat-ai-source-sheets-v1'
                or data['source'] != annotation.source.model_dump(mode='json')
                or data['decoded_frames'] != annotation.source.frame_count):
            raise ValueError('Sheet manifest source linkage mismatch')
        matches=[s for s in data['sheets'] if s['file']==sheet.name]
        if (len(matches)!=1 or tuple(matches[0]['frame_indices'])!=evidence.frame_indices
                or matches[0]['identity']!=file_info(sheet)
                or matches[0]['timestamps_seconds'] != [i/annotation.source.fps for i in evidence.frame_indices]):
            raise ValueError('Sheet manifest sample/hash linkage mismatch')


def save_pass(path, annotation, root):
    """Revalidate content/bytes and exclusively create a checkpoint."""
    import json
    annotation = AIAnnotation.model_validate_json(annotation.model_dump_json())
    verify_evidence(annotation, root)
    payload=json.loads(annotation.model_dump_json())
    write_json(path, dict(payload=payload, sha256=canonical_hash(payload)))


def load_pass(path):
    import json
    envelope=json.loads(Path(path).read_bytes())
    if set(envelope) != {'payload','sha256'} or canonical_hash(envelope['payload']) != envelope['sha256']:
        raise ValueError('AI checkpoint hash mismatch')
    return AIAnnotation.model_validate_json(json.dumps(envelope['payload']))


def compare_passes(a, b):
    if a.pass_id != 'A' or b.pass_id != 'B' or a.source != b.source or a.protocol_sha256 != b.protocol_sha256 or a.semantics_protocol_sha256 != b.semantics_protocol_sha256:
        raise ValueError('Pass/source/protocol identity mismatch')
    fields=('whole_clip_count','observed_complete_count','partials','uncertain','cycle_hints')
    differences=[key for key in fields if getattr(a,key) != getattr(b,key)]
    eligible=a.whole_clip_count is not None and b.whole_clip_count is not None
    return dict(kind='AI_pass_consistency_not_independent_agreement', comparable_count=eligible,
                count_equal=(a.whole_clip_count==b.whole_clip_count) if eligible else None,
                differences=differences, human_adjudicated=False, independent_ground_truth=False)


def conservative_pair(a, b):
    """Keep unresolved labels null; never average counts or synthesize boundaries."""
    comparison=compare_passes(a,b)
    accepted=comparison['count_equal'] is True
    observed_equal=a.observed_complete_count==b.observed_complete_count
    return dict(comparison=comparison,
                status='AI_COUNT_CONSISTENT' if accepted else 'AI_COUNT_UNRESOLVED',
                whole_clip_count=a.whole_clip_count if accepted else None,
                count_reason=None if accepted else 'At least one pass is unavailable or passes disagree; retain both references',
                observed_complete_count=a.observed_complete_count if observed_equal else None,
                observed_reason=None if observed_equal else 'Observed lower bounds disagree; no automatic selection',
                boundaries=None, boundaries_reason='Exact exhaustive event labels not established',
                method='conservative_AI_reconciliation_no_new_observation',
                validation_status='NOT_VALIDATED',official_eligibility=False,assessment=False,
                human_adjudicated=False,independent_ground_truth=False)


def summarize_pairs(pairs, expected_ids):
    """Full cohort accounting. Missing/extra/duplicate input is an error, not a drop."""
    from collections import Counter
    ids=[a.source.video_id for a,b in pairs]
    if len(ids)!=len(set(ids)) or len(expected_ids)!=len(set(expected_ids)) or set(ids)!=set(expected_ids):
        raise ValueError('Incomplete/duplicate/extra AI cohort')
    records=[dict(video_id=a.source.video_id,source=a.source.model_dump(mode='json'),
                  pass_a_count=a.whole_clip_count,pass_b_count=b.whole_clip_count,
                  **conservative_pair(a,b)) for a,b in pairs]
    comparable=sum(r['comparison']['comparable_count'] for r in records)
    equal=sum(r['comparison']['count_equal'] is True for r in records)
    differences=Counter(k for r in records for k in r['comparison']['differences'])
    return dict(records=records, summary=dict(
        clips=len(pairs),passes=2*len(pairs),unique_frames=sum(a.source.frame_count for a,b in pairs),
        comparable_count_pairs=comparable,matching_count_pairs=equal,
        count_consistency_fraction=equal/comparable if comparable else None,
        noncomparable_count_pairs=len(pairs)-comparable,
        count_unresolved_clips=sum(r['whole_clip_count'] is None for r in records),
        count_available_clips=sum(r['whole_clip_count'] is not None for r in records),
        known_count_sum=sum(r['whole_clip_count'] for r in records if r['whole_clip_count'] is not None),
        known_count_sum_scope='Available whole-clip AI references only; not total cohort repetitions',
        observed_count_disagreements=sum('observed_complete_count' in r['comparison']['differences'] for r in records),
        difference_fields=dict(differences),clips_with_any_difference=sum(bool(r['comparison']['differences']) for r in records),
        exact_exhaustive_boundaries_available=0,pending_passes=0,
        independent_raters=False,independent_ground_truth=False,human_validated=False))
