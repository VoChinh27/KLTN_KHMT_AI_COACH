"""Additive point-resolution audit; frozen temporal v2 remains untouched."""
import json
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from ..contracts import ContractModel, ErrorCategory, SourceTier, Stage
from .models import Hash, content_hash, strict_json
from .temporal_v2_models import SquatTemporalResultV2, Support, read_temporal_json


class PointResolutionPolicy(ContractModel):
    policy_version: Literal['squat-v2-point-resolution-v1'] = 'squat-v2-point-resolution-v1'
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    raw_rule: Literal['strict_adjacent_observed_minimum'] = 'strict_adjacent_observed_minimum'
    candidate_policy: Literal['unchanged_default_s3_full_pvp_only'] = 'unchanged_default_s3_full_pvp_only'
    unstable_policy: Literal['inclusive_overlap_or_global_reject'] = 'inclusive_overlap_or_global_reject'
    noise_policy: Literal['no_independent_noise_evidence_abstain'] = 'no_independent_noise_evidence_abstain'

    def content_hash(self):
        return content_hash(self)


class PointReason(ContractModel):
    reason: Literal['noise_not_identifiable', 'raw_minimum_not_strict',
                    'neighbor_unavailable', 'missing_candidate_sample', 'imputed_support',
                    'nonobserved_support', 'side_switch', 'candidate_at_clip_boundary',
                    'unstable_signal_overlap', 'no_full_pvp_candidate']
    category: Literal[ErrorCategory.INSUFFICIENT_DATA] = ErrorCategory.INSUFFICIENT_DATA
    stage: Literal[Stage.SEGMENTATION] = Stage.SEGMENTATION
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    support: Support | None = None


class PointSupportCheck(ContractModel):
    bottom_frame: int = Field(ge=0, strict=True)
    candidate_support: Support
    raw_support: Support
    input_hash: Hash
    strict_local_minimum: StrictBool | None
    necessary_conditions_met: StrictBool
    status: Literal['unavailable'] = 'unavailable'
    point_turn_event: None = None
    arrival: None = None
    departure: None = None
    pause_duration_seconds: None = None
    reasons: tuple[PointReason, ...]

    @model_validator(mode='after')
    def abstention(self):
        if not self.candidate_support.start_frame <= self.bottom_frame <= self.candidate_support.end_frame:
            raise ValueError('Anchor must be in the declared candidate support')
        if not self.raw_support.start_frame <= self.bottom_frame <= self.raw_support.end_frame:
            raise ValueError('Raw support must retain the actual anchor')
        if not self.reasons:
            raise ValueError('Unavailable point-turn must retain structured reasons')
        if self.necessary_conditions_met:
            if self.strict_local_minimum is not True or {r.reason for r in self.reasons} != {'noise_not_identifiable'}:
                raise ValueError('Raw conditions alone cannot remove noise abstention')
        elif not any(r.reason != 'noise_not_identifiable' for r in self.reasons):
            raise ValueError('A failed necessary condition needs its actual reason')
        return self


class SquatPointResolutionAudit(ContractModel):
    point_resolution_schema_version: Literal['0.1.0'] = '0.1.0'
    contract_version: Literal['squat-point-resolution-audit-v1'] = 'squat-point-resolution-audit-v1'
    producer_version: Literal['squat-v2-point-resolution-v1'] = 'squat-v2-point-resolution-v1'
    experimental_label: Literal['EXPERIMENTAL / NOT VALIDATED'] = 'EXPERIMENTAL / NOT VALIDATED'
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    policy: PointResolutionPolicy = Field(default_factory=PointResolutionPolicy)
    policy_hash: Hash
    base_result: SquatTemporalResultV2
    base_hash: Hash
    checks: tuple[PointSupportCheck, ...]
    necessary_condition_count: int = Field(ge=0, strict=True)
    accepted_point_count: int = Field(default=0, ge=0, strict=True)
    diagnostics: tuple[PointReason, ...] = ()

    @model_validator(mode='after')
    def linkage(self):
        base = SquatTemporalResultV2.model_validate_json(strict_json(self.base_result), strict=True)
        if (base.producer_version != 'squat-v2-conservative-adapter-v1'
                or self.base_hash != base.content_hash() or self.policy_hash != self.policy.content_hash()):
            raise ValueError('Exact conservative base/policy identity required')
        if self.accepted_point_count != 0:
            raise ValueError('This policy has no independent noise discriminator; no accepted point')
        if len(self.checks) != base.candidate_count or self.necessary_condition_count != sum(c.necessary_conditions_met for c in self.checks):
            raise ValueError('One check per unchanged full PVP candidate; counts must match')
        source = base.legacy.input
        for check, rep in zip(self.checks, base.legacy.repetitions):
            if (check.input_hash != base.legacy.input_hash or check.bottom_frame != rep.bottom_frame
                    or (check.candidate_support.start_frame, check.candidate_support.end_frame) != (rep.start_frame, rep.end_frame)
                    or (check.raw_support.start_frame, check.raw_support.end_frame) != (rep.bottom_frame-1, rep.bottom_frame+1)):
                raise ValueError('Check must link exact PVP/raw neighbor support and input hash')
            if check.raw_support.end_frame >= len(source.samples):
                raise ValueError('Raw support outside input')
            for reason in check.reasons:
                if reason.support and reason.support.end_frame >= len(source.samples):
                    raise ValueError('Reason support outside input')
        if not self.checks and not any(r.reason == 'no_full_pvp_candidate' for r in self.diagnostics):
            raise ValueError('No PVP must remain an explicit unavailable result')
        return self

    def content_hash(self):
        return content_hash(self)


def read_point_resolution_json(payload: str) -> SquatPointResolutionAudit:
    data = json.loads(payload, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {token}')))
    if not isinstance(data, dict) or data.get('point_resolution_schema_version') != '0.1.0':
        raise ValueError('Unsupported point-resolution schema')
    return SquatPointResolutionAudit.model_validate_json(payload, strict=True)


def read_temporal_experiment_json(payload: str):
    """Opt-in dispatch only; no changes to old readers or inferred migrations."""
    data = json.loads(payload)
    if isinstance(data, dict) and 'point_resolution_schema_version' in data:
        return read_point_resolution_json(payload)
    return read_temporal_json(payload)
