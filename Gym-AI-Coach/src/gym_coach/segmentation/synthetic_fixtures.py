"""Analytic signal fixtures and independent phase references, never pose data.

Ground truth describes this generator only. No geometry, QualityGate, inference,
interpolation or athlete annotation is involved in constructing a fixture.
"""
from dataclasses import dataclass
import math
from typing import Literal

from pydantic import Field, model_validator

from ..contracts import ContractModel, JointProvenance
from ..flags import FlagCode
from .models import SquatSignalSample, SquatTemporalInput, content_hash, strict_json


class CycleSpec(ContractModel):
    descent_seconds: float = Field(gt=0, strict=True)
    bottom_pause_seconds: float = Field(default=0., ge=0, strict=True)
    ascent_seconds: float = Field(gt=0, strict=True)
    excursion_degrees: float = Field(default=60., gt=0, le=170, strict=True)


class MissingInterval(ContractModel):
    start_seconds: float = Field(ge=0, strict=True)
    end_seconds: float = Field(gt=0, strict=True)

    @model_validator(mode='after')
    def ordered(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError('Missing interval must have positive duration')
        return self


class SyntheticCase(ContractModel):
    fixture_version: Literal['squat-analytic-fixture-v1'] = 'squat-analytic-fixture-v1'
    ground_truth_scope: Literal['analytic_synthetic_signal_only'] = 'analytic_synthetic_signal_only'
    case_id: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    description: str = Field(min_length=1)
    cycles: tuple[CycleSpec, ...] = Field(min_length=1, max_length=32)
    leading_guard_seconds: float = Field(default=1., ge=0, strict=True)
    trailing_guard_seconds: float = Field(default=1., ge=0, strict=True)
    missing_intervals: tuple[MissingInterval, ...] = ()
    side_switch_seconds: float | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode='after')
    def bounds(self):
        end = self.leading_guard_seconds + self.trailing_guard_seconds + sum(
            c.descent_seconds + c.bottom_pause_seconds + c.ascent_seconds for c in self.cycles)
        if end > 120:
            raise ValueError('Fixture duration exceeds this bounded synthetic audit')
        if any(interval.end_seconds > end for interval in self.missing_intervals):
            raise ValueError('Missing interval outside fixture')
        if self.side_switch_seconds is not None and self.side_switch_seconds >= end:
            raise ValueError('Side switch outside fixture')
        return self


class ReferenceCycle(ContractModel):
    reference_index: int = Field(ge=1, strict=True)
    start_seconds: float
    bottom_arrival_seconds: float
    bottom_departure_seconds: float
    end_seconds: float
    start_frame: int
    bottom_midpoint_frame: int
    end_frame: int
    boundary_partial: bool

    @property
    def bottom_midpoint_seconds(self):
        return (self.bottom_arrival_seconds + self.bottom_departure_seconds) / 2


@dataclass(frozen=True)
class SampledFixture:
    definition: SyntheticCase
    signal: SquatTemporalInput
    references: tuple[ReferenceCycle, ...]
    reference_phases: tuple[str, ...]
    full_reference_mask: tuple[bool, ...]

    def payload(self):
        return {
            'definition': self.definition.model_dump(mode='json'),
            'definition_hash': content_hash(self.definition),
            'input': self.signal.model_dump(mode='json'),
            'input_hash': self.signal.content_hash(),
            'references': [r.model_dump(mode='json') for r in self.references],
            'reference_phases': list(self.reference_phases),
            'full_reference_mask': list(self.full_reference_mask),
        }


def nearest_frame(seconds: float, fps: float) -> int:
    """Nearest CFR sample, ties up; epsilon only stabilizes decimal arithmetic."""
    return math.floor(seconds * fps + .5 + 1e-9)


def _reference_phase(index, time, references):
    # Independent of phase_at/_turns. Shared top belongs to the earlier Return.
    for ref in references:
        if index == ref.start_frame:
            return 'Start/Top'
        if index == ref.end_frame:
            return 'Return/Top'
        if ref.start_frame < index < ref.end_frame:
            if ref.bottom_arrival_seconds == ref.bottom_departure_seconds:
                if index == ref.bottom_midpoint_frame:
                    return 'Bottom'
            elif ref.bottom_arrival_seconds - 1e-9 <= time <= ref.bottom_departure_seconds + 1e-9:
                return 'Bottom'
            return 'Descent' if time < ref.bottom_arrival_seconds else 'Ascent'
    return 'OutsideReference'


def build_fixture(definition: SyntheticCase, fps: float) -> SampledFixture:
    definition = SyntheticCase.model_validate_json(strict_json(definition), strict=True)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError('Fixture FPS must be finite within [1,240]')
    fps = float(fps)
    timing = []
    cursor = definition.leading_guard_seconds
    for cycle in definition.cycles:
        arrival = cursor + cycle.descent_seconds
        departure = arrival + cycle.bottom_pause_seconds
        end = departure + cycle.ascent_seconds
        timing.append((cursor, arrival, departure, end, cycle.excursion_degrees))
        cursor = end
    count = nearest_frame(cursor + definition.trailing_guard_seconds, fps) + 1
    references = tuple(ReferenceCycle(reference_index=i + 1,
        start_seconds=a, bottom_arrival_seconds=b, bottom_departure_seconds=c, end_seconds=d,
        start_frame=nearest_frame(a, fps), bottom_midpoint_frame=nearest_frame((b + c) / 2, fps),
        end_frame=nearest_frame(d, fps), boundary_partial=(nearest_frame(a, fps) == 0 or nearest_frame(d, fps) == count - 1))
        for i, (a, b, c, d, _) in enumerate(timing))

    def angle(t):
        if t < timing[0][0]:
            # Ascending guard confirms the first interior top; it is not a rep.
            return 140. + 30. * (1. - math.cos(math.pi * t / timing[0][0])) / 2
        for start, arrival, departure, end, excursion in timing:
            bottom = 170. - excursion
            if t <= arrival:
                return bottom + excursion * (1. + math.cos(math.pi * (t - start) / (arrival - start))) / 2
            if t <= departure:
                return bottom
            if t <= end:
                return bottom + excursion * (1. - math.cos(math.pi * (t - departure) / (end - departure))) / 2
        if definition.trailing_guard_seconds == 0:
            return 170.
        u = min(1., (t - cursor) / definition.trailing_guard_seconds)
        return 170. - 30. * (1. - math.cos(math.pi * u)) / 2

    samples = []
    for index in range(count):
        t = index / fps
        missing = any(g.start_seconds - 1e-9 <= t <= g.end_seconds + 1e-9 for g in definition.missing_intervals)
        side = 'right' if definition.side_switch_seconds is not None and t >= definition.side_switch_seconds - 1e-9 else 'left'
        samples.append(SquatSignalSample(frame_index=index, timestamp_seconds=t,
            value=None if missing else angle(t), side_source=side,
            joint_provenance=(JointProvenance.MISSING if missing else JointProvenance.OBSERVED,) * 3,
            reasons=(FlagCode.MISSING_JOINT,) if missing else ()))
    signal = SquatTemporalInput(video_id=f'synthetic-{definition.case_id}-{fps:g}fps', fps=fps,
        source_hash=content_hash(definition), samples=tuple(samples))
    phases = tuple(_reference_phase(s.frame_index, s.timestamp_seconds, references) for s in samples)
    mask = tuple(any(not r.boundary_partial and r.start_frame <= s.frame_index <= r.end_frame for r in references)
                 for s in samples)
    return SampledFixture(definition, signal, references, phases, mask)


def canonical_cases() -> tuple[SyntheticCase, ...]:
    """Fixed stress matrix, not selected by favorable segmentation outcomes."""
    steady = CycleSpec(descent_seconds=1., ascent_seconds=1.)
    return (
        SyntheticCase(case_id='steady_three', description='Three equal cycles with confirming guards', cycles=(steady,) * 3),
        SyntheticCase(case_id='bottom_pauses', description='Bottom holds 0.24/0.6/1.2 seconds',
            cycles=tuple(CycleSpec(descent_seconds=1., bottom_pause_seconds=p, ascent_seconds=1.) for p in (.24, .6, 1.2))),
        SyntheticCase(case_id='asymmetric_pauses', description='Bottom holds and unequal descending/ascending speeds',
            cycles=tuple(CycleSpec(descent_seconds=d, bottom_pause_seconds=p, ascent_seconds=a)
                         for d, p, a in ((.6, 1.2, 1.4), (1.4, .8, .4), (.4, .4, 1.8)))),
        SyntheticCase(case_id='variable_tempo', description='Slow/fast/slow/fast successive cycles',
            cycles=tuple(CycleSpec(descent_seconds=d, ascent_seconds=a) for d, a in ((1.4, .6), (.4, .4), (1.8, 1.2), (.24, .36)))),
        SyntheticCase(case_id='eight_consecutive', description='Eight consecutive cycles with shared top events',
            cycles=(CycleSpec(descent_seconds=.6, ascent_seconds=.8),) * 8),
        SyntheticCase(case_id='fast_cycles', description='0.08-second halves stress the fixed temporal spacing',
            cycles=(CycleSpec(descent_seconds=.08, ascent_seconds=.08),) * 3),
        SyntheticCase(case_id='small_excursion', description='Three-degree excursions stress the fixed hysteresis',
            cycles=(CycleSpec(descent_seconds=1., ascent_seconds=1., excursion_degrees=3.),) * 3),
        SyntheticCase(case_id='missing_bottom', description='Unresolved gap crosses the first bottom', cycles=(steady,) * 3,
            missing_intervals=(MissingInterval(start_seconds=1.9, end_seconds=2.1),)),
        SyntheticCase(case_id='explicit_side_switch', description='Declared side change within second cycle; no fallback',
            cycles=(steady,) * 3, side_switch_seconds=4.),
        SyntheticCase(case_id='partial_edges', description='First/last cycles touch clip edges and remain partial',
            cycles=(steady,) * 3, leading_guard_seconds=0., trailing_guard_seconds=0.),
    )
