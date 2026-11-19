# Squat temporal v2 — frozen slice contract

2026-10-18. `squat-temporal-components-v2`, schema `0.2.0`, S3,
**EXPERIMENTAL / NOT VALIDATED**. Contract/acceptance freeze precedes producer.
This is an additive synthetic-only slice of
[the proposal](SQUAT_TEMPORAL_CONTRACT_PROPOSAL.md), not completion of G6.

## Frozen semantics

- Events and motion intervals are separate records. Evidence spans are inclusive;
  motion/pause intervals are half-open `[)`. Event times are exact input sample
  times, never interpolated. Event IDs are deterministic within input/run/side.
- A complete candidate has top-start S, bottom anchor B, top-end E. A shared top
  is one event referenced as previous end and next start; phase at that frame
  belongs to the next cycle. A final E has an event but no continuing phase.
- `dwell_candidate`: S < arrival A < departure D < E, A <= B <= D.
  Descent[S,A), Bottom[A,D), Ascent[D,E); pause duration D-A, never +1/FPS.
- `point_turn_at_resolution`: qualified point evidence is required separately
  from PVP. Descent[S,B), Ascent[B,E), event at B; arrival/departure and pause
  remain null/unavailable with `pause_unresolved_at_resolution`, not zero.
- `bounds_unresolved`: retain supported anchor/candidate; arrival/departure,
  pause and phase may all abstain. Never default to point-turn from legacy PVP.
- Component states: candidate/unavailable/not_run/blocked. A non-candidate
  component has null value and at least one structured reason (category,
  segmentation stage, S3, inclusive support location). No fabricated numbers.
- Full-cycle count is candidate count only. Gap and explicit side change split
  runs including at shared tops. No imputation, side choice or bridging. Partial
  edges are diagnostics; an unfinished hold never creates an end/departure.
- Per-component contract evidence cites method + inclusive support + input hash;
  validation checks observed support for bounds/point evidence. Hashes establish
  linkage, not scientific accuracy or authenticity of an annotation.

## Conservative producer capability frozen for this slice

`segment_squat_v2(SquatTemporalInput)` accepts ONLY the existing explicit
synthetic capability, copies/revalidates it, invokes unchanged v1 PVP with the
unchanged DEFAULT `squat-temporal-s3-v1` profile, and links its exact result/hash.
No custom-profile argument, new detector, threshold, angle calculation or
fixture-reference input. It emits supported top and bottom-anchor events,
cycle references and run diagnostics; **every bottom mode is bounds_unresolved**.
It does not claim to detect dwell or point-turn. These modes are supported by
the schema/phase projection and hand-authored contract acceptance cases ONLY.
Their reference event records are never supplied to the producer or demo.

Reason: the proposal explicitly leaves bounds/resolution evidence rules open.
Choosing flatness epsilon or a reversal rule now would add an unreviewed detector.
In clip-ending-hold fixtures this slice abstains even on arrival, preserves
partial-end evidence and does not fabricate a complete candidate. This is a
capability limit, not evidence that no pause occurred.

## Acceptance frozen before producer (tests/test_temporal_v2.py)

1. Dwell record: exact half-open A/D ownership, D-A arithmetic, independent anchor.
2. Point record: B belongs to Ascent + bottom event; bounds/pause null, no 0s.
3. Unresolved record and actual producer: count/anchor retained, phase unavailable.
4. Shared top: same event ID/two roles, one next-cycle phase, no overlap.
5. Gap/side switch: no candidate/run/event support crosses discontinuity.
6. Clip ends in pause: no fake end, departure, duration or complete cycle.
7. Status/value/reason invariants; invalid ordering, identity, hash, closure,
   provenance and nonfinite values rejected. Independent component abstention.
8. Explicit 0.1.0/v1 vs 0.2.0/v2 dispatch; unknown versions rejected, no migration.
9. V1 byte/canonical output compatibility; profile hash fixed; light imports.
10. Producer never consumes fixture references, does not read Data/ or import
    geometry/pose/video backends; no source mutation. JSON readback preserves all
    statuses/reasons/provenance. Renderer shows unavailable (not 0s), separates
    events/motion and escapes embedded text. Offline synthetic demo runs once
    into an exclusive directory, refuses overwrites.

Positive dwell/point tests check contract behavior, **not detector accuracy**.
New fixture cases are not the 30-condition sensitivity matrix; no tuning or
before/after accuracy claim. Floating time arithmetic tolerance is 1e-9 seconds
for CFR checks; event frame identity is exact. No accuracy tolerance is inferred.

## Compatibility and evidence

V1 files/public exports/schema/API/profile are unchanged. Explicit new modules
are opt-in; no dependency or package version change. A new result carries full
legacy input/result/hash and policy identity/hash. NOT_VALIDATED,
official_eligibility=false, assessment.enabled=false, G5.7 blocked and G6
IN_PROGRESS remain required. Freeze receipt includes schema/tests/document/policy
hashes, pre-producer absence and expected red test result in the new run.
No real data, assessment, score, technical advice or athlete validation.
