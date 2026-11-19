# Frozen point-resolution evidence rule — v1

2026-10-18. Policy/producer `squat-v2-point-resolution-v1`, S3,
**EXPERIMENTAL / NOT VALIDATED**. Opt-in audit envelope, not an amendment of
[the frozen v2 contract](SQUAT_TEMPORAL_V2_FROZEN_CONTRACT.md).

## Rule frozen before producer

For each unchanged default-S3 full PVP candidate [S,E] with bottom anchor B:

1. Require raw samples B-1, B, B+1 with consecutive frame identities and valid
   original CFR timestamps. B and the candidate must not touch a clip boundary.
2. Require the support and entire candidate to have finite available values,
   observed joint provenance, and one explicitly supplied run/side. Missing,
   low-visibility, imputed or side-switch support is never repaired or fused.
3. Require raw[B] < raw[B-1] AND raw[B] < raw[B+1], with exact strict comparisons.
   No smoothing, epsilon, added threshold or anchor relocation for this check.
   The PVP anchors themselves remain outputs of the unchanged S3/v1 algorithm.
4. Reject any UNSTABLE_SIGNAL flag whose inclusive location intersects the raw
   support/candidate; an unlocated unstable flag is conservatively global.
5. These are **necessary, not sufficient** conditions. A sample-resolution
   reversal candidate must additionally have evidence that distinguishes the
   raw minimum from measurement noise. `observed` denotes provenance/availability,
   not noise-free accuracy. This input contract supplies no independent noise
   bound/model/repeat measurement or other discriminator. Therefore this policy
   **abstains even when 1–4 pass**, reason `noise_not_identifiable`.

This is a deliberate, testable negative capability. Do not relabel a strict
three-sample pattern as an accepted point-turn. No safe accepted point-turn is
available under this policy/input capability. There is no flag/argument to
declare synthetic ground truth noise-free or bypass step5. Fixture labels,
references and generation formulas are never producer inputs.

## Identifiability counterexample (acceptance evidence)

The same input samples around a smooth minimum are consistent with (a) a smooth
latent reversal with zero noise, or (b) a latent hold across those samples plus
additive measurement errors exactly equal to observed minus the held value.
Both histories yield identical full input bytes/hash, observed provenance,
timestamps, PVP and three-sample inequalities. No deterministic decision on this
input can accept one and reject the other. Contextual smoothness or exact
symmetry does not remove this ambiguity without an independent noise assumption.
The two latent histories exist only in the test proof; neither reaches producer.

No numerical noise bound, amplitude/speed threshold or noise distribution is
invented here. The conclusion is limited to the supplied capability and the
required abstention against indistinguishable noise, not impossibility of all
future temporal analysis. A strict minimum remains a computable raw-sample fact.

## Output and compatibility

New envelope `point_resolution_schema_version=0.1.0`, contract
`squat-point-resolution-audit-v1`, policy/producer identity above. It links the
entire byte-equivalent conservative v2 base result/hash plus per-candidate raw
checks, exact supports, necessary-condition status and structured reasons.
`accepted_point_count=0`; point event/arrival/departure/pause duration null;
the base stays `bounds_unresolved` with unavailable motion.

Reasons retain segmentation/S3/insufficient_data and inclusive support. Original
v1 flags are preserved. Input/timeline failures retain structured input errors.
Hash establishes linkage, not authenticity or signal accuracy. A new opt-in
reader dispatches this envelope vs old segmentation0.1/0.2; old readers, models,
schemas, producer outputs, APIs and renderers remain unchanged.

New standalone demo displays raw strict-minimum support separately from an
accepted point-turn and lists abstention reasons. It may link to a separately
rendered unchanged v2 base page. No positive point-turn will be fabricated just
to populate the demo. Synthetic-only; no angles computed, no Data, inference,
real integration, dependencies, score or assessment. G5.7 blocked, G6 IN_PROGRESS,
official_eligibility=false and assessment.enabled=false.

## Acceptance freeze

Freeze models, this document, fixtures and tests before producer/rule/demo files
exist. Preserve schema/tests/document hashes and the expected-red test receipt.
Tests cover smooth minima at25/30/60FPS, flat plateau, jitter, identical-observation
counterexample, missing neighbor, candidate gap/imputation/side switch, overlapping
and unlocated UNSTABLE_SIGNAL, clip edges, bad timestamps, JSON/hash/version
dispatch, byte-compatible v1/conservative v2, heavy-import/geometry/Data guards,
label independence and exclusive demo output. None uses fixture references as
producer input. No30-condition sensitivity run or threshold fitting.

Completion of this milestone means the rule has been evaluated and its safe
abstention implemented/tested; it does not mean a usable point-turn detector has
been validated. A future accepting policy needs independent noise evidence and
new frozen acceptance, not a relaxed condition in this policy.
