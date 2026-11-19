"""Read-only policy checks on real Stage 4 data; no geometry/inference claims."""

import json
from pathlib import Path

import pytest

from gym_coach.features import DEFAULT_REGISTRY, FeatureAvailability, FeatureObservation, check_feature_availability
from gym_coach.flags import FlagCode
from gym_coach.preprocessing import read_preprocessing_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize("exercise", ["BodyWeightSquats", "HandstandPushups"])
def test_registry_policies_on_real_stage4_input_preserve_everything(exercise):
    measured = json.loads((ROOT / "docs/artifacts/stage4_measured_results.json").read_text(encoding="utf-8"))
    video = next(v for v in measured["videos"] if v["exercise"] == exercise)
    artifact = ROOT / video["artifact_relative_path"]
    before_file = artifact.read_bytes()
    result = read_preprocessing_result(artifact)
    before_result = result.model_dump_json()
    # Sample beginning/middle/end to exercise actual raw/processed identity and
    # provenance. This is policy integration, not a new coverage experiment.
    for index in (0, len(result.frames) // 2, len(result.frames) - 1):
        for definition in DEFAULT_REGISTRY.for_exercise(exercise):
            availability = check_feature_availability(definition.feature_id, result.frames[index], result.quality)
            assert availability.status == "unavailable"
            assert {FlagCode.FEATURE_QUALITY_BLOCKED, FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING,
                    FlagCode.VIEWPOINT_UNKNOWN} <= set(availability.reasons)
            assert availability.input.quality == result.quality
            assert availability.input.frame == result.frames[index]
            assert FeatureAvailability.model_validate_json(availability.model_dump_json()) == availability
            observation = FeatureObservation.from_availability(availability)
            assert observation.value is None and observation.status == "unavailable"
    assert artifact.read_bytes() == before_file
    assert result.model_dump_json() == before_result
