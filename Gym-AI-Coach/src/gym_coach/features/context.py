"""Versioned, self-contained linkage to Stage 4; no measurements or new gates."""

import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, model_validator

from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..pose.models import PoseFrame
from ..preprocessing.models import PreprocessingResult, content_hash as preprocessing_hash
from .models import FeatureInputContext, Hash, Text, content_hash

FEATURE_CONTEXT_SCHEMA_VERSION = "0.2.0"
FEATURE_CONTEXT_FACTORY_VERSION = "preprocessing-context-v1"


def _strict_json(value: FeatureInputContext | PreprocessingResult) -> str:
    # JSON-mode serialization can hide corrupt input: NaN becomes null and a
    # boolean in a float field becomes 1.0. Python mode preserves these types;
    # only the declared Path type needs an explicit JSON representation.
    def encode_path(item: object) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported serialized type: {type(item).__name__}")

    return json.dumps(value.model_dump(mode="python", warnings="error"),
                      default=encode_path, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _audit_fields(result: PreprocessingResult, index: int) -> dict[str, str]:
    return {
        "preprocessing_result_hash": preprocessing_hash(result),
        "raw_sequence_hash": result.raw_sequence_hash,
        "raw_frame_hash": result.frames[index].raw_frame_hash,
        "processed_frame_hash": preprocessing_hash(result.frames[index]),
        "quality_hash": result.quality_hash,
        "profile_hash": result.profile_hash,
        "profile_version": result.profile.profile_version,
        "metadata_hash": preprocessing_hash(result.metadata),
    }


class LinkedFeatureInputContext(FeatureInputContext):
    """A detached snapshot linked to one complete, validated PreprocessingResult.

    Legacy FeatureInputContext and numeric observation schemas stay unchanged.
    Persist this context alongside the numeric observation to retain linkage.
    Hashes detect inconsistent content; they do not authenticate its producer.
    """

    context_schema_version: Literal["0.2.0"] = FEATURE_CONTEXT_SCHEMA_VERSION
    context_factory_version: Literal["preprocessing-context-v1"] = FEATURE_CONTEXT_FACTORY_VERSION
    preprocessing_result: PreprocessingResult
    preprocessing_result_hash: Hash
    raw_sequence_hash: Hash
    raw_frame_hash: Hash
    processed_frame_hash: Hash
    quality_hash: Hash
    profile_hash: Hash
    profile_version: Text
    metadata_hash: Hash

    @model_validator(mode="after")
    def source_linkage(self) -> "LinkedFeatureInputContext":
        # Nested model instances can bypass Pydantic validation when constructed
        # or copied unsafely. Validate serialized content, including the full
        # raw timeline, endpoints, runs and Stage 4's core quality consistency.
        checked = PreprocessingResult.model_validate_json(
            _strict_json(self.preprocessing_result), strict=True)
        index = self.frame.frame_index
        if type(index) is not int or not 0 <= index < len(checked.frames):
            raise ValueError("selected frame index is outside the preprocessing result")
        if self.frame != checked.frames[index]:
            raise ValueError("context frame differs from the selected processed frame")
        if self.quality != checked.quality:
            raise ValueError("context QualityReport differs from the preprocessing snapshot")
        for name, expected in _audit_fields(checked, index).items():
            if getattr(self, name) != expected:
                raise ValueError(f"context {name} differs from the preprocessing identity")
        return self

    @property
    def raw_frame(self) -> PoseFrame:
        return self.preprocessing_result.raw_frames[self.frame.frame_index]

    @property
    def measurement_input_hash(self) -> str:
        """Reference to the unchanged frame/quality input in numeric schema 0.2.0."""
        return content_hash(FeatureInputContext(frame=self.frame, quality=self.quality))

    def content_hash(self) -> str:
        return content_hash(self)


def _failure(code: FlagCode, reason: str, message: str, *, index: int | None = None) -> GymError:
    return GymError(StructuredFlag(
        code=code, category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
        stage=Stage.FEATURES, source_tier=SourceTier.S3, message=message,
        location=FlagLocation(frame_start=index, frame_end=index),
        evidence={"component": "feature_input_context_factory", "reason": reason},
    ))


def _validation_failure(exc: ValidationError, reason: str, index: int | None) -> GymError:
    # Schema/type failures are input errors. An after-validator failure means
    # schema-shaped data contradict its audit/identity relationships.
    mismatch = any(error["type"] == "value_error" for error in exc.errors())
    code = FlagCode.FEATURE_CONTEXT_MISMATCH if mismatch else FlagCode.INVALID_FEATURE_INPUT
    return _failure(code, reason, str(exc), index=index)


def build_feature_input_context(preprocessing_result: PreprocessingResult, *,
                                frame_index: int) -> LinkedFeatureInputContext:
    """Select an exact Stage 4 frame/report snapshot, rejecting invalid inputs.

    No quality decision is replaced, promoted or inferred. Full Stage 4
    revalidation uses its existing core-count audit; no preprocessing rerun or
    geometry calculation occurs. Empty results have no selectable frame.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise _failure(FlagCode.INVALID_FEATURE_CONTEXT_INDEX, "invalid_frame_index",
                       "frame_index must be a non-negative integer (bool is not accepted)")
    if not isinstance(preprocessing_result, PreprocessingResult):
        raise _failure(FlagCode.INVALID_FEATURE_INPUT, "preprocessing_result_type",
                       "Expected a complete PreprocessingResult", index=frame_index)
    try:
        encoded = _strict_json(preprocessing_result)
    except (ValueError, TypeError) as exc:
        raise _failure(FlagCode.INVALID_FEATURE_INPUT, "preprocessing_serialization_failed",
                       str(exc), index=frame_index) from exc
    try:
        checked = PreprocessingResult.model_validate_json(encoded, strict=True)
    except ValidationError as exc:
        raise _validation_failure(exc, "preprocessing_validation_failed", frame_index) from exc
    except OverflowError as exc:
        raise _failure(FlagCode.INVALID_FEATURE_INPUT, "preprocessing_validation_failed",
                       str(exc), index=frame_index) from exc
    if frame_index >= len(checked.frames):
        raise _failure(FlagCode.INVALID_FEATURE_CONTEXT_INDEX, "frame_index_out_of_range",
                       f"frame_index {frame_index} exceeds sequence length {len(checked.frames)}",
                       index=frame_index)
    try:
        return LinkedFeatureInputContext(
            frame=checked.frames[frame_index], quality=checked.quality,
            preprocessing_result=checked, **_audit_fields(checked, frame_index))
    except ValidationError as exc:
        raise _validation_failure(exc, "context_linkage_validation_failed", frame_index) from exc


def _validated_linked_context(context: LinkedFeatureInputContext) -> LinkedFeatureInputContext:
    """Adapter boundary: detect post-factory corruption and detach caller state."""
    try:
        encoded = _strict_json(context)
    except (ValueError, TypeError) as exc:
        raise _failure(FlagCode.INVALID_FEATURE_INPUT, "context_serialization_failed", str(exc)) from exc
    try:
        return LinkedFeatureInputContext.model_validate_json(encoded, strict=True)
    except ValidationError as exc:
        raise _validation_failure(exc, "context_linkage_validation_failed", None) from exc
    except OverflowError as exc:
        raise _failure(FlagCode.INVALID_FEATURE_INPUT, "context_linkage_validation_failed", str(exc)) from exc
