"""Clip-local validated snapshots; no replacement for a serialized linked context.

Only immutable bytes/tuples are retained. Returned Pydantic models and audit
dicts are fresh, detached values; frozen models alone are not an isolation
boundary. The complete Stage4 audit runs once at intake, including unsafe
model_construct/model_copy inputs. No caller-supplied validation token exists.
"""
from dataclasses import dataclass
from pydantic import ValidationError

from ..flags import FlagCode
from ..pose.models import PoseFrame
from ..preprocessing.models import PreprocessingResult, content_hash as source_hash
from .context import (
    FEATURE_CONTEXT_FACTORY_VERSION, FEATURE_CONTEXT_SCHEMA_VERSION,
    _failure, _strict_json, _validation_failure,
)
from .models import FeatureInputContext, content_hash

FEATURE_CONTEXT_SESSION_VERSION = "validated-preprocessing-session-v1"


@dataclass(frozen=True, slots=True)
class _FrameSelection:
    """Internal immutable handle, not an external schema or validation credential."""
    _input_json: bytes
    _raw_json: bytes
    _audit: tuple[tuple[str, str | int | float], ...]

    def input_context(self) -> FeatureInputContext:
        """Each call returns a new ordinary context; adapters still validate it."""
        return FeatureInputContext.model_validate_json(self._input_json, strict=True)

    @property
    def raw_frame(self) -> PoseFrame:
        return PoseFrame.model_validate_json(self._raw_json, strict=True)

    @property
    def measurement_input_hash(self) -> str:
        return dict(self._audit)["measurement_input_hash"]

    def reference_linkage(self) -> dict:
        """Exact historical identity projection, not a LinkedFeatureInputContext.

        Version fields identify the reference linkage format. Actual execution
        via a session is recorded separately, never in numerical observations.
        This projection has no full source and cannot be reloaded as a trusted
        linked context or a session. External Stage4 JSON requires full intake.
        """
        return dict(self._audit)


class FeatureContextSession:
    """Own one strictly audited clip snapshot; select detached frame inputs.

    Caller mutation after construction cannot affect this session. A context
    obtained from a selection may be edited by its owner, but remains an ordinary
    FeatureInputContext and has no authority to alter the snapshot or its source
    receipt. Request a fresh input/selection to recover the original data.
    Sessions are process-local, non-picklable and have no global identity cache.
    Deliberate private-memory modification/monkeypatching is not a Python security
    boundary; neither legacy hashes nor this session authenticate a producer.
    """
    __slots__ = ("__frames", "__raw", "__quality", "__audit", "__hashes", "__size")

    def __init__(self, result: PreprocessingResult):
        if not isinstance(result, PreprocessingResult):
            raise _failure(FlagCode.INVALID_FEATURE_INPUT, "preprocessing_result_type",
                           "Expected a complete PreprocessingResult")
        try:
            encoded = _strict_json(result)
        except (TypeError, ValueError) as exc:
            raise _failure(FlagCode.INVALID_FEATURE_INPUT, "preprocessing_serialization_failed",
                           str(exc)) from exc
        self.__capture(self.__validate(encoded))

    @staticmethod
    def __validate(encoded: str | bytes) -> PreprocessingResult:
        try:
            return PreprocessingResult.model_validate_json(encoded, strict=True)
        except ValidationError as exc:
            raise _validation_failure(exc, "session_source_validation_failed", None) from exc
        except OverflowError as exc:
            raise _failure(FlagCode.INVALID_FEATURE_INPUT, "session_source_validation_failed",
                           str(exc)) from exc

    @classmethod
    def from_json(cls, encoded: str | bytes) -> "FeatureContextSession":
        """External input always undergoes the complete strict Stage4 audit."""
        if not isinstance(encoded, (str, bytes)):
            raise _failure(FlagCode.INVALID_FEATURE_INPUT, "session_json_type",
                           "Expected JSON str or bytes, not a cache/token")
        checked = cls.__validate(encoded)
        session = object.__new__(cls)
        session.__capture(checked)
        return session

    def __capture(self, checked: PreprocessingResult) -> None:
        # No model/dict references survive this method, including nested flag
        # evidence, QualityReport dictionaries or the caller's profile.
        frames = tuple(f.model_dump_json().encode("utf-8") for f in checked.frames)
        raw = tuple(f.model_dump_json().encode("utf-8") for f in checked.raw_frames)
        quality = checked.quality.model_dump_json().encode("utf-8")
        audit = tuple(dict(
            context_schema_version=FEATURE_CONTEXT_SCHEMA_VERSION,
            context_factory_version=FEATURE_CONTEXT_FACTORY_VERSION,
            preprocessing_result_hash=source_hash(checked),
            raw_sequence_hash=checked.raw_sequence_hash,
            quality_hash=checked.quality_hash, profile_hash=checked.profile_hash,
            profile_version=checked.profile.profile_version,
            metadata_hash=source_hash(checked.metadata),
        ).items())
        hashes = tuple((f.raw_frame_hash, source_hash(f)) for f in checked.frames)
        for name, value in (("frames", frames), ("raw", raw), ("quality", quality),
                            ("audit", audit), ("hashes", hashes),
                            ("size", sum(map(len, frames)) + sum(map(len, raw)) + len(quality))):
            object.__setattr__(self, "_FeatureContextSession__" + name, value)

    def __setattr__(self, name, value):
        raise AttributeError("FeatureContextSession is immutable; create a new session")

    def __reduce_ex__(self, protocol):
        raise TypeError("Create a new validated session in each worker; sessions are not serialized")

    def __len__(self):
        return len(self.__frames)

    @property
    def snapshot_size_bytes(self) -> int:
        """Encoded payload bytes only, not Python object overhead or peak RSS."""
        return self.__size

    @property
    def metadata(self) -> dict:
        return dict(session_version=FEATURE_CONTEXT_SESSION_VERSION,
                    preprocessing_result_hash=dict(self.__audit)["preprocessing_result_hash"],
                    frame_count=len(self), snapshot_payload_bytes=self.__size,
                    source_audit_scope="complete Stage4 strict validation once at intake",
                    reference_linkage_is_projection=True)

    def select(self, *, frame_index: int) -> _FrameSelection:
        if type(frame_index) is not int or frame_index < 0:
            raise _failure(FlagCode.INVALID_FEATURE_CONTEXT_INDEX, "invalid_frame_index",
                           "frame_index must be a non-negative integer (bool is not accepted)")
        if frame_index >= len(self):
            raise _failure(FlagCode.INVALID_FEATURE_CONTEXT_INDEX, "frame_index_out_of_range",
                           f"frame_index {frame_index} exceeds sequence length {len(self)}",
                           index=frame_index)
        payload = b'{"frame":' + self.__frames[frame_index] + b',"quality":' + self.__quality + b'}'
        # Frame/report validation and hash are still per selection. No full
        # PreprocessingResult or LinkedFeatureInputContext is constructed here.
        context = FeatureInputContext.model_validate_json(payload, strict=True)
        audit = dict(self.__audit)
        audit.update(raw_frame_hash=self.__hashes[frame_index][0],
                     processed_frame_hash=self.__hashes[frame_index][1],
                     frame_index=context.frame.frame_index,
                     timestamp_seconds=context.frame.timestamp_seconds,
                     measurement_input_hash=content_hash(context))
        return _FrameSelection(payload, self.__raw[frame_index], tuple(audit.items()))
