"""Versioned foundation settings; no algorithm thresholds or backend activation."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, ValidationError

from ._version import CONFIG_SCHEMA_VERSION
from .contracts import ContractModel, ErrorCategory, Exercise, SourceTier, Stage, Viewpoint
from .flags import FlagCode, GymError, Severity, StructuredFlag


class ResolvedPaths(ContractModel):
    input_dir: Path
    output_dir: Path


class PathsConfig(ContractModel):
    input_dir: Path = Path("Data")
    output_dir: Path = Path("outputs")

    def resolve(self, base_dir: Path) -> ResolvedPaths:
        """Resolve against an explicit base, without creating or touching files."""
        return ResolvedPaths(
            input_dir=(base_dir / self.input_dir).resolve(),
            output_dir=(base_dir / self.output_dir).resolve(),
        )


class AssessmentConfig(ContractModel):
    enabled: StrictBool = False
    status: Literal["NOT_AVAILABLE_PENDING_EXPERT_VALIDATION"] = (
        "NOT_AVAILABLE_PENDING_EXPERT_VALIDATION"
    )

    def model_post_init(self, context: object) -> None:
        if self.enabled:
            raise ValueError("Assessment requires expert validation; unavailable in phase 0")


class PipelineConfig(ContractModel):
    config_schema_version: Literal["0.1.0"] = CONFIG_SCHEMA_VERSION
    # Identifies parameter/profile changes independently from schema changes.
    config_version: str = Field(default="foundation-v1", min_length=1)
    exercise: Exercise = Exercise.BODY_WEIGHT_SQUATS
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    paths: PathsConfig = Field(default_factory=PathsConfig)
    assessment: AssessmentConfig = Field(default_factory=AssessmentConfig)

    def content_hash(self) -> str:
        """SHA-256 of canonical validated config, including unresolved portable paths."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> PipelineConfig:
    """Read explicit TOML; malformed, unsupported or unreadable config is structured."""
    try:
        with Path(path).open("rb") as stream:
            return PipelineConfig.model_validate(tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise GymError(StructuredFlag(
            code=FlagCode.INVALID_CONFIG,
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=Severity.ERROR,
            stage=Stage.CONFIGURATION,
            message=str(exc),
            source_tier=SourceTier.S3,
            evidence={"config_path": str(path)},
        )) from exc
