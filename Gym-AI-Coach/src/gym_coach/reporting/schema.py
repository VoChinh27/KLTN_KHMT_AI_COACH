"""Foundation audit header; full domain report will grow with tested stages."""

from typing import Literal

from pydantic import Field

from .._version import SCHEMA_VERSION, __version__
from ..contracts import ContractModel, Exercise, Stage, StageStatus, Viewpoint
from ..flags import StructuredFlag


class AuditHeader(ContractModel):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    package_version: str = __version__
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    exercise: Exercise
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    stage_status: dict[Stage, StageStatus] = Field(default_factory=lambda: {
        stage: StageStatus.NOT_RUN for stage in Stage
    })
    flags: tuple[StructuredFlag, ...] = ()
