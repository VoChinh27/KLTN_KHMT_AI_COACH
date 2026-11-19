"""Strict, self-contained JSON artifacts preserving raw data and quality."""

from pathlib import Path

from ..flags import FlagCode
from .models import PreprocessingResult
from .pipeline import failure


def write_preprocessing_result(result: PreprocessingResult, path: str | Path) -> Path:
    target = Path(path)
    try:
        checked = PreprocessingResult.model_validate_json(result.model_dump_json())
        with target.open("x", encoding="utf-8") as stream:
            stream.write(checked.model_dump_json(indent=2))
            stream.write("\n")
        return target
    except (OSError, ValueError, TypeError) as exc:
        raise failure(FlagCode.PREPROCESSING_SERIALIZATION_ERROR, str(exc)) from exc


def read_preprocessing_result(path: str | Path) -> PreprocessingResult:
    try:
        return PreprocessingResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise failure(FlagCode.PREPROCESSING_SERIALIZATION_ERROR, str(exc)) from exc
