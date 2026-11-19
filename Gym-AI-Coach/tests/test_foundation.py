"""Phase 0 smoke and safety contracts. No real-video/algorithm validation claimed."""

import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import __version__
from gym_coach.config import PipelineConfig, load_config
from gym_coach.contracts import ErrorCategory, Exercise, SourceTier, Stage, StageStatus, Viewpoint
from gym_coach.flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from gym_coach.reporting import AuditHeader

ROOT = Path(__file__).resolve().parents[1]


def test_installed_package_and_all_stage_imports():
    assert importlib.metadata.version("ai-coach-gym") == __version__
    for module in ("video", "pose", "quality", "preprocessing", "features",
                   "segmentation", "assessment", "reporting"):
        importlib.import_module(f"gym_coach.{module}")


def test_import_from_other_directory_without_algorithm_backends(tmp_path):
    code = """
import importlib, importlib.abc, sys
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mediapipe', 'cv2', 'numpy', 'scipy', 'torch'}:
            raise AssertionError('unexpected backend: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
for name in ('config', 'contracts', 'flags', 'video', 'pose', 'quality',
             'preprocessing', 'features', 'segmentation', 'assessment', 'reporting'):
    importlib.import_module('gym_coach.' + name)
print('foundation-import-ok')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "foundation-import-ok"


def test_config_portability_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = ROOT / "configs/default.toml"
    config = load_config(path)
    assert config.exercise is Exercise.BODY_WEIGHT_SQUATS
    assert config.viewpoint is Viewpoint.UNKNOWN
    assert config.assessment.enabled is False
    assert config.paths.resolve(path.parent).input_dir == ROOT / "Data"
    assert config.paths.resolve(path.parent).output_dir == ROOT / "outputs"
    assert not (tmp_path / "outputs").exists()
    assert PipelineConfig(exercise="HandstandPushups").exercise is Exercise.HANDSTAND_PUSHUPS


def test_config_hash_roundtrip_and_changes():
    original = PipelineConfig()
    restored = PipelineConfig.model_validate_json(original.model_dump_json())
    assert original.content_hash() == restored.content_hash()
    assert original.content_hash() != PipelineConfig(exercise="HandstandPushups").content_hash()
    assert original.content_hash() != PipelineConfig(config_version="foundation-v2").content_hash()


@pytest.mark.parametrize("content", [
    'config_schema_version = "99.0.0"',
    'exercise = "BenchPress"',
    'viewpoint = "invented"',
    'unknown_setting = 1',
    '[assessment]\nenabled = true',
    '[assessment]\nenabled = "false"',
    '[assessment]\nscore = 100',
    'invalid toml !!!',
])
def test_invalid_config_is_structured(tmp_path, content):
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GymError) as caught:
        load_config(path)
    flag = caught.value.flag
    assert flag.code is FlagCode.INVALID_CONFIG
    assert flag.category is ErrorCategory.CONFIGURATION_ERROR
    assert flag.stage is Stage.CONFIGURATION
    assert flag.evidence["config_path"] == str(path)


def test_missing_config_is_structured(tmp_path):
    with pytest.raises(GymError) as caught:
        load_config(tmp_path / "missing.toml")
    assert caught.value.flag.code is FlagCode.INVALID_CONFIG


def test_audit_header_and_flags_roundtrip():
    config = PipelineConfig()
    flag = StructuredFlag(
        code=FlagCode.VIEWPOINT_UNKNOWN, category=ErrorCategory.MEASUREMENT_ERROR,
        severity=Severity.WARNING, stage=Stage.QUALITY, source_tier=SourceTier.S3,
        message="Viewpoint metadata unavailable; dependent metrics unavailable",
        location=FlagLocation(video_id="synthetic", frame_start=0, frame_end=2),
        evidence={"viewpoint": "unknown"},
    )
    header = AuditHeader(
        config_version=config.config_version, config_hash=config.content_hash(),
        run_id="smoke", video_id="synthetic", exercise=config.exercise, flags=(flag,),
    )
    encoded = header.model_dump_json()
    assert AuditHeader.model_validate_json(encoded) == header
    assert json.loads(encoded)["flags"][0]["code"] == "viewpoint_unknown"
    assert all(status is StageStatus.NOT_RUN for status in header.stage_status.values())
    assert "score" not in json.loads(encoded)
    with pytest.raises(ValidationError):
        AuditHeader.model_validate({**header.model_dump(), "schema_version": "9.0.0"})


@pytest.mark.parametrize("location", [
    {"frame_start": -1, "frame_end": 2},
    {"frame_start": 3, "frame_end": 2},
    {"frame_start": 0},
    {"frame_start": True, "frame_end": 2},
    {"rep_index": 0},
])
def test_invalid_flag_boundaries_rejected(location):
    with pytest.raises(ValidationError):
        FlagLocation(**location)
