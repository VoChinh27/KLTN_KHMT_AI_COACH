"""Stage 5.2 mathematical primitive tests; no pose/form accuracy claims."""

import math
import subprocess
import sys

import pytest

from gym_coach.contracts import ErrorCategory, SourceTier, Stage
from gym_coach.flags import FlagCode, GymError, Severity, StructuredFlag
from gym_coach.geometry import joint_angle

# Absolute degrees, no relative tolerance and no UI rounding.
ANGLE_TOLERANCE_DEGREES = 1e-10


@pytest.mark.parametrize("a,b,c,expected", [
    ((1, 0), (0, 0), (0, 1), 90.0),
    ((-1, 0), (0, 0), (1, 0), 180.0),
    ((1, 0), (0, 0), (2, 0), 0.0),
    ((1, 0), (0, 0), (0.5, math.sqrt(3) / 2), 60.0),
    ((1, 0), (0, 0), (-1, 1), 135.0),
    ([2, 3], [1, 2], [1, 3], 45.0),
    ((1, 0.1), (0, 0), (-2, -0.2), 180.0),
    ((1, 0.1), (0, 0), (2, 0.2), 0.0),
])
def test_known_angles(a, b, c, expected):
    angle = joint_angle(a, b, c)
    assert isinstance(angle, float)
    assert angle == pytest.approx(expected, abs=ANGLE_TOLERANCE_DEGREES, rel=0)


@pytest.mark.parametrize("a,b,c", [
    ((2, 3), (1, -2), (-3, 4)),
    ((1, 0), (0, 0), (1, 1e-16)),
    ((1e308, 1e308), (-1e308, -1e308), (-1e308, 1e308)),
])
def test_endpoint_symmetry(a, b, c):
    assert joint_angle(a, b, c) == pytest.approx(joint_angle(c, b, a), abs=ANGLE_TOLERANCE_DEGREES, rel=0)


@pytest.mark.parametrize("offset", [(13, -27), (-1000, 2000), (0.25, -0.125)])
def test_translation_invariance(offset):
    points = ((2, 3), (1, -2), (-3, 4))
    translated = tuple((p[0] + offset[0], p[1] + offset[1]) for p in points)
    assert joint_angle(*translated) == pytest.approx(joint_angle(*points), abs=ANGLE_TOLERANCE_DEGREES, rel=0)


@pytest.mark.parametrize("scale", [0.125, 3.0, 1e-300, 1e300])
def test_positive_uniform_scale_invariance(scale):
    points = ((2, 3), (1, -2), (-3, 4))
    scaled = tuple((p[0] * scale, p[1] * scale) for p in points)
    assert joint_angle(*scaled) == pytest.approx(joint_angle(*points), abs=ANGLE_TOLERANCE_DEGREES, rel=0)


def assert_error(caught, code, category):
    error = caught.value
    flag = error.flag
    assert flag.code is code and flag.category is category
    assert flag.stage is Stage.FEATURES and flag.source_tier is SourceTier.S3
    assert flag.severity is Severity.ERROR
    assert flag.message and flag.evidence["primitive"] == "joint_angle" and flag.evidence["reason"]
    assert "value" not in flag.evidence and "angle" not in flag.evidence
    assert not hasattr(error, "value")
    assert StructuredFlag.model_validate_json(flag.model_dump_json()) == flag


@pytest.mark.parametrize("a,b,c,vector", [
    ((1, 2), (1, 2), (3, 4), "BA"),
    ((3, 4), (1, 2), (1, 2), "BC"),
    ((0, 0), (0, 0), (0, 0), "BA"),
])
def test_zero_vector_is_structured_measurement_error(a, b, c, vector):
    with pytest.raises(GymError) as caught:
        joint_angle(a, b, c)
    assert_error(caught, FlagCode.DEGENERATE_GEOMETRY, ErrorCategory.MEASUREMENT_ERROR)
    assert caught.value.flag.evidence["vector"] == vector
    assert caught.value.flag.evidence["reason"] == "zero_length_vector"


@pytest.mark.parametrize("endpoint", [(1, 0), (1, 0.1), (1, 1), (5e-324, 0), (1e308, 1e308)])
def test_coincident_endpoints_distinct_from_vertex_are_exact_zero(endpoint):
    assert joint_angle(endpoint, (0, 0), endpoint) == 0.0


@pytest.mark.parametrize("invalid,reason", [
    (None, "point_type"), ("12", "point_type"), ({"x": 1, "y": 2}, "point_type"),
    (1.0, "point_type"), ({1, 2}, "point_type"),
    ((), "point_dimension"), ((1,), "point_dimension"), ((1, 2, 3), "point_dimension"),
    ((None, 1), "coordinate_type"), ((1, "2"), "coordinate_type"),
    ((True, 1), "coordinate_type"), ((1, False), "coordinate_type"),
    ((1j, 1), "coordinate_type"), (([1], 2), "coordinate_type"),
    ((float("nan"), 1), "coordinate_not_finite"),
    ((1, float("inf")), "coordinate_not_finite"),
    ((float("-inf"), 1), "coordinate_not_finite"),
    ((10**400, 1), "coordinate_not_representable"),
])
@pytest.mark.parametrize("point_index,label", [(0, "A"), (1, "B"), (2, "C")])
def test_invalid_points_have_structured_input_error(invalid, reason, point_index, label):
    points = [(1, 0), (0, 0), (0, 1)]
    points[point_index] = invalid
    with pytest.raises(GymError) as caught:
        joint_angle(*points)
    assert_error(caught, FlagCode.INVALID_GEOMETRY_INPUT, ErrorCategory.INPUT_ERROR)
    assert caught.value.flag.evidence["point"] == label
    assert caught.value.flag.evidence["reason"] == reason


def test_all_input_is_validated_before_degeneracy():
    with pytest.raises(GymError) as caught:
        joint_angle((0, 0), (0, 0), (None, 1))
    assert_error(caught, FlagCode.INVALID_GEOMETRY_INPUT, ErrorCategory.INPUT_ERROR)
    assert caught.value.flag.evidence["point"] == "C"


@pytest.mark.parametrize("a,b,c,expected", [
    ((5e-324, 0), (0, 0), (0, 5e-324), 90.0),
    ((1e-300, 1e-300), (0, 0), (-1e-300, -1e-300), 180.0),
    ((1e308, 1e308), (-1e308, -1e308), (-1e308, 1e308), 45.0),
    ((1, 1e-16), (0, 0), (1, 0), math.degrees(math.atan(1e-16))),
    ((1, 1e-16), (0, 0), (-1, 0), 180.0 - math.degrees(math.atan(1e-16))),
])
def test_tiny_huge_and_nearly_collinear_nonzero_vectors_are_finite(a, b, c, expected):
    angle = joint_angle(a, b, c)
    assert math.isfinite(angle) and 0.0 <= angle <= 180.0
    assert angle == pytest.approx(expected, abs=ANGLE_TOLERANCE_DEGREES, rel=0)
    if expected > 0 and expected < 1e-10:
        assert angle > 0.0  # Near-parallel nonzero angle is not rounded to zero.


def test_lists_are_not_mutated():
    points = [[2, 3], [1, -2], [-3, 4]]
    before = [p.copy() for p in points]
    joint_angle(*points)
    assert points == before


def test_large_integer_translation_does_not_create_false_zero_vectors():
    offset = 10**100
    assert joint_angle((offset, offset), (offset + 1, offset), (offset + 1, offset + 1)) == 90.0


def test_large_integer_differences_do_not_overflow():
    extent = 10**308
    assert joint_angle((extent, extent), (-extent, -extent), (-extent, extent)) == pytest.approx(
        45.0, abs=ANGLE_TOLERANCE_DEGREES, rel=0)


@pytest.mark.parametrize("norm", [0.0, float("nan"), float("inf"), 1e308])
def test_invalid_denominator_stops_before_angle_evaluation(monkeypatch, norm):
    # Fault injection exercises the numerical backstop independently of input
    # validation; finite bounded directions should not normally reach it.
    import gym_coach.geometry as geometry
    monkeypatch.setattr(geometry.math, "hypot", lambda *args: norm)
    def unexpected_angle(*args):
        pytest.fail("invalid denominator reached angle evaluation")
    monkeypatch.setattr(geometry.math, "atan2", unexpected_angle)
    with pytest.raises(GymError) as caught:
        joint_angle((1, 0), (0, 0), (0, 1))
    assert_error(caught, FlagCode.GEOMETRY_NUMERICAL_ERROR, ErrorCategory.MEASUREMENT_ERROR)
    assert caught.value.flag.evidence["reason"] == "invalid_denominator"


def test_public_geometry_import_and_execution_load_no_backends_or_pipeline(tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'cv2', 'mediapipe', 'torch'}:
            raise AssertionError('unexpected backend: ' + fullname)
        if fullname.startswith(('gym_coach.video', 'gym_coach.pose', 'gym_coach.quality',
                                'gym_coach.preprocessing', 'gym_coach.features')):
            raise AssertionError('unexpected pipeline/registry: ' + fullname)
sys.meta_path.insert(0, Block())
from gym_coach.geometry import Point2D, joint_angle
from gym_coach.flags import GymError
assert joint_angle((1,0), (0,0), (0,1)) == 90.0
try:
    joint_angle((0,0), (0,0), (1,0))
except GymError as error:
    assert error.flag.code == 'degenerate_geometry'
    assert error.flag.model_dump_json()
else:
    raise AssertionError('zero vector accepted')
assert not {'numpy', 'scipy', 'cv2', 'mediapipe', 'torch'} & sys.modules.keys()
print('geometry-import-ok')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "geometry-import-ok"
