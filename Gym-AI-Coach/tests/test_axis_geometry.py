import math
import sys
import subprocess

import pytest

from gym_coach.axis_geometry import midpoint_axis_angle_vertical as angle
from gym_coach.contracts import ErrorCategory, Stage
from gym_coach.flags import GymError


@pytest.mark.parametrize("dx,dy,expected", [(0,-1,0), (1,0,90), (0,1,180), (1,-1,45), (-1,-1,45)])
def test_known_directed_angles(dx, dy, expected):
    assert angle((-1,0), (1,0), (dx-1,dy), (dx+1,dy)) == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_positive_scale_and_bilateral_symmetry(scale):
    p = [(-scale,0), (scale,0), (0,-scale), (2*scale,-scale)]
    assert angle(*p) == pytest.approx(45, abs=1e-10, rel=0)
    assert angle(p[1],p[0],p[3],p[2]) == angle(*p)


def test_translation_and_large_integer_precision():
    t = 10**100
    assert angle((t-1,t),(t+1,t),(t,t-1),(t+2,t-1)) == pytest.approx(45, abs=1e-10, rel=0)


@pytest.mark.parametrize("p", [5e-324, 1.7976931348623157e308])
def test_extreme_finite_midpoints_do_not_overflow_or_underflow(p):
    result = angle((-p,p),(p,p),(-p,-p),(p,-p))
    assert math.isfinite(result) and result == 0


def test_zero_midpoint_axis_is_measurement_error():
    with pytest.raises(GymError) as caught:
        angle((-1,0),(1,0),(0,-1),(0,1))
    flag = caught.value.flag
    assert flag.category == ErrorCategory.MEASUREMENT_ERROR and flag.stage == Stage.FEATURES
    assert flag.source_tier == "S3" and flag.code == "degenerate_geometry"
    assert flag.evidence == {"primitive":"midpoint_axis_angle_vertical", "reason":"zero_length_vector", "vector":"hip_to_shoulder"}


@pytest.mark.parametrize("bad", [None, (), (1,), (1,2,3), (None,0), (True,0), ("1",0), (float('nan'),0), (float('inf'),0), (-float('inf'),0)])
def test_invalid_input_is_not_measurement(bad):
    with pytest.raises(GymError) as caught:
        angle((0,0),(0,0),(0,0),bad)
    assert caught.value.flag.category == ErrorCategory.INPUT_ERROR
    assert caught.value.flag.evidence["primitive"] == "midpoint_axis_angle_vertical"


def test_axis_import_is_independent_of_pipeline(tmp_path):
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'} or name.startswith(('gym_coach.features','gym_coach.pose','gym_coach.preprocessing','gym_coach.quality','gym_coach.video')):
   raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.axis_geometry import midpoint_axis_angle_vertical
assert midpoint_axis_angle_vertical((-1,0),(1,0),(-1,-1),(1,-1)) == 0
"""
    proc = subprocess.run([sys.executable,"-I","-c",code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
