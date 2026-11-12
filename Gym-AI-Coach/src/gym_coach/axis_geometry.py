"""Directed midpoint-axis angle in the existing image coordinate convention."""

from fractions import Fraction

from . import geometry
from .flags import FlagCode, GymError

AXIS_ANGLE_PRIMITIVE_VERSION = "midpoint-axis-angle-vertical-2d-v1"


def axis_vector(points):
    """Exact difference of midpoint sums; inputs must already be validated.

    Fractions preserve finite float/int inputs without midpoint overflow,
    subnormal underflow or cancellation of distinct large integers.
    """
    lh, rh, ls, rs = points
    return tuple((Fraction(ls[i]) + Fraction(rs[i]) - Fraction(lh[i]) - Fraction(rh[i])) / 2
                 for i in (0, 1))


def midpoint_axis_angle_vertical(left_hip: geometry.Point2D, right_hip: geometry.Point2D,
                                 left_shoulder: geometry.Point2D, right_shoulder: geometry.Point2D) -> float:
    """Angle of hip-midpoint -> shoulder-midpoint against image_up=(0,-1).

    Degrees in [0,180], directed axis: an inverted vertical body yields 180.
    No rotation, flip, calibration or modification of caller coordinates.
    """
    try:
        points = tuple(geometry._point(p, label) for p, label in zip(
            (left_hip, right_hip, left_shoulder, right_shoulder),
            ("left_hip", "right_hip", "left_shoulder", "right_shoulder")))
        dx, dy = axis_vector(points)
        magnitude = max(abs(dx), abs(dy))
        if magnitude == 0:
            raise geometry._error(FlagCode.DEGENERATE_GEOMETRY,
                                  "Hip-to-shoulder midpoint vector has zero length",
                                  reason="zero_length_vector", vector="hip_to_shoulder")
        # Internal positive arithmetic rescaling, not coordinate normalization.
        return geometry.joint_angle((float(dx / magnitude), float(dy / magnitude)),
                                    (0.0, 0.0), (0.0, -1.0))
    except GymError as exc:
        raise GymError(exc.flag.model_copy(update={"evidence": {
            **exc.flag.evidence, "primitive": "midpoint_axis_angle_vertical"}})) from exc


__all__ = ["AXIS_ANGLE_PRIMITIVE_VERSION", "midpoint_axis_angle_vertical"]
