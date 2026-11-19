"""Pure 2D geometry: standard-library arithmetic and shared structured errors.

No pose/provenance, quality, registry or coordinate-system policy is inferred.
"""

import math
from typing import TypeAlias

from .contracts import ErrorCategory, SourceTier, Stage
from .flags import FlagCode, GymError, Severity, StructuredFlag

Coordinate: TypeAlias = int | float
Point2D: TypeAlias = tuple[Coordinate, Coordinate] | list[Coordinate]
JOINT_ANGLE_PRIMITIVE_VERSION = "joint-angle-2d-v1"

__all__ = ["JOINT_ANGLE_PRIMITIVE_VERSION", "Point2D", "joint_angle"]


def _error(code: FlagCode, message: str, **evidence: str | int) -> GymError:
    return GymError(StructuredFlag(
        code=code,
        category=(ErrorCategory.INPUT_ERROR if code == FlagCode.INVALID_GEOMETRY_INPUT
                  else ErrorCategory.MEASUREMENT_ERROR),
        severity=Severity.ERROR, stage=Stage.FEATURES, source_tier=SourceTier.S3,
        message=message, evidence={"primitive": "joint_angle", **evidence},
    ))


def _point(point: Point2D, label: str) -> tuple[Coordinate, Coordinate]:
    if not isinstance(point, (tuple, list)):
        raise _error(FlagCode.INVALID_GEOMETRY_INPUT, f"Point {label} must be a tuple/list of two coordinates",
                     point=label, reason="point_type")
    if len(point) != 2:
        raise _error(FlagCode.INVALID_GEOMETRY_INPUT, f"Point {label} must have exactly two coordinates",
                     point=label, reason="point_dimension", dimension=len(point))
    checked = []
    for axis, value in zip(("x", "y"), point):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _error(FlagCode.INVALID_GEOMETRY_INPUT, f"Point {label}.{axis} must be an int/float, excluding bool",
                         point=label, coordinate=axis, reason="coordinate_type")
        try:
            coordinate = float(value)
        except OverflowError as exc:
            raise _error(FlagCode.INVALID_GEOMETRY_INPUT, f"Point {label}.{axis} is outside finite float range",
                         point=label, coordinate=axis, reason="coordinate_not_representable") from exc
        if not math.isfinite(coordinate):
            raise _error(FlagCode.INVALID_GEOMETRY_INPUT, f"Point {label}.{axis} must be finite",
                         point=label, coordinate=axis, reason="coordinate_not_finite")
        # Retain integers until subtraction: casting translated large integers
        # first could collapse distinct endpoints into a false zero vector.
        checked.append(value)
    return checked[0], checked[1]


def _direction(endpoint: tuple[Coordinate, Coordinate], vertex: tuple[Coordinate, Coordinate],
               label: str) -> tuple[float, float]:
    dx, dy = endpoint[0] - vertex[0], endpoint[1] - vertex[1]
    if any(isinstance(component, float) and not math.isfinite(component) for component in (dx, dy)):
        # Opposite finite endpoints can overflow subtraction. Halving both
        # components preserves direction and keeps each difference finite.
        dx = endpoint[0] / 2.0 - vertex[0] / 2.0
        dy = endpoint[1] / 2.0 - vertex[1] / 2.0
    magnitude = max(abs(dx), abs(dy))
    if magnitude == 0.0:
        raise _error(FlagCode.DEGENERATE_GEOMETRY, f"Vector {label} has zero length; angle is undefined",
                     vector=label, reason="zero_length_vector")
    if isinstance(magnitude, int):
        try:
            magnitude = float(magnitude)
        except OverflowError:
            # Integer differences can exceed float range even though each
            # coordinate passes input validation. int/int division is safe.
            dx, dy = dx / 2, dy / 2
            magnitude = max(abs(dx), abs(dy))
    # Bound arithmetic without squaring tiny/huge lengths. This is internal
    # numerical rescaling, not a transformation of caller coordinates.
    return dx / magnitude, dy / magnitude


def joint_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Return the internal angle ABC in degrees, in [0, 180], without rounding.

    Points are tuple/list pairs of int/float coordinates representable as finite
    floats; bool, null, strings and non-finite coordinates are invalid inputs.
    All three points are validated before checking geometry. A=B or C=B raises
    GymError(degenerate_geometry); A=C distinct from B is a valid 0-degree angle.
    Arithmetic uses positively rescaled A-B and C-B vectors, their dot/cross
    products and atan2(abs(cross)/denominator, clamped cosine). The denominator
    is checked before division/clamping. No epsilon or image-axis convention,
    implicit input transformation, provenance policy or feature result is used.

    Raises:
        GymError: invalid_geometry_input (input_error), degenerate_geometry or
            geometry_numerical_error (measurement_error); stage features, S3.
    """
    pa, pb, pc = _point(a, "A"), _point(b, "B"), _point(c, "C")
    ux, uy = _direction(pa, pb, "BA")
    vx, vy = _direction(pc, pb, "BC")
    denominator = math.hypot(ux, uy) * math.hypot(vx, vy)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise _error(FlagCode.GEOMETRY_NUMERICAL_ERROR, "Angle denominator must be finite and positive",
                     reason="invalid_denominator")
    cosine = math.fsum((ux * vx, uy * vy)) / denominator
    sine = abs(math.fsum((ux * vy, -uy * vx))) / denominator
    if not (math.isfinite(cosine) and math.isfinite(sine)):
        raise _error(FlagCode.GEOMETRY_NUMERICAL_ERROR, "Angle dot/cross ratios must be finite",
                     reason="non_finite_ratio")
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.atan2(sine, cosine))
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise _error(FlagCode.GEOMETRY_NUMERICAL_ERROR, "Angle must be finite and in [0,180] degrees",
                     reason="invalid_angle_result")
    return angle
