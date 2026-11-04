"""Plan a small measured jaw release and a geometric detachment height.

Only encoder measurements and externally observed heights enter this module.
It does not assume the jaw's zero encoder value means zero physical gap.
"""
from dataclasses import dataclass
import math

from .batch_vision import LocalizationError


class ReleasePlanningError(LocalizationError):
    """The requested release cannot be supported by the supplied measurements."""


@dataclass(frozen=True)
class NarrowReleasePlan:
    opening: float
    target_gap_m: float
    gap_offset_m: float
    opening_increase: float


def _finite(value, name):
    if isinstance(value, bool):
        raise ReleasePlanningError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReleasePlanningError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ReleasePlanningError(f"{name} must be finite")
    return number


def plan_narrow_release(measured_opening, measured_gap_m, *, gap_scale_m=.085,
                        clearance_m=.002, max_opening=.49):
    """Increase the measured physical jaw gap by exactly ``clearance_m``.

    ``gap_scale_m`` is the public robot model's gap change per normalized
    encoder unit.  The intercept comes from the current measured gap, so thin
    pads and different jaw zero positions do not change the requested motion.
    The cap is a feasibility constraint: an insufficient release raises rather
    than silently clipping a command and pretending the gap was achieved.
    Execution must independently confirm the actual gap and a stationary arm.
    """
    opening = _finite(measured_opening, "measured opening")
    gap = _finite(measured_gap_m, "measured jaw gap")
    scale = _finite(gap_scale_m, "gap scale")
    clearance = _finite(clearance_m, "release clearance")
    limit = _finite(max_opening, "maximum opening")
    if not 0 <= opening <= 1:
        raise ReleasePlanningError("Measured opening is outside [0, 1]")
    if gap < 0:
        raise ReleasePlanningError("Measured jaw gap cannot be negative")
    if scale <= 0 or clearance <= 0:
        raise ReleasePlanningError("Gap scale and release clearance must be positive")
    if not 0 < limit <= 1:
        raise ReleasePlanningError("Maximum opening must be in (0, 1]")
    increase = clearance / scale
    target_opening = opening + increase
    target_gap = gap + clearance
    offset = gap - scale * opening
    if not all(math.isfinite(v) for v in (increase, target_opening, target_gap, offset)):
        raise ReleasePlanningError("Release calculation is not finite")
    # A tiny representation error at an exact feasible boundary is harmless;
    # larger deficits must not be hidden by a saturating command.
    if target_opening > limit + 1e-12:
        raise ReleasePlanningError("Required release exceeds maximum opening")
    target_opening = min(target_opening, limit)
    if target_gap <= gap or target_opening <= opening:
        raise ReleasePlanningError("No positive measurable release can be commanded")
    return NarrowReleasePlan(target_opening, target_gap, offset, target_opening-opening)


def required_detach_height(pad_lowest_z, protected_top_z, *, margin_m=.002,
                           max_detach_m=.13):
    """Return upward travel needed for the entire pad to clear protected tops.

    The caller supplies the lowest point of both physical pads from robot FK
    and the highest currently supported object/rim evidence.  This calculation
    does not grant permission to open the jaws: measured final clearance must
    still be checked after the commanded retreat, including tracking error.
    """
    lowest = _finite(pad_lowest_z, "lowest pad height")
    highest = _finite(protected_top_z, "protected top height")
    margin = _finite(margin_m, "detachment margin")
    limit = _finite(max_detach_m, "maximum detachment travel")
    if margin < 0 or limit <= 0:
        raise ReleasePlanningError("Detachment margin must be nonnegative and travel positive")
    travel = max(0., highest + margin - lowest)
    if not math.isfinite(travel) or travel > limit + 1e-12:
        raise ReleasePlanningError("Required detachment exceeds available upward travel")
    return min(travel, limit)
