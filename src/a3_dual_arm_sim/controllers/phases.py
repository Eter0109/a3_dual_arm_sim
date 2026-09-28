from __future__ import annotations

from enum import Enum, auto


class GraspPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    HOLD = auto()
    DONE = auto()


class CookiePhase(Enum):
    SUPPORT_BOX = auto()
    SELECT_COOKIE = auto()
    APPROACH = auto()
    PRE_CLOSE = auto()
    ALIGN = auto()
    DESCEND = auto()
    CLOSE = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    VERIFY_LIFT = auto()
    MOVE_TO_SLOT = auto()
    DESCEND_TO_PLACE = auto()
    OPEN = auto()
    VERIFY_RELEASE = auto()
    RETRACT = auto()
    DONE = auto()
    FAILED = auto()
