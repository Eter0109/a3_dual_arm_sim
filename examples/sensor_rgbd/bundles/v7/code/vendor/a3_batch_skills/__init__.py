"""Repeatable five-cookie transfers on the current A3 environment.

The default reference implementation uses explicitly privileged localization.
Learned policies remain left-arm 8D; environment commands retain both arms.
"""

from .contracts import BatchConfig, BatchGoal, Phase

__all__ = ["BatchConfig", "BatchGoal", "Phase"]
