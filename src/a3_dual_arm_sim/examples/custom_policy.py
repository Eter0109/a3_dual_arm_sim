"""Minimal user policy loaded with ``--policy module:factory``."""

from a3_dual_arm_sim.policy import SineJointPolicy


def make_policy() -> SineJointPolicy:
    return SineJointPolicy(amplitude_rad=0.08, period_steps=160)

