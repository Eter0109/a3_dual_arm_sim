"""Load task choices separately from the scene randomization snapshot."""

from pathlib import Path

import yaml

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_plan import parse_first_grasp

DEFAULT_COOKIE_2X10_SETTINGS = resource_root() / "configs/randomization_2x10.yaml"


def load_cookie_2x10_settings(path=None):
    """Read the independent 2x10 choices once, leaving the legacy loader unchanged."""
    with Path(path or DEFAULT_COOKIE_2X10_SETTINGS).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "first_grasp" not in config:
        raise ValueError("2x10 settings require first_grasp; use configs/randomization_2x10.yaml")
    first_grasp = parse_first_grasp(config["first_grasp"])
    physics = {key: value for key, value in config.items() if key != "first_grasp"}
    settings = load_randomization_config(config=physics)
    return settings, first_grasp
