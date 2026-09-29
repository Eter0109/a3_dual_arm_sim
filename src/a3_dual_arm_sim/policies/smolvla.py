from __future__ import annotations

import os
from pathlib import Path

from a3_dual_arm_sim.policies.lerobot import LeRobotPolicyPlugin


class SmolVLAPolicyPlugin(LeRobotPolicyPlugin):
    """Backward-compatible SmolVLA constructor and policy entry point."""

    def __init__(self, checkpoint: Path, dataset_root: Path, repo_id: str, device: str) -> None:
        super().__init__(checkpoint, dataset_root, repo_id, device, policy_type="smolvla")


def make_policy() -> SmolVLAPolicyPlugin:
    """Factory configured through environment variables for ``a3-sim run``."""
    checkpoint = os.environ.get("A3_SMOLVLA_CHECKPOINT")
    dataset_root = os.environ.get("A3_SMOLVLA_DATASET_ROOT")
    if not checkpoint or not dataset_root:
        raise RuntimeError(
            "Set A3_SMOLVLA_CHECKPOINT and A3_SMOLVLA_DATASET_ROOT before loading this policy"
        )
    return SmolVLAPolicyPlugin(
        Path(checkpoint),
        Path(dataset_root),
        os.environ.get("A3_SMOLVLA_REPO_ID", "local/a3-grasp"),
        os.environ.get("A3_SMOLVLA_DEVICE", "cuda"),
    )
