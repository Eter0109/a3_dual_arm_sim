from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.contracts import ActionMode, EpisodeContext
from a3_dual_arm_sim.paths import project_root

_RUNTIME = project_root() / ".runtime"
os.environ.setdefault("HF_HOME", str(_RUNTIME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_RUNTIME / "datasets"))


DEPLOYMENT_HOME_RIGHT = np.array(
    [-1.294467, -0.986556, 1.192891, 1.005229, 0.336042, -0.367167, 1.506819, 1.0],
    dtype=np.float64,
)


def validate_execution_horizon(n_action_steps: int, chunk_size: int) -> None:
    if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int):
        raise ValueError("n_action_steps must be an integer")
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError("n_action_steps must be between 1 and chunk_size")


class SmolVLAPolicyPlugin:
    """LeRobot SmolVLA checkpoint adapter for the common A3 policy protocol."""

    action_mode: ActionMode = "joint_position"

    def __init__(
        self,
        checkpoint: Path,
        dataset_root: Path,
        repo_id: str,
        device: str,
        *,
        num_steps: int | None = 25,
        ema_alpha: float = 0.0,
        anchor_right_arm: bool = False,
        gripper_sharpening: bool = False,
        align_vertical: bool = False,
        clamp_z: bool = False,
        n_action_steps: int | None = None,
        inference_seed: int | None = None,
    ) -> None:
        try:
            import mujoco
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            raise RuntimeError("Install SmolVLA support with: pip install -e '.[train]'") from exc

        checkpoint = checkpoint.expanduser().resolve()
        dataset_root = dataset_root.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing SmolVLA config.json: {checkpoint}")
        self._mujoco = mujoco
        self._torch = torch
        self._prepare_observation = prepare_observation_for_inference
        self._device = torch.device(device)
        self.ema_alpha = ema_alpha
        self.anchor_right_arm = anchor_right_arm
        self.gripper_sharpening = gripper_sharpening
        self.align_vertical = align_vertical
        self.clamp_z = clamp_z
        self.env: Any | None = None
        rot_canonical = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
        )
        self._target_quat_canonical = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quat_canonical, rot_canonical.reshape(-1))

        self._prev_action: np.ndarray | None = None
        if not 0 <= ema_alpha <= 1:
            raise ValueError("ema_alpha must be in [0, 1]")
        self.inference_seed = inference_seed
        self.last_raw_action: np.ndarray | None = None

        try:
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                download_videos=False,
                return_uint8=True,
            )
        except TypeError:
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                download_videos=False,
            )
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = device
        config.use_amp = device == "cuda"
        self._action_dim = config.output_features["action"].shape[0]
        if self._action_dim not in (8, 16) or list(
            config.input_features["observation.state"].shape
        ) != [self._action_dim]:
            raise ValueError("Expected matching 8-D left-only or 16-D dual-arm state/actions")
        self._right_hold = None
        if n_action_steps is not None:
            validate_execution_horizon(n_action_steps, config.chunk_size)
            config.n_action_steps = n_action_steps
        if num_steps is not None and hasattr(config, "num_steps"):
            config.num_steps = num_steps
        self._input_keys = tuple(config.input_features)
        self._policy = make_policy(config, ds_meta=dataset.meta)
        if num_steps is not None and hasattr(self._policy.config, "num_steps"):
            self._policy.config.num_steps = num_steps
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._policy.eval()

    def bind_env(self, env: Any) -> None:
        self.env = env

    def reset(self, context: EpisodeContext) -> None:
        if self.inference_seed is not None:
            self._torch.manual_seed(self.inference_seed)
        self._policy.reset()
        self._prev_action = None
        self.last_raw_action = None
        self._right_hold = None

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        policy_observation = {key: observation[key] for key in self._input_keys}
        if getattr(self, "_action_dim", 16) == 8:
            state = np.asarray(observation["observation.state"])
            if state.shape != (16,):
                raise ValueError("Left-only deployment requires the environment's 16-D state")
            if self._right_hold is None:
                self._right_hold = state[8:16].copy()
            policy_observation["observation.state"] = state[:8].copy()
        batch = self._prepare_observation(policy_observation, self._device, task, "A3_dual_arm")
        batch = self._preprocessor(batch)
        with self._torch.inference_mode():
            action = self._policy.select_action(batch)
            action = self._postprocessor(action)
        action_np = action.detach().float().cpu().numpy().reshape(getattr(self, "_action_dim", 16))
        if getattr(self, "_action_dim", 16) == 8:
            action_np = np.concatenate((action_np, self._right_hold))
        self.last_raw_action = action_np.copy()

        # 1. Stationary Right Arm Joint Anchoring (Pillar 3)
        if self.anchor_right_arm and getattr(self, "_action_dim", 16) == 16:
            action_np[8:16] = DEPLOYMENT_HOME_RIGHT

        # 2. EEF Coordinate & Posture Correction (FK/IK)
        if (
            (self.align_vertical or self.clamp_z)
            and self.env is not None
            and hasattr(self.env, "solve_ik")
        ):
            try:
                from a3_dual_arm_sim.contracts import LEFT_JOINTS

                work = self.env._ik._work
                left_qpos_indices = [self.env._qpos_ids[name] for name in LEFT_JOINTS]
                work.qpos[:] = self.env.data.qpos
                work.qpos[left_qpos_indices] = action_np[:7]
                self._mujoco.mj_kinematics(self.env.model, work)
                site_id = self.env._eef_sites[0]
                pos = work.site_xpos[site_id].copy()

                if self.clamp_z:
                    # Floor clearance above source box floor (z=0.753) while staying below the
                    # expert grasp depth (z=0.7638, constant during CLOSE per expert traces).
                    pos[2] = max(pos[2], 0.760)
                    # Closed fingers are also needed while descending to place.
                    # A transport-height floor here would prevent placement.

                target_quat = self._target_quat_canonical.copy()
                if not self.align_vertical:
                    self._mujoco.mju_mat2Quat(target_quat, work.site_xmat[site_id])
                changed_height = abs(pos[2] - work.site_xpos[site_id, 2]) > 1e-9
                if self.align_vertical or changed_height:
                    q_corrected = self.env.solve_ik(
                        pos, target_quat, initial_q=action_np[:7], arm="L"
                    )
                    if np.all(np.isfinite(q_corrected)):
                        action_np[:7] = q_corrected
            except Exception:
                pass

        # 3. Gripper Sharpening & Hysteresis (Pillar 3)
        if self.gripper_sharpening:
            if action_np[7] < 0.45:
                action_np[7] = 0.2804
            elif action_np[7] > 0.60:
                action_np[7] = 1.0

        # 4. Action EMA Smoothing for left arm joints 0..6 (Pillar 3)
        if self.ema_alpha > 0 and self._prev_action is not None:
            action_np[:7] = (
                self.ema_alpha * action_np[:7] + (1.0 - self.ema_alpha) * self._prev_action[:7]
            )

        self._prev_action = action_np.copy()
        return action_np

    def close(self) -> None:
        self._policy = None


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
