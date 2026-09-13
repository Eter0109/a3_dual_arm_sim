"""One adapter that runs any LeRobot policy against the A3 environment.

LeRobot already provides a single factory for its sixteen policy types
(``make_policy``) plus checkpoint-owned pre/post processing, so the only A3
specific work is projecting an observation onto the keys a checkpoint declares
and adapting the returned action back to the A3 contract. Everything else is
data: which policy class, which weights, which camera names, how often to
re-plan.

See ``docs/policy-interface.md`` for the measurements this design rests on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import checkpoint_embeds_vlm
from .contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    ActionMode,
    ContractError,
    EpisodeContext,
    validate_action,
)

#: Observation keys an A3 environment can supply. A checkpoint may declare a
#: subset; the adapter refuses to run when it declares anything else, because
#: LeRobot silently accepts a missing camera and would degrade the rollout
#: without a warning.
ENVIRONMENT_KEYS = frozenset(
    {
        FRONT_IMAGE,
        LEFT_WRIST_IMAGE,
        RIGHT_WRIST_IMAGE,
        STATE,
        VELOCITY,
        EEF_POSE,
        FORCE,
        "time",
        "safety_stop",
    }
)

#: Feature types LeRobot uses for the action channel.
_ACTION_KEY = "action"


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """How to build a policy. Nothing here is A3-specific."""

    checkpoint: Path
    dataset_root: Path
    repo_id: str
    #: Overrides the checkpoint's own ``type``; only needed for bare weight files.
    policy_type: str | None = None
    #: ``"auto"`` picks CUDA when it is available, otherwise CPU.
    device: str = "auto"
    #: Only meaningful for checkpoints that expose a ``dtype`` (pi0/pi05).
    dtype: str | None = None
    #: Actions executed per plan. Larger is cheaper, smaller is more reactive.
    n_action_steps: int | None = None
    #: Maps an A3 observation key to the name a checkpoint expects.
    rename_map: dict[str, str] = field(default_factory=dict)
    #: ``None`` keeps the checkpoint's own setting; see ``checkpoint_embeds_vlm``.
    load_vlm_weights: bool | None = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def translated_keys(declared: Iterable[str], rename_map: Mapping[str, str]) -> tuple[str, ...]:
    """Map a checkpoint's declared input names back to the A3 keys that feed them."""

    mapped = {value: key for key, value in rename_map.items()}
    return tuple(mapped.get(name, name) for name in declared)


def validate_declared_keys(
    declared: Iterable[str],
    rename_map: Mapping[str, str],
    provided: Iterable[str] = ENVIRONMENT_KEYS,
) -> None:
    """Raise when a checkpoint wants inputs an A3 observation cannot supply.

    This exists because LeRobot accepts a missing camera silently: it returns a
    valid-shaped action and emits no warning, so a mis-wired camera would quietly
    degrade a rollout rather than fail. Checkpoints using another naming scheme
    are fine as long as a ``rename_map`` points an A3 key at each of them.
    """

    available = set(provided)
    mapped = {value: key for key, value in rename_map.items()}
    missing = [name for name in declared if mapped.get(name, name) not in available]
    if missing:
        raise ContractError(
            f"Checkpoint expects {sorted(missing)}, which an A3 observation cannot "
            f"provide. Available keys: {sorted(available)}. Map them with rename_map."
        )


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class LeRobotPolicyAdapter:
    """A3 policy protocol over any LeRobot checkpoint."""

    action_mode: ActionMode = "joint_position"

    def __init__(self, config: PolicyRuntimeConfig) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Install policy support with: pip install -e '.[dataset]'"
            ) from exc

        checkpoint = config.checkpoint.expanduser().resolve()
        dataset_root = config.dataset_root.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint has no config.json: {checkpoint}")
        if not (dataset_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset has no meta/info.json: {dataset_root}")

        self._torch = torch
        self._prepare_observation = prepare_observation_for_inference
        self._device = torch.device(_resolve_device(config.device))
        self._checkpoint = checkpoint

        # The dataset defines the action contract the policy must obey: LeRobot
        # takes output_features from the dataset metadata, not from the config.
        dataset_meta = LeRobotDatasetMetadata(config.repo_id, root=dataset_root)
        action_shape = dataset_meta.features[_ACTION_KEY]["shape"]
        self._action_dim = int(action_shape[0])
        if self._action_dim <= 0:
            raise ContractError(f"Dataset declares a non-positive action width: {action_shape}")

        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        policy_config.pretrained_path = checkpoint
        if config.policy_type is not None:
            policy_config.type = config.policy_type
        policy_config.device = str(self._device)
        policy_config.use_amp = self._device.type == "cuda"
        if config.dtype is not None:
            policy_config.dtype = config.dtype
        if config.n_action_steps is not None:
            policy_config.n_action_steps = config.n_action_steps
        if config.load_vlm_weights is None:
            if checkpoint_embeds_vlm(checkpoint):
                policy_config.load_vlm_weights = False
        else:
            policy_config.load_vlm_weights = config.load_vlm_weights

        self._declared_keys = tuple(policy_config.input_features or ())
        if not self._declared_keys:
            raise ContractError(
                "Checkpoint declares no input features; supply a dataset whose "
                "features match it, or set policy_type."
            )
        self._rename_map = dict(config.rename_map)
        validate_declared_keys(self._declared_keys, self._rename_map)

        self._policy = make_policy(policy_config, ds_meta=dataset_meta)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(checkpoint),
            # Normalisation lives inside the checkpoint, so no dataset stats are
            # needed at inference time.
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": str(self._device)},
                **(
                    {"rename_observations_processor": {"rename_map": self._rename_map}}
                    if self._rename_map
                    else {}
                ),
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._policy.eval()

    # -- construction helpers ------------------------------------------------

    # -- Policy protocol ----------------------------------------------------

    def reset(self, context: EpisodeContext) -> None:
        self._policy.reset()

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        batch = self._prepare_observation(
            {key: observation[key] for key in self._declared_environment_keys()},
            self._device,
            task,
            "A3_dual_arm",
        )
        with self._torch.inference_mode():
            action = self._policy.select_action(self._preprocessor(batch))
            action = self._postprocessor(action)
        return self._adapt(action)

    def close(self) -> None:
        self._policy = None

    # -- translation helpers -------------------------------------------------

    def _declared_environment_keys(self) -> tuple[str, ...]:
        """Declared names translated back to the A3 keys that feed them."""

        return translated_keys(self._declared_keys, self._rename_map)

    def _adapt(self, action: Any) -> np.ndarray:
        """Trim to the A3 action width and validate it against the contract."""

        values = np.asarray(action.detach().float().cpu()).reshape(-1)
        if values.size < self._action_dim:
            raise ContractError(
                f"Policy returned {values.size} action values, fewer than the "
                f"{self._action_dim} the dataset declares."
            )
        # Policies pad to max_action_dim internally; the real width is the
        # dataset's.
        values = values[: self._action_dim]
        return validate_action(values, "joint_position").astype(np.float64)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def declared_keys(self) -> tuple[str, ...]:
        return self._declared_keys

    @property
    def device(self) -> str:
        return str(self._device)


def config_from_environment() -> PolicyRuntimeConfig:
    """Read a runtime config from the ``A3_POLICY_*`` environment variables."""

    checkpoint = os.environ.get("A3_POLICY_CHECKPOINT")
    dataset_root = os.environ.get("A3_POLICY_DATASET_ROOT")
    if not checkpoint or not dataset_root:
        raise RuntimeError(
            "Set A3_POLICY_CHECKPOINT and A3_POLICY_DATASET_ROOT before loading this policy"
        )
    raw_rename = os.environ.get("A3_POLICY_RENAME_MAP", "").strip()
    rename_map = json.loads(raw_rename) if raw_rename else {}
    if not isinstance(rename_map, dict):
        raise TypeError("A3_POLICY_RENAME_MAP must be a JSON object of {a3_key: checkpoint_key}")
    n_action_steps = os.environ.get("A3_POLICY_N_ACTION_STEPS")
    raw_vlm = os.environ.get("A3_POLICY_LOAD_VLM_WEIGHTS")
    return PolicyRuntimeConfig(
        checkpoint=Path(checkpoint),
        dataset_root=Path(dataset_root),
        repo_id=os.environ.get("A3_POLICY_REPO_ID", "local/a3-dual-arm"),
        policy_type=os.environ.get("A3_POLICY_TYPE") or None,
        device=os.environ.get("A3_POLICY_DEVICE", "auto"),
        dtype=os.environ.get("A3_POLICY_DTYPE") or None,
        n_action_steps=int(n_action_steps) if n_action_steps else None,
        rename_map=rename_map,
        load_vlm_weights=None if raw_vlm is None else raw_vlm.strip().lower() in {"1", "true", "yes"},
    )


def make_policy() -> LeRobotPolicyAdapter:
    """CLI factory: ``--policy a3_dual_arm_sim.lerobot_policy:make_policy``.

    Configured through ``A3_POLICY_*`` environment variables so that switching
    models needs no code and no new CLI flags.
    """

    return LeRobotPolicyAdapter(config_from_environment())
