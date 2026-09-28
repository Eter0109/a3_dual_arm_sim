"""Benchmark suite for single-box batch cookie transfer.

Provides a standardized 20-episode benchmark evaluating policies or expert models
on the 10-cookie transfer task under slight randomization of source/target boxes
and cookies. Scoring is based on the number of cookies settled in the target box.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from a3_dual_arm_sim.contracts import ActionMode, EpisodeContext
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

DEFAULT_CONFIG_PATH = project_root() / "configs" / "cookie_batch.yaml"


@runtime_checkable
class BenchmarkPolicy(Protocol):
    """Protocol for any policy evaluated in the benchmark.

    Compatible with neural network policies (e.g. SmolVLA, OpenVLA, ACT, Diffusion Policy),
    RL agents, or rule-based expert controllers.
    """

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        """Produce an action given the current observation."""
        ...

    def reset(self, context: EpisodeContext | None = None) -> None:
        """Reset the policy state at the beginning of an episode."""
        ...


class ExpertPolicyAdapter:
    """Adapts an expert model to the standard BenchmarkPolicy interface."""

    def __init__(self, expert_factory: Callable[[A3CookieTransferEnv], Any], name: str = "expert"):
        self.expert_factory = expert_factory
        self.name = name
        self.expert: Any | None = None
        self.action_mode: ActionMode = "cartesian_delta"

    def bind_env(self, env: A3CookieTransferEnv) -> None:
        self.expert = self.expert_factory(env)
        self.expert.reset()

    def reset(self, context: EpisodeContext | None = None) -> None:
        if self.expert is not None:
            self.expert.reset(context)

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        if self.expert is None:
            raise RuntimeError("Expert policy must be bound to an environment before acting")
        return self.expert.act(observation, task)

    @property
    def finished(self) -> bool:
        return bool(getattr(self.expert, "finished", False))

    @property
    def phase_name(self) -> str:
        phase = getattr(self.expert, "phase", None)
        return phase.name if phase is not None else ""


def resolve_smolvla_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if (path / "checkpoints").is_dir():
        numbered = [p for p in (path / "checkpoints").iterdir() if p.name.isdigit()]
        if numbered:
            path = max(numbered, key=lambda p: int(p.name))
        else:
            path = path / "checkpoints" / "last"
        ema = path / "pretrained_model_ema"
        path = ema if (ema / "config.json").is_file() else path / "pretrained_model"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint config: {path}")
    return path.resolve()


class SmolVLAPolicyAdapter:
    """Adapts a trained SmolVLA checkpoint to the standard BenchmarkPolicy interface."""

    def __init__(
        self,
        checkpoint: str | Path,
        dataset_root: str | Path = Path("datasets/a3_front_close_left_100"),
        repo_id: str = "local/a3-front-close-left-100",
        device: str | None = None,
        name: str = "smolvla",
        **kwargs: Any,
    ):
        import torch

        from a3_dual_arm_sim.policies.smolvla import SmolVLAPolicyPlugin

        checkpoint_path = resolve_smolvla_checkpoint(checkpoint)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.plugin = SmolVLAPolicyPlugin(
            checkpoint_path, Path(dataset_root), repo_id, dev, **kwargs
        )
        self.name = name
        self.action_mode: ActionMode = "joint_position"

    def reset(self, context: EpisodeContext | None = None) -> None:
        self.plugin.reset(
            context
            or EpisodeContext(
                seed=0, task="transfer 10 cookies into target box", action_mode="joint_position"
            )
        )

    def act(
        self, observation: dict[str, Any], task: str = "transfer 10 cookies into target box"
    ) -> np.ndarray:
        return self.plugin.act(observation, task)

    def bind_env(self, env: Any) -> None:
        if hasattr(self.plugin, "bind_env"):
            self.plugin.bind_env(env)

    def close(self) -> None:
        self.plugin.close()


def make_policy_adapter(policy_or_spec: Any, env: A3CookieTransferEnv | None = None) -> Any:
    """Create a policy runner from a string name, spec, or object."""
    if isinstance(policy_or_spec, str):
        spec = policy_or_spec.strip().lower()
        if spec in ("same_column", "same-column", "same"):
            adapter = ExpertPolicyAdapter(A3SameColumnBatchExpert, name="same_column")
            if env is not None:
                adapter.bind_env(env)
            return adapter
        if spec in ("cross_column", "cross-column", "cross", "batch"):
            adapter = ExpertPolicyAdapter(A3CookieBatchExpert, name="cross_column")
            if env is not None:
                adapter.bind_env(env)
            return adapter
        if spec == "smolvla" or spec.startswith("smolvla:") or Path(policy_or_spec).exists():
            ckpt = Path("outputs/smolvla_front_close_left_20k")
            if spec.startswith("smolvla:"):
                ckpt = Path(policy_or_spec.split(":", 1)[1])
            elif Path(policy_or_spec).exists():
                ckpt = Path(policy_or_spec)
            return SmolVLAPolicyAdapter(ckpt)
        # Handle module:factory policy spec
        if ":" in policy_or_spec:
            mod_name, func_name = policy_or_spec.rsplit(":", 1)
            mod = importlib.import_module(mod_name)
            factory = getattr(mod, func_name)
            policy = factory()
            return policy
        raise ValueError(f"Unknown policy specifier: {policy_or_spec}")

    # Callable or class
    if isinstance(policy_or_spec, type):
        if issubclass(policy_or_spec, (A3CookieBatchExpert, A3SameColumnBatchExpert)):
            adapter = ExpertPolicyAdapter(policy_or_spec, name=policy_or_spec.__name__)
            if env is not None:
                adapter.bind_env(env)
            return adapter
        return policy_or_spec()

    # Already an expert instance
    if isinstance(policy_or_spec, (A3CookieBatchExpert, A3SameColumnBatchExpert)):
        adapter = ExpertPolicyAdapter(lambda e: policy_or_spec, name=type(policy_or_spec).__name__)
        adapter.expert = policy_or_spec
        return adapter

    return policy_or_spec
