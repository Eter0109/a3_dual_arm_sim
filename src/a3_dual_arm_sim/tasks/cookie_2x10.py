"""Independent 2x10 target task; the 2x5 environment remains the default."""

from dataclasses import dataclass

import numpy as np

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.sim.randomization import expanded_basic_reset_options
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.tasks.expanded_ik import ExpandedAbsoluteIKMixin
from a3_dual_arm_sim.tasks.source_clearance import SourceColumnClearanceMixin


@dataclass(frozen=True)
class Cookie2x10TaskConfig(CookieTransferTaskConfig):
    required_cookies: int = 20
    require_exact_slots: bool = False
    require_released: bool = True

    def __post_init__(self) -> None:
        if self.required_cookies != 20 or self.cookie_count < 20:
            raise ValueError("the 2x10 target requires 20 cookies and at least 20 source cookies")


class A3Cookie2x10Env(SourceColumnClearanceMixin, A3CookieTransferEnv):
    TASK_CONFIG_CLASS = Cookie2x10TaskConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config or resource_root() / "configs/cookie_2x10.yaml", **kwargs)
        if self.config.cookie_transfer.target_rows != 10:
            raise ValueError("A3Cookie2x10Env requires the separate 2x10 scene configuration")

    def reset(self, *, seed=None, options=None):
        self._begin_source_clearance_sampling()
        observation, info = super().reset(seed=seed, options=options)
        self._record_settled_source_clearances(np.ones(self.task_config.cookie_count, dtype=bool))
        if info["cookies_in_source"] != self.task_config.cookie_count:
            raise RuntimeError("invalid 2x10 initial state: not all cookies are inside the source box")
        return observation, info

    def target_column_counts(self, in_target: tuple[bool, ...]) -> tuple[int, int]:
        columns = np.unique(np.asarray(self.TARGET_SLOTS_LOCAL)[:, 0])
        counts = [0, 0]
        for index, contained in enumerate(in_target):
            if contained:
                local_x = self.privileged_cookie_target_position(index)[0]
                counts[int(np.argmin(np.abs(columns - local_x)))] += 1
        return tuple(counts)

    def _target_fill_is_valid(self, in_target: tuple[bool, ...]) -> bool:
        return self.target_column_counts(in_target) == (10, 10)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["target_layout"] = "2x10"
        info["target_column_counts"] = self.target_column_counts(info["cookies_in_target_mask"])
        return observation, reward, terminated, truncated, info


SOURCE_TASKS = ("2x10-four-source", "2x10-paired-columns")


class SourceQuota2x10Env(ExpandedAbsoluteIKMixin, A3Cookie2x10Env):
    """Score released cookies by their original source and current destination."""

    source_task: str = SOURCE_TASKS[0]

    def _quota(self, mask):
        xs = np.asarray(self.SOURCE_POSITIONS)[:, 0]
        sources = np.searchsorted(np.unique(xs), xs)
        targets = np.unique(np.asarray(self.TARGET_SLOTS_LOCAL)[:, 0])
        counts = np.zeros((4, 2), dtype=int)
        for index in np.flatnonzero(mask):
            x = self.privileged_cookie_target_position(index)[0]
            counts[sources[index], np.argmin(abs(targets - x))] += 1
        if self.source_task == SOURCE_TASKS[0]:
            score = int(np.minimum(counts.sum(axis=1), 5).sum())
        else:
            score = min(10, int(counts[self.first_source, 0]))
            score += min(10, int(counts[(self.first_source + 1) % 4, 1]))
        return counts, score

    def _target_fill_is_valid(self, mask):
        return super()._target_fill_is_valid(mask) and self._quota(mask)[1] == 20

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        counts, score = self._quota(info["cookies_in_target_mask"])
        info["scenario_metrics"] = {"source_by_target_counts": counts.tolist(),
                                    "first_source_column": self.first_source + 1}
        info["cookies_in_target"] = score
        return observation, score / 20, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        task = getattr(self, "source_task", "2x10-source-quota")
        options = expanded_basic_reset_options(options, self.task_config, task)
        self.first_source = int(seed or 0) % 4
        return super().reset(seed=seed, options=options)
