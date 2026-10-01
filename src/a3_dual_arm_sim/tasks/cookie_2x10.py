"""Independent 2x10 target task; the 2x5 environment remains the default."""

from dataclasses import dataclass

import numpy as np

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig


@dataclass(frozen=True)
class Cookie2x10TaskConfig(CookieTransferTaskConfig):
    required_cookies: int = 20
    require_exact_slots: bool = False
    require_released: bool = True

    def __post_init__(self) -> None:
        if self.required_cookies != 20 or self.cookie_count < 20:
            raise ValueError("the 2x10 target requires 20 cookies and at least 20 source cookies")


class A3Cookie2x10Env(A3CookieTransferEnv):
    TASK_CONFIG_CLASS = Cookie2x10TaskConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config or resource_root() / "configs/cookie_2x10.yaml", **kwargs)
        if self.config.cookie_transfer.target_rows != 10:
            raise ValueError("A3Cookie2x10Env requires the separate 2x10 scene configuration")

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
