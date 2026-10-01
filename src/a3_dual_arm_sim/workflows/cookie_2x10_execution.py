"""Episode runner that reuses the recording contract with a separate 20-cookie task."""

from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env, Cookie2x10TaskConfig
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan, parse_first_grasp
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.workflows.episode_execution import EpisodeExecution
from a3_dual_arm_sim.workflows.policy_adapters import ExpertPolicyAdapter


class Cookie2x10EpisodeExecution(EpisodeExecution):
    required_cookies = 20

    def __init__(self, config_path=None, *, first_grasp=None, **kwargs):
        settings_path = kwargs.pop("randomization_config", None)
        if kwargs.get("randomization_settings") is None:
            settings, configured_grasp = load_cookie_2x10_settings(settings_path)
            kwargs["randomization_settings"] = settings
        else:
            if settings_path is not None:
                raise ValueError("provide either randomization_config or randomization_settings")
            settings = kwargs["randomization_settings"]
            configured_grasp = "random"
        self.first_grasp = parse_first_grasp(
            configured_grasp if first_grasp is None else first_grasp
        )
        kwargs.setdefault("profile", settings["profile"])
        kwargs.setdefault("max_steps", 3200)
        if kwargs.get("source_column") is None:
            kwargs["source_column"] = settings["source_column"]
        super().__init__(config_path or resource_root() / "configs/cookie_2x10.yaml", **kwargs)
        if self.config.cookie_transfer.target_rows != 10:
            raise ValueError("the 2x10 runner requires configs/cookie_2x10.yaml")

    def create_env(self, *, render_cameras=False):
        return A3Cookie2x10Env(
            self.config,
            task_config=Cookie2x10TaskConfig(
                cookie_count=len(self.config.cookie_transfer.cookie_source_positions_m),
                position_noise_m=self.cookie_noise_m,
                yaw_noise_rad=self.cookie_yaw_noise_rad,
                terminate_on_success=False,
            ),
            render_mode="human" if self.render else None,
            render_cameras=render_cameras,
        )

    def _episode_task(self, column, seed):
        return Cookie2x10Plan.from_seed(self.first_grasp, seed).prompt(column or 1)

    def _episode_policy(self, policy, column, seed):
        if isinstance(policy, str) and policy == "variable_grasp":
            return ExpertPolicyAdapter(
                lambda env: A3Cookie2x10Expert(
                    env,
                    first_grasp=self.first_grasp,
                    sampling_seed=seed,
                    requested_source_column_index=(column or 1) - 1,
                ),
                name="cookie_2x10_variable_grasp",
            )
        if isinstance(policy, str) and policy in {"same_column", "cross_column"}:
            raise ValueError("use 'variable_grasp' for the separate 2x10 expert")
        return super()._episode_policy(policy, column, seed)

    def _episode_metadata(self, column, seed):
        return {
            **super()._episode_metadata(column, seed),
            **Cookie2x10Plan.from_seed(self.first_grasp, seed).metadata(),
            "task_instruction": self._episode_task(column, seed),
        }

    def _completed_episode_metadata(self, runner_policy, info):
        expert = getattr(runner_policy, "expert", None)
        return {
            "target_column_counts": list(info.get("target_column_counts", (0, 0))),
            "cookies_in_target": int(info.get("cookies_in_target", 0)),
            "cookies_in_source": int(info.get("cookies_in_source", 0)),
            "grasp_reports": getattr(expert, "batch_reports", []),
        }
