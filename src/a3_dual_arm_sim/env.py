from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .actions import CartesianDeltaAdapter
from .config import SimConfig, load_config
from .contracts import (
    ARM_JOINTS,
    CARTESIAN_ACTION_DIM,
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    JOINT_ACTION_DIM,
    LEFT_JOINTS,
    LEFT_WRIST_IMAGE,
    RIGHT_JOINTS,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    ActionMode,
    validate_action,
    validate_observation,
)
from .model import (
    ROBOTIQ_2F85_JAW_TRAVEL_M,
    ModelBundle,
    SceneName,
    build_model,
)


class A3DualArmEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Functional A3 tabletop sandbox with a policy-neutral API."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    # Robotiq 2F-85: each simplified jaw travels 42.5 mm for an 85 mm opening.
    GRIPPER_RANGE_M = ROBOTIQ_2F85_JAW_TRAVEL_M

    def __init__(
        self,
        config: SimConfig | str | Path | None = None,
        *,
        action_mode: ActionMode = "joint_position",
        render_mode: str | None = None,
        render_cameras: bool = True,
        scene: SceneName = "sandbox",
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, SimConfig) else load_config(config)
        if action_mode not in ("joint_position", "cartesian_delta"):
            raise ValueError(f"unsupported action mode: {action_mode}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render mode: {render_mode}")
        self.action_mode = action_mode
        self.render_mode = render_mode
        # Keep the observation schema fixed in fast debug mode: disabled policy
        # cameras produce black frames without constructing an offscreen GL context.
        self.render_cameras = render_cameras
        self.scene = scene
        bundle: ModelBundle = build_model(self.config, scene=scene)
        self.model = bundle.model
        self.model_xml = bundle.xml
        self.source_joints = bundle.source_joints
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._view_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._viewer: Any = None
        self._key_callback: Any = None
        self._step_count = 0
        self._max_recorded_gripper_force = 0.0
        self._safety_stop = False
        self._safety_reason: str | None = None
        self._last_applied_action = np.zeros(JOINT_ACTION_DIM, dtype=np.float64)
        self._joint_ids = {name: self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS}
        self._qpos_ids = {
            name: int(self.model.jnt_qposadr[jid]) for name, jid in self._joint_ids.items()
        }
        self._dof_ids = {
            name: int(self.model.jnt_dofadr[jid]) for name, jid in self._joint_ids.items()
        }
        self._actuator_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position") for name in ARM_JOINTS
        }
        self._finger_joints = {
            side: tuple(
                self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger_{finger}_joint")
                for finger in ("inner", "outer")
            )
            for side in ("L", "R")
        }
        self._finger_actuators = {
            side: tuple(
                self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger_{finger}_position")
                for finger in ("inner", "outer")
            )
            for side in ("L", "R")
        }
        self._eef_sites = (
            self._id(mujoco.mjtObj.mjOBJ_SITE, "L_eef"),
            self._id(mujoco.mjtObj.mjOBJ_SITE, "R_eef"),
        )
        self._camera_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("front", "left_wrist", "right_wrist")
        }
        self._ik = CartesianDeltaAdapter(self.model, self.config)
        self._arm_ranges = np.asarray(
            [self.model.jnt_range[self._joint_ids[name]] for name in ARM_JOINTS],
            dtype=np.float64,
        )
        joint_low = np.r_[self._arm_ranges[:7, 0], 0.0, self._arm_ranges[7:, 0], 0.0]
        joint_high = np.r_[self._arm_ranges[:7, 1], 1.0, self._arm_ranges[7:, 1], 1.0]
        if self.action_mode == "joint_position":
            self.action_space = spaces.Box(joint_low, joint_high, dtype=np.float64)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (CARTESIAN_ACTION_DIM,), dtype=np.float64)
        image_shape = (self.config.image_height, self.config.image_width, 3)
        self.observation_space = spaces.Dict(
            {
                FRONT_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                LEFT_WRIST_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                RIGHT_WRIST_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                STATE: spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
                VELOCITY: spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
                EEF_POSE: spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
                FORCE: spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),
                "time": spaces.Box(0.0, np.inf, (), dtype=np.float64),
                "safety_stop": spaces.Discrete(2),
            }
        )
        self.metadata = dict(self.metadata, render_fps=self.config.control_hz)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise ValueError(f"model is missing {object_type.name} {name}")
        return result

    def _id_or_none(self, object_type: mujoco.mjtObj, name: str) -> int:
        """Look up an optional model element, returning -1 when it is absent.

        Scenes are built from shared code, so some elements only exist for some
        configs (the spare box, for one).  Callers that can work without an
        element should not have to know which configs include it.
        """

        return mujoco.mj_name2id(self.model, object_type, name)

    @property
    def last_applied_action(self) -> np.ndarray:
        return self._last_applied_action.copy()

    @property
    def current_joint_action(self) -> np.ndarray:
        return self._current_joint_action()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._safety_stop = False
        self._safety_reason = None
        homes = dict(zip(LEFT_JOINTS, self.config.home.left, strict=True))
        homes.update(dict(zip(RIGHT_JOINTS, self.config.home.right, strict=True)))
        for name, value in homes.items():
            joint_id = self._joint_ids[name]
            limits = self.model.jnt_range[joint_id]
            self.data.qpos[self._qpos_ids[name]] = np.clip(value, limits[0], limits[1])
        for side, opening in zip(("L", "R"), self.config.home.grippers, strict=True):
            physical = (1.0 - np.clip(opening, 0.0, 1.0)) * self.GRIPPER_RANGE_M
            for joint_id in self._finger_joints[side]:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = physical
        self._randomize_objects(enabled=(options or {}).get("randomize_objects", True))
        mujoco.mj_forward(self.model, self.data)
        home_action = np.r_[
            self.config.home.left,
            self.config.home.grippers[0],
            self.config.home.right,
            self.config.home.grippers[1],
        ]
        self._apply_controls(home_action)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self._max_recorded_gripper_force = 0.0
        self._last_applied_action = np.asarray(home_action, dtype=np.float64)
        if self.render_mode == "human":
            self._ensure_viewer()
        observation = self._observation()
        return observation, {
            "seed": seed,
            "action_mode": self.action_mode,
            "scene": self.scene,
            "policy_camera_rendering": self.render_cameras,
        }

    def _randomize_objects(self, *, enabled: bool) -> None:
        if not enabled:
            return
        noise = self.config.object_position_noise_m
        for index in range(3):
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"object_{index}_free")
            address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[address] += self.np_random.uniform(-noise, noise)
            self.data.qpos[address + 1] += self.np_random.uniform(-noise, noise)

    def emergency_stop(self, reason: str = "user emergency stop") -> None:
        self._safety_stop = True
        self._safety_reason = reason

    def set_key_callback(self, callback: Any) -> None:
        if self._viewer is not None:
            raise RuntimeError("key callback must be set before opening the viewer")
        self._key_callback = callback

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        requested = validate_action(action, self.action_mode)
        if self._safety_stop:
            applied = self._current_joint_action()
        else:
            canonical = (
                requested
                if self.action_mode == "joint_position"
                else self._ik.convert(self.data, requested)
            )
            if self.action_mode == "cartesian_delta":
                # An idle arm holds its commanded pose, not its gravity-deflected
                # measurement. Re-targeting the measurement integrates sag forever.
                for offset, start in ((0, 0), (7, 8)):
                    if not np.any(requested[offset:offset + 6]):
                        canonical[start:start + 7] = self._last_applied_action[start:start + 7]
            applied = self._limit_joint_action(canonical)
            self._apply_controls(applied)
            for _ in range(self.config.substeps):
                mujoco.mj_step(self.model, self.data)
                if not np.all(np.isfinite(self.data.qpos)) or not np.all(
                    np.isfinite(self.data.qvel)
                ):
                    self.emergency_stop("non-finite simulator state")
                    break
                if np.max(np.abs(self.data.qvel)) > 80.0:
                    self.emergency_stop("joint velocity exceeded safety threshold")
                    break
        self._last_applied_action = applied.copy()
        self._step_count += 1
        terminated = self._safety_stop
        truncated = self._step_count >= self.config.horizon and not terminated
        if self.render_mode == "human" and self._viewer is not None:
            self._update_viewer_force_hud()
            self._viewer.sync()
        observation = self._observation()
        info = {
            "applied_action": applied.copy(),
            "requested_action": requested.copy(),
            "safety_reason": self._safety_reason,
            "step": self._step_count,
        }
        return observation, 0.0, terminated, truncated, info

    def _limit_joint_action(self, canonical: np.ndarray) -> np.ndarray:
        command = validate_action(canonical, "joint_position").copy()
        # Rate-limit the commanded trajectory, not the measured state. Limiting against
        # measured qpos caps the position error (and therefore actuator torque), which can
        # make gravity compensation impossible for the shoulder joints.
        current = self._last_applied_action
        arm_command = np.r_[command[:7], command[8:15]]
        arm_current = np.r_[current[:7], current[8:15]]
        arm_command = np.clip(arm_command, self._arm_ranges[:, 0], self._arm_ranges[:, 1])
        if self.action_mode == "cartesian_delta":
            arm_delta = arm_command - arm_current
            max_delta = float(np.max(np.abs(arm_delta)))
            if max_delta > self.config.max_joint_step_rad:
                arm_delta = arm_delta * (self.config.max_joint_step_rad / max_delta)
            arm_command = np.clip(
                arm_current + arm_delta,
                self._arm_ranges[:, 0],
                self._arm_ranges[:, 1],
            )
        else:
            arm_command = np.clip(
                arm_command,
                arm_current - self.config.max_joint_step_rad,
                arm_current + self.config.max_joint_step_rad,
            )
        command[:7] = arm_command[:7]
        command[8:15] = arm_command[7:]
        for index in (7, 15):
            command[index] = np.clip(command[index], 0.0, 1.0)
            command[index] = np.clip(
                command[index],
                current[index] - self.config.max_gripper_step,
                current[index] + self.config.max_gripper_step,
            )
        return command

    def _apply_controls(self, canonical: np.ndarray) -> None:
        arm_values = np.r_[canonical[:7], canonical[8:15]]
        for name, value in zip(ARM_JOINTS, arm_values, strict=True):
            self.data.ctrl[self._actuator_ids[name]] = value
        for side, opening in zip(("L", "R"), (canonical[7], canonical[15]), strict=True):
            value = (1.0 - float(opening)) * self.GRIPPER_RANGE_M
            for actuator_id in self._finger_actuators[side]:
                self.data.ctrl[actuator_id] = value

    def _current_joint_action(self) -> np.ndarray:
        left = [self.data.qpos[self._qpos_ids[name]] for name in LEFT_JOINTS]
        right = [self.data.qpos[self._qpos_ids[name]] for name in RIGHT_JOINTS]
        grips = [self._gripper_state(side)[0] for side in ("L", "R")]
        return np.asarray([*left, grips[0], *right, grips[1]], dtype=np.float64)

    def _gripper_state(self, side: str) -> tuple[float, float]:
        positions = []
        velocities = []
        for joint_id in self._finger_joints[side]:
            positions.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            velocities.append(self.data.qvel[self.model.jnt_dofadr[joint_id]])
        return (
            float(1.0 - np.mean(positions) / self.GRIPPER_RANGE_M),
            float(-np.mean(velocities) / self.GRIPPER_RANGE_M),
        )

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address : address + dimension].copy()

    def _observation(self) -> dict[str, Any]:
        left_state = [self.data.qpos[self._qpos_ids[name]] for name in LEFT_JOINTS]
        right_state = [self.data.qpos[self._qpos_ids[name]] for name in RIGHT_JOINTS]
        left_velocity = [self.data.qvel[self._dof_ids[name]] for name in LEFT_JOINTS]
        right_velocity = [self.data.qvel[self._dof_ids[name]] for name in RIGHT_JOINTS]
        left_grip = self._gripper_state("L")
        right_grip = self._gripper_state("R")
        poses = []
        for site_id in self._eef_sites:
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[site_id])
            poses.extend(self.data.site_xpos[site_id])
            poses.extend(quaternion)
        force = np.r_[
            self._sensor("L_wrist_force"),
            self._sensor("L_wrist_torque"),
            self._sensor("R_wrist_force"),
            self._sensor("R_wrist_torque"),
            self._sensor("L_finger_inner_touch_sensor"),
            self._sensor("L_finger_outer_touch_sensor"),
            self._sensor("R_finger_inner_touch_sensor"),
            self._sensor("R_finger_outer_touch_sensor"),
            sum(self.data.actuator_force[list(self._finger_actuators["L"])]),
            sum(self.data.actuator_force[list(self._finger_actuators["R"])]),
        ]
        observation: dict[str, Any] = {
            FRONT_IMAGE: self._render_camera("front"),
            LEFT_WRIST_IMAGE: self._render_camera("left_wrist"),
            RIGHT_WRIST_IMAGE: self._render_camera("right_wrist"),
            STATE: np.asarray(
                [*left_state, left_grip[0], *right_state, right_grip[0]], dtype=np.float32
            ),
            VELOCITY: np.asarray(
                [*left_velocity, left_grip[1], *right_velocity, right_grip[1]], dtype=np.float32
            ),
            EEF_POSE: np.asarray(poses, dtype=np.float32),
            FORCE: np.asarray(force, dtype=np.float32),
            "time": float(self.data.time),
            "safety_stop": bool(self._safety_stop),
        }
        validate_observation(observation)
        return observation

    def _render_camera(self, name: str) -> np.ndarray:
        if not self.render_cameras:
            return np.zeros((self.config.image_height, self.config.image_width, 3), dtype=np.uint8)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_height,
                width=self.config.image_width,
            )
        self._renderer.update_scene(self.data, camera=self._camera_ids[name])
        self._apply_render_flags(self._renderer.scene)
        return np.ascontiguousarray(self._renderer.render(), dtype=np.uint8)

    def _apply_render_flags(self, scene: Any) -> None:
        """Force the configured light passes on every frame.

        The renderer reuses one persistent scene, so the flags are re-applied
        here rather than once at construction to stay correct regardless of how
        ``update_scene`` leaves them.
        """
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.config.render_shadows)
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(self.config.render_reflections)

    def use_fast_render(self) -> None:
        """Drop the per-light shadow and reflection passes for speed.

        Software rasterizers spend most of their frame time generating shadow
        and reflection maps, whose cost does not depend on camera resolution.
        Disabling both keeps geometry, materials, and colours intact at roughly
        four times the frame rate.
        """
        self.config = replace(self.config, render_shadows=False, render_reflections=False)

    def render_view(self, camera: str, *, height: int, width: int) -> np.ndarray:
        """Render one scene camera at an arbitrary resolution.

        The policy cameras are pinned to the resolution the checkpoint was
        trained on, so a larger frame for a human viewer needs its own renderer.
        Renderers are cached per resolution and reused across steps, and the view
        inherits the same light passes as the policy cameras so a recording shows
        exactly the scene the dataset recorded.

        Unlike :meth:`_render_camera` this ignores ``render_cameras``: the caller
        is explicitly asking for pixels.
        """
        if camera not in self._camera_ids:
            raise ValueError(f"unknown camera {camera!r}; available: {sorted(self._camera_ids)}")
        if height < 1 or width < 1:
            raise ValueError("view dimensions must be positive")
        # MuJoCo refuses to allocate an offscreen render larger than the model's
        # framebuffer, which the scene XML sizes to the training resolution.
        # Growing it is harmless for the policy cameras: each renderer keeps its
        # own height and width, so they still produce training-sized images.
        global_settings = self.model.vis.global_
        global_settings.offwidth = max(global_settings.offwidth, width)
        global_settings.offheight = max(global_settings.offheight, height)
        renderer = self._view_renderers.get((height, width))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._view_renderers[(height, width)] = renderer
        renderer.update_scene(self.data, camera=self._camera_ids[camera])
        self._apply_render_flags(renderer.scene)
        return np.ascontiguousarray(renderer.render(), dtype=np.uint8)

    def _ensure_viewer(self) -> None:
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            # Keep the useful front-camera viewpoint, but use a free camera so
            # mouse rotation/pan/zoom work immediately. Policy cameras stay fixed.
            self._configure_viewer_camera(self._viewer.cam)

    def _configure_viewer_camera(self, camera: mujoco.MjvCamera) -> None:
        target = np.asarray(self.config.cameras.workspace_target_m, dtype=np.float64)
        offset = np.asarray(self.config.cameras.front_position_m) - target
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.lookat[:] = target
        camera.distance = float(np.linalg.norm(offset))
        camera.azimuth = float(np.rad2deg(np.arctan2(-offset[1], -offset[0])))
        camera.elevation = -float(np.rad2deg(np.arctan2(offset[2], np.linalg.norm(offset[:2]))))

    def _update_viewer_force_hud(self) -> None:
        if self._viewer is None:
            return
        try:
            inner_f = float(self._sensor("L_finger_inner_touch_sensor")[0])
            outer_f = float(self._sensor("L_finger_outer_touch_sensor")[0])
            max_f = max(inner_f, outer_f)
            self._max_recorded_gripper_force = max(self._max_recorded_gripper_force, max_f)

            # 在屏幕内部以紧凑小字 HUD 叠加显示最大值与历史峰值
            if hasattr(self._viewer, "set_texts"):
                text1 = "MAX GRIPPER FORCE:\nPEAK RECORDED:"
                text2 = f"{max_f:.2f} N\n{self._max_recorded_gripper_force:.2f} N"
                try:
                    self._viewer.set_texts([
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            text1,
                            text2,
                        )
                    ])
                except Exception:
                    pass

            # 窗口标题栏同步显示当前最大值与峰值
            title_text = (
                f"MuJoCo : a3_cookie_transfer | "
                f"Max Force: {max_f:.2f} N (Peak: {self._max_recorded_gripper_force:.2f} N)"
            )
            try:
                self._viewer.title = title_text
            except Exception:
                pass

            window = getattr(self._viewer, "_window", None)
            if window is not None:
                try:
                    import glfw
                    glfw.set_window_title(window, title_text)
                except Exception:
                    pass
        except Exception:
            pass

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._ensure_viewer()
            self._update_viewer_force_hud()
            self._viewer.sync()
            return None
        return self._render_camera("front")

    def close(self) -> None:
        # A human viewer and the offscreen camera renderer own separate OpenGL
        # contexts.  Destroy the viewer first; GLFW may otherwise tear down shared
        # process state while the visible context is still live (exit-time SIGSEGV).
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        for renderer in self._view_renderers.values():
            renderer.close()
        self._view_renderers.clear()
