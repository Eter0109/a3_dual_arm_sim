import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env, Cookie2x10TaskConfig
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan, parse_first_grasp
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.workflows.cookie_2x10_execution import Cookie2x10EpisodeExecution

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", range(8))
def test_initial_random_source_poses_are_clear_and_within_noise_bounds(seed):
    env = A3Cookie2x10Env(
        render_cameras=False,
        task_config=Cookie2x10TaskConfig(position_noise_m=0.0003, yaw_noise_rad=0.015),
    )
    try:
        env.reset(seed=seed, options={"randomize_cookies": True})
        offsets = env.cookie_positions[:, :2] - np.asarray(env.SOURCE_POSITIONS)
        assert np.max(np.abs(offsets)) <= env.task_config.position_noise_m + 1e-5
        geom_cookie = {
            geom: index for index, geoms in enumerate(env._cookie_collision_geoms) for geom in geoms
        }
        interpenetrations = [
            contact
            for contact in env.data.contact
            if int(contact.geom1) in geom_cookie
            and int(contact.geom2) in geom_cookie
            and geom_cookie[int(contact.geom1)] != geom_cookie[int(contact.geom2)]
            and contact.dist < -1e-6
        ]
        assert not interpenetrations
        assert env.randomization_metadata["source_pose_resamples"] >= 0
    finally:
        env.close()


@pytest.mark.parametrize("seed,n", [(0, 0), (1, 9), (3, 5)])
def test_far_column_final_group_has_reachable_pose_clear_of_box_wall(seed, n):
    runner = Cookie2x10EpisodeExecution(first_grasp=n, source_column="4", profile="advanced")
    env = runner.create_env(render_cameras=False)
    try:
        env.reset(
            seed=seed,
            options={
                "randomize_cookies": runner.randomize_cookies,
                "randomize_boxes": runner.randomize_boxes,
                "randomize_source_bin": runner.randomize_boxes,
                "randomize_target_bin": runner.randomize_boxes,
                "source_bin_noise_m": runner.source_bin_noise_m,
                "target_bin_noise_m": runner.target_bin_noise_m,
                "target_bin_yaw_noise_rad": runner.target_bin_yaw_noise_rad,
                "randomization_profile": runner.profile,
                "appearance_seed": seed,
                "randomization_settings": runner.randomization_settings,
            },
        )
        expert = A3Cookie2x10Expert(env, first_grasp=n, requested_source_column_index=3)
        expert.reset(EpisodeContext(seed=seed, task="", action_mode=env.action_mode))
        # Probe the last group without running the preceding physical transfers.
        expert.batch_index = 3
        expert.completed_cookie_indices = list(range(60, 70 + n))
        expert._select_batch()
        work = mujoco.MjData(env.model)
        work.qpos[:] = env.data.qpos
        work.qpos[expert._l_qpos] = env.solve_ik(
            expert._pick_eef, expert._grasp_quat, expert.q_transit
        )
        mujoco.mj_forward(env.model, work)
        assert np.linalg.norm(work.site_xpos[expert._l_site] - expert._pick_eef) < 0.001
        base = env.model.geom("L_2f85_base_collision").id
        wall_contacts = [
            contact
            for contact in work.contact
            if base in (contact.geom1, contact.geom2)
            and "source"
            in (env.model.geom(int(contact.geom1)).name or "")
            + (env.model.geom(int(contact.geom2)).name or "")
        ]
        assert not wall_contacts
    finally:
        env.close()


@pytest.mark.parametrize("n", range(10))
def test_plan_transfers_twenty_in_two_complete_columns(n):
    plan = Cookie2x10Plan.from_seed(str(n), 37)
    assert plan.batch_sizes == (n, 10 - n, n, 10 - n)
    for first in (0, 2):
        rows = plan.target_rows(first) + plan.target_rows(first + 1)
        assert rows == tuple(range(10))
    prompt = plan.prompt(3)
    assert "20 cookies" in prompt and "2x10" in prompt and "source column 3" in prompt
    if n == 0:
        assert "Skip grasp 1" in prompt and "Skip grasp 3" in prompt
    else:
        assert f"Grasp {n} cookies first, then {10 - n}" in prompt


@pytest.mark.parametrize("invalid", [-1, 10, 3.5, "10", "-1", "3.0", "RANDOM", ""])
def test_invalid_first_grasp_is_rejected(invalid):
    with pytest.raises(ValueError):
        parse_first_grasp(invalid)


def test_random_is_repeatable_and_uses_one_draw_per_episode():
    plans = [Cookie2x10Plan.from_seed("random", seed) for seed in range(100)]
    assert plans == [Cookie2x10Plan.from_seed("random", seed) for seed in range(100)]
    assert {p.first_grasp for p in plans} == set(range(10))
    assert all(p.batch_sizes[0] == p.batch_sizes[2] for p in plans)


def test_only_target_box_and_slots_change_in_new_scene():
    old = asdict(load_config(ROOT / "configs/cookie_batch.yaml"))
    new = asdict(load_config(ROOT / "configs/cookie_2x10.yaml"))
    assert len(old["cookie_transfer"]["target_slots_local_m"]) == 10
    assert len(new["cookie_transfer"]["target_slots_local_m"]) == 20
    assert "target_rows" not in old["cookie_transfer"]
    for key in ("target_bin_half_size_m", "target_slots_local_m"):
        old["cookie_transfer"].pop(key)
        new["cookie_transfer"].pop(key)
    assert new == old


@pytest.mark.parametrize("n", range(10))
def test_expert_consumes_exact_batches_from_one_source_column(n):
    expert = object.__new__(A3Cookie2x10Expert)
    expert.plan = Cookie2x10Plan.from_seed(n, 0)
    source = load_config(
        ROOT / "configs/cookie_2x10.yaml"
    ).cookie_transfer.cookie_source_positions_m
    expert.env = SimpleNamespace(
        SOURCE_POSITIONS=source, privileged_cookie_in_source=lambda i: True
    )
    expert._column_order = [source[40][0]]
    expert.completed_cookie_indices = []
    for batch_index, count in enumerate(expert.plan.batch_sizes):
        expert.batch_index = batch_index
        if count:
            batch = next(expert._candidate_batches())
            assert len(batch) == count
            assert set(batch).isdisjoint(expert.completed_cookie_indices)
            expert.completed_cookie_indices.extend(batch)
    assert expert.completed_cookie_indices == list(range(40, 60))


def test_new_task_cannot_silently_run_in_legacy_environment():
    with pytest.raises(ValueError, match="target slot count"):
        A3CookieTransferEnv(ROOT / "configs/cookie_2x10.yaml", render_cameras=False)


def test_twenty_success_requires_ten_released_per_column(monkeypatch):
    env = A3Cookie2x10Env(
        render_cameras=False,
        task_config=Cookie2x10TaskConfig(success_hold_steps=2, terminate_on_success=False),
    )
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        assert len(env.TARGET_SLOTS_LOCAL) == 20 and env.task_config.required_cookies == 20
        # Direct positioning is a fixture for the evaluator, never expert behavior.
        for i in range(20):
            env.set_cookie_pose(
                i, tuple(env.privileged_target_slot_world(i, env._target_cookie_center_z))
            )
        for _ in range(10):
            _, _, _, _, info = env.step(env.last_applied_action.copy())
        assert info["cookies_in_target"] == 20 and info["cookies_in_source"] == 60
        assert info["target_column_counts"] == (10, 10) and info["success"]
        assert "exact_2x5_fill" not in info
        assert not env._target_fill_is_valid(tuple(i < 10 for i in range(80)))
        monkeypatch.setattr(env, "privileged_left_finger_contacts", lambda i: (i == 0, False))
        assert not env.privileged_cookie_in_target(0)
        _, _, _, _, info = env.step(env.last_applied_action.copy())
        assert not info["success"]
    finally:
        env.close()


def test_zero_grasp_is_skipped_without_selecting_empty_cookie_batch(monkeypatch):
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert

    expert = object.__new__(A3Cookie2x10Expert)
    expert.plan = Cookie2x10Plan.from_seed(0, 0)
    expert.batch_index = 0
    expert.batch_reports = []
    expert._selection_batch_index = None
    expert.env = SimpleNamespace(COOKIE_HALF_SIZE=np.array([0.025, 0.0095 / 3, 0.0125]))
    selected = []
    monkeypatch.setattr(
        A3VariedColumnBatchExpert, "_select_batch", lambda self: selected.append(self.batch_size)
    )
    expert._select_batch()
    assert selected == [10] and expert.batch_index == 1
    assert expert.batch_reports == [
        {
            "grasp": 1,
            "planned_count": 0,
            "cookies": [],
            "lifted": 0,
            "released": True,
            "skipped": True,
        }
    ]
    expert.batch_index = 2
    expert._select_batch()
    assert selected == [10, 10] and expert.batch_index == 3
    assert expert.release_opening > 0.8


def test_episode_prompt_and_metadata_agree_with_seeded_plan():
    runner = Cookie2x10EpisodeExecution(first_grasp="random", profile="basic")
    for seed in range(20):
        metadata = runner._episode_metadata(1, seed)
        plan = Cookie2x10Plan.from_seed("random", seed)
        assert metadata["grasp_counts"] == list(plan.batch_sizes)
        assert metadata["task_instruction"] == runner._episode_task(1, seed)
        assert runner.required_cookies == 20


def test_twenty_runner_scores_twenty_and_stores_joint_actions():
    class Recorder:
        def start_episode(self, context, controller):
            self.context = context

        def set_episode_metadata(self, metadata):
            self.metadata = metadata.copy()

        def add_frame(self, observation, action):
            self.action = action

        def finish_episode(self, success):
            self.saved = success

        def discard_episode(self):
            raise AssertionError("complete 20-cookie episode should be saved")

    policy = SimpleNamespace(
        action_mode="cartesian_delta",
        name="fake_expert",
        finished=True,
        reset=lambda context: None,
        act=lambda obs, task: np.zeros(14),
        phase_name=CookiePhase.DONE.name,
    )
    joint_action = np.arange(16, dtype=float)
    env = SimpleNamespace(
        action_mode="joint_position",
        reset=lambda seed, options: ({"test": seed}, {}),
        step=lambda action: (
            {},
            1.0,
            True,
            False,
            {
                "success": True,
                "cookies_in_target": 20,
                "cookies_in_source": 60,
                "target_column_counts": (10, 10),
                "applied_action": joint_action,
            },
        ),
    )
    recorder = Recorder()
    result = Cookie2x10EpisodeExecution(first_grasp=3).run_episode(
        policy, 0, env=env, recorder=recorder
    )
    assert result.success and result.max_score == result.score == 20
    assert recorder.saved and recorder.context.task == Cookie2x10Plan.from_seed(3, 0).prompt(1)
    assert recorder.metadata["grasp_counts"] == [3, 7, 3, 7]
    assert recorder.metadata["target_column_counts"] == [10, 10]
    np.testing.assert_array_equal(recorder.action, joint_action)


@pytest.mark.parametrize("n", [0, 3, 9, "random"])
def test_dataset_audit_checks_actual_plan_and_prompt(tmp_path, n):
    from a3_dual_arm_sim.data.audit import CAMERA_KEYS, audit_training_dataset

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    features = {
        **{key: {"dtype": "video", "shape": [256, 256, 3]} for key in CAMERA_KEYS},
        "observation.state": {"shape": [16]},
        "action": {"shape": [16]},
    }
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
                "total_episodes": 1,
                "total_frames": 12,
                "total_tasks": 1,
            }
        )
    )
    (root / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "stored_action_mode": "joint_position",
                "target_layout": "2x10",
                "first_grasp_mode": n,
                "use_videos": True,
            }
        )
    )
    plan = Cookie2x10Plan.from_seed(n, 0)
    offset = 0
    grasp_reports = []
    for count in plan.batch_sizes:
        grasp_reports.append(
            {
                "planned_count": count,
                "lifted": count,
                "cookies": list(range(offset, offset + count)),
                "released": True,
            }
        )
        offset += count
    report = {
        "seed": 0,
        "success": True,
        "task": plan.prompt(1),
        "randomization": {
            **plan.metadata(),
            "source_column": 1,
            "cookies_in_target": 20,
            "cookies_in_source": 60,
            "target_column_counts": [10, 10],
            "grasp_reports": grasp_reports,
        },
    }
    (root / "a3_episode_metadata.jsonl").write_text(json.dumps(report) + "\n")
    assert audit_training_dataset(root, repo_id="local/test")["episodes"] == 1
    report["task"] = "transfer 10 cookies into target box"
    (root / "a3_episode_metadata.jsonl").write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="task instruction"):
        audit_training_dataset(root, repo_id="local/test")


def test_release_opening_uses_real_pad_offset_and_stack_width():
    expert = object.__new__(A3Cookie2x10Expert)
    expert.env = SimpleNamespace(
        _left_finger_geoms=(0, 1),
        _cookie_bodies=(0, 1, 2),
        config=SimpleNamespace(
            cookie_transfer=SimpleNamespace(left_finger_pad_half_thickness_m=0.0005)
        ),
        current_joint_action=np.array([0.0] * 7 + [0.4]),
        COOKIE_HALF_SIZE=np.array([0.025, 0.0095 / 3, 0.0125]),
    )
    expert.batch_indices = [0, 1, 2]
    gap_offset = 0.0053616
    pad_distance = 0.4 * 0.085 + gap_offset + 0.001
    expert.data = SimpleNamespace(
        geom_xpos=np.array([[0, -pad_distance / 2, 0], [0, pad_distance / 2, 0]]),
        xmat=np.tile(np.eye(3).ravel(), (3, 1)),
    )
    positions = np.array([[0, y, 0] for y in (-0.019 / 3, 0, 0.019 / 3)])
    expert._positions = lambda: positions
    expert._set_release_opening()
    physical_gap = expert.release_opening * 0.085 + gap_offset
    assert physical_gap == pytest.approx(0.019 + 0.006)


@pytest.mark.parametrize(
    "columns,source_count,should_fail",
    [((10, 10), 60, False), ((10, 10), 59, True), ((9, 11), 60, True)],
)
def test_expert_done_requires_complete_environment_goal(columns, source_count, should_fail):
    expert = object.__new__(A3Cookie2x10Expert)
    expert.env = SimpleNamespace(
        SOURCE_POSITIONS=[None] * 80,
        privileged_cookie_in_target=lambda i: i < 20,
        target_column_counts=lambda mask: columns,
        privileged_cookie_in_source=lambda i: i >= 80 - source_count,
    )
    failures = []
    expert._fail = failures.append
    expert._verify_complete_fill()
    assert bool(failures) == should_fail
