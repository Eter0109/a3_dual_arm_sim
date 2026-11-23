"""Regression contracts for opt-in scene experts, without altering native demos."""

from types import SimpleNamespace

import pytest

from a3_dual_arm_sim.controllers.scene_batch_expert import A3Scene2x5Expert, A3Scene2x10Expert
from a3_dual_arm_sim.data.audit import _check_scene_episode
from a3_dual_arm_sim.data.recording import lerobot_features
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def _fake_expert(cls, present, tilted=(), boxes=3):
    e = object.__new__(cls)
    e.env = SimpleNamespace(
        _scene_randomizer=SimpleNamespace(groups=tuple(tuple(range(i, i + 5)) for i in range(0, 80, 5))),
        privileged_cookie_in_source=lambda i: i in present,
        target_box_bodies=tuple(range(100, 100 + boxes)),
        task_config=SimpleNamespace(required_cookies=20 if cls is A3Scene2x10Expert else 10),
    )
    e._selected_group = None
    e._tilted_ids = frozenset(tilted)
    e.completed_cookie_indices = []
    e._preferred_source_column = e._current_source_column = 0
    e.batch_index = 0
    e.plan = Cookie2x10Plan.from_seed(5, 0)
    return e


@pytest.mark.parametrize("cls", [A3Scene2x5Expert, A3Scene2x10Expert])
def test_skips_removed_groups_and_switches_columns(cls):
    present = set(range(15, 80))
    e = _fake_expert(cls, present)
    assert next(e._candidate_batches()) == list(range(15, 20))
    assert e._is_followup_batch
    e.completed_cookie_indices.extend(range(15, 20))
    present.difference_update(range(15, 20))
    e._selected_group = None
    assert next(e._candidate_batches()) == list(range(20, 25))
    assert e.selected_source_column_index == 1
    assert not e._is_followup_batch


def test_explicit_lean_is_selected_instead_of_avoiding_it():
    e = _fake_expert(A3Scene2x5Expert, set(range(80)), tilted=range(40, 45))
    assert next(e._candidate_batches()) == list(range(40, 45))
    assert e.selected_source_column_index == 2
    e._grasp_tilt_deg = 8
    e._scene_retries = 0
    e._base_push_attempts = 0
    assert e._align_grasp_requested()
    assert not e._straighten_requested()


@pytest.mark.parametrize("cls,batches", [(A3Scene2x5Expert, 2), (A3Scene2x10Expert, 4)])
def test_target_slots_repeat_per_box_and_previous_release_is_not_miscounted(cls, batches):
    e = _fake_expert(cls, set(range(80)), boxes=4)
    assert e.num_batches == batches * 4
    e.batch_index = batches + 1
    assert e._target_box_index == 1 and e._target_batch_index == 1
    assert e.batch_size == 5
    e._tb_id = 101
    e._assigned_target_bodies = {0: 100}
    e.env.privileged_cookie_in_target = lambda i, target_body: (i, target_body) in {(0, 100), (5, 101)}
    assert e._released_cookie_in_target(0)
    assert e._released_cookie_in_target(5)
    assert not e._released_cookie_in_target(6)


def test_native_variable_grasp_is_unchanged_and_scene_plan_is_explicit():
    default = make_cookie_benchmark("2x10", first_grasp=3)
    assert default.first_grasp == 3
    settings = load_randomization_config()
    settings["scene"] = {"target_box_count": 3, "source_gap_groups": 2}
    scene = make_cookie_benchmark("2x10", first_grasp="random", randomization_settings=settings)
    assert scene.first_grasp == 5 and scene.scene_requested_first_grasp == "random"
    assert scene.required_cookies == 60
    meta = scene._episode_metadata(1, 0)
    assert meta["grasp_counts"] == [5] * 12 and meta["required_cookies"] == 60
    assert "all 3 target boxes" in scene._scene_task_instruction()
    assert scene._episode_policy("variable_grasp", 1, 0).name == "cookie_scene_2x10_expert"


def test_left8_collection_features_keep_all_three_cameras():
    f = lerobot_features(256, 256, use_videos=True, left_arm_only=True)
    assert f["observation.state"]["shape"] == (8,)
    assert f["action"]["shape"] == (8,)
    assert f["observation.velocity"]["shape"] == (8,)
    assert f["observation.eef_pose"]["shape"] == (7,)
    assert f["observation.force"]["shape"] == (9,)
    assert sum(v["dtype"] == "video" for v in f.values()) == 3
    assert lerobot_features(256, 256)["action"]["shape"] == (16,)


def test_scene_dataset_acceptance_requires_every_box_and_no_duplicate_or_missing_cookies():
    summary = {"target_layout": "2x10", "required_cookies": 40, "task_instruction": "fill both"}
    episode = {"task": "fill both", "randomization": {
        "scene": {"target_boxes": [{}, {}], "initial_source_count": 70,
                  "gap_groups": [list(range(40, 45)), list(range(45, 50))]},
        "cookies_in_target": 40, "cookies_in_source": 30, "final_expert_phase": "DONE",
        "scenario_metrics": {"target_counts": [20, 20], "target_filled": [True, True],
                             "crushed_cookie_ids": [], "target_column_counts_by_box": [[10, 10], [10, 10]]},
        "grasp_reports": [{"target_box": j // 4 + 1, "target_batch": j % 4 + 1,
                           "planned_count": 5, "lifted": 5, "released": True,
                           "cookies": list(range(j * 5, j * 5 + 5))} for j in range(8)],
    }}
    _check_scene_episode(episode, summary)
    episode["randomization"]["grasp_reports"][-1]["cookies"] = list(range(45, 50))
    with pytest.raises(ValueError):
        _check_scene_episode(episode, summary)
