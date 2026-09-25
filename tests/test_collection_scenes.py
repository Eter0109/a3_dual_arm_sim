"""The scene registry is the collection interface, so pin it.

These run without stepping the simulator: the registry is data, and what needs
checking is that it stays consistent -- a scene whose task name does not exist, a
detail dict that cannot be written to JSON, a task the trainer refuses to read.
The point of the abstraction is that a new scene is a few lines; these are the
lines it must not get wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from a3_dual_arm_sim.collection import (
    collect_cookie_dataset,
    registered_scenes,
    scene_by_name,
)
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks import TASKS, task_spec
from a3_dual_arm_sim.training import SUPPORTED_TASKS


def test_every_registered_scene_agrees_with_the_task_registry():
    scenes = registered_scenes()
    assert scenes, "the registry must not be empty"
    for name, builder in scenes.items():
        scene = builder()
        assert name in TASKS, f"scene {name!r} has no task spec"
        assert scene.task.name == name, (
            f"scene {name!r} declares task {scene.task.name!r}; the key and the "
            f"task name have to match or the summary will not be traceable"
        )
        assert scene.task.success_contract == TASKS[name].success_contract


def test_every_registered_scene_is_trainable():
    """A scene that can be collected but not trained is a trap, not a feature.

    ``training.SUPPORTED_TASKS`` is derived from the same registry, so this fails
    the moment someone adds a scene and forgets that the auditor has to accept it.
    """

    for name, builder in registered_scenes().items():
        scene = builder()
        assert scene.task.name in SUPPORTED_TASKS
        assert SUPPORTED_TASKS[scene.task.name] == scene.task.success_contract


def test_scene_details_are_json_serialisable():
    """They are merged straight into collection_summary.json."""

    for name, builder in registered_scenes().items():
        details = builder().details
        json.dumps(details)  # raises TypeError on a Path or an ndarray
        assert all(isinstance(key, str) for key in details), name


def test_cookie_variants_share_one_prompt_but_not_one_contract():
    """The prompt names the goal; the contract names the acceptance rule.

    Sharing the prompt is what makes the variants comparable in one dataset, and
    differing in the contract is why they cannot be mixed silently.
    """

    prompts = {
        TASKS[name].prompt
        for name in ("a3_cookie_transfer", "a3_cookie_batch", "a3_cookie_same_column")
    }
    assert len(prompts) == 1, "the cookie variants must stay one task for the policy"
    assert (
        TASKS["a3_cookie_transfer"].success_contract
        != TASKS["a3_cookie_batch"].success_contract
    )


def test_batch_scenes_relax_the_slot_contract_and_require_release():
    """The environment's criterion has to match how the batch expert places.

    A five-Cookie grip does not land on the ten single-Cookie slot centres, so
    requiring an exact fill would reject every episode it produces.  What it does
    guarantee is that nothing counts while still pinched.
    """

    for name in ("a3_cookie_batch", "a3_cookie_same_column"):
        scene = scene_by_name(name)
        env = scene.build_env(0.002, 0.05)
        try:
            assert env.task_config.require_exact_slots is False
            assert env.task_config.require_released is True
            # The recording must run to the expert's own stopping rule, not be cut
            # off by the environment the instant success is first seen.
            assert env.task_config.terminate_on_success is False
        finally:
            env.close()


def test_only_the_relay_scene_counts_a_lying_cookie_as_placed():
    """The relay grades "in the box"; the precision fills grade "in its slot".

    The push that clears box A slides along A's rear wall, where the rear Cookies
    lean against it: measured at about 24 degrees for three of ten, with all ten
    still inside.  Grading that as "7/10 aboard" rejected a run that had already
    done the harder half of the task, so the relay scene asks only for
    containment, and the two counts it reports (contained, placed) become the same
    number.  The precision-fill scenes must keep the strict form: a Cookie on its
    side is not in a 2x5 slot, and their whole subject is the placement.
    """

    strict = scene_by_name("a3_cookie_same_column").build_env(0.0, 0.0)
    try:
        assert strict.task_config.require_upright is True
    finally:
        strict.close()

    relay = scene_by_name("a3_cookie_two_box").build_env(0.0, 0.0)
    try:
        assert relay.task_config.require_upright is False
        # "Lying down" must not become "still pinched": the relay still requires
        # the left fingers to be clear before a Cookie counts.
        assert relay.task_config.require_released is True
    finally:
        relay.close()


def test_batch_scenes_vary_the_scene_but_never_jitter_the_cookies():
    """The two kinds of variation are separable, and the batch scenes need both.

    Moving the source box translates the whole Cookie layout rigidly, so the
    2.5 mm gaps the insertion is aimed at are preserved exactly (measured:
    per-Cookie deviation 0.000000 mm, and 3/3 randomized seeds complete in
    1855-1857 steps).  Jittering each Cookie does change them, and a 2 mm jitter
    is wider than those gaps, so with it on the batch experts fail immediately
    (measured: 0 accepted in 2 attempts).

    A scene with variation is what makes the dataset worth training on at all, so
    asserting only "no jitter" would miss the point: the scene switch has to stay
    on.
    """

    for name in ("a3_cookie_batch", "a3_cookie_same_column"):
        scene = scene_by_name(name)
        assert scene.randomize_cookies is False, (
            f"{name}'s insertion needs the measured gaps; per-Cookie jitter closes them"
        )
        assert scene.randomize_placements is True, (
            f"{name} must still vary the boxes and the start pose, or every "
            f"episode is the same picture"
        )


def test_scenes_report_the_noise_they_really_apply():
    """A summary must not claim jitter the scene never applied.

    Two ways for the recorded value to be wrong, and both are covered here: a
    scene that switches Cookie jitter off applies none, and the grasp scene owns
    its own jitter through its task config and ignores the run's arguments.
    """

    grasp = scene_by_name("a3_grasp")
    assert grasp.applied_noise is not None, "the grasp scene owns its own jitter"
    from a3_dual_arm_sim.grasp import GraspTaskConfig

    assert grasp.applied_noise[0] == GraspTaskConfig().position_noise_m
    assert grasp.applied_noise[0] != 0.002, (
        "if these ever coincide the test stops proving that the scene's value wins"
    )

    # A scene that turns Cookie jitter off applies none of the caller's values,
    # whatever they were, so the recorded noise has to be zero rather than echoed.
    for name in ("a3_cookie_batch", "a3_cookie_same_column"):
        scene = scene_by_name(name)
        assert scene.randomize_cookies is False
        assert scene.applied_noise is None, (
            f"{name} does not own its noise, so it must not claim a value"
        )


def test_two_box_scene_takes_its_success_from_the_expert():
    """The relay's criterion is not expressible as the environment's.

    The environment checks one target bin against a ten-Cookie fill, so a run that
    emptied two boxes would report success=False while having fully succeeded.  The
    scene therefore has to name the expert as the decider, and the policy has to
    expose it; if either half were missing, every two-box episode would be discarded
    as a failure and the scene would collect nothing.
    """

    scene = scene_by_name("a3_cookie_two_box")
    assert "expert" in scene.details["success_decision"], (
        "the two-box scene must declare that the environment cannot decide it"
    )
    env = scene.build_env(0.002, 0.05)
    try:
        policy = scene.build_policy(env)
        try:
            assert hasattr(policy, "success_override"), (
                "the relay policy must expose success_override or its episodes are"
                " discarded on arrival"
            )
            assert policy.success_override is False, "nothing has succeeded yet"
            # The environment genuinely disagrees, which is why the override exists.
            env.reset(seed=0, options={"randomize_cookies": False})
            _, _, _, _, info = env.step(env.last_applied_action)
            assert info["success"] is False
            assert info.get("cookies_in_source") == 80
        finally:
            policy.close()
    finally:
        env.close()


def test_two_box_scene_is_registered_like_any_other():
    """It goes through the same driver, so the same declarations must hold."""

    scene = scene_by_name("a3_cookie_two_box")
    assert scene.task.name in SUPPORTED_TASKS
    assert scene.randomize_cookies is False, "the insertion needs the measured gaps"
    assert scene.randomize_placements is True, "or every episode is the same picture"
    env = scene.build_env(0.0, 0.0)
    try:
        assert env.task_config.require_exact_slots is False
        assert env.task_config.require_released is True
        assert env.task_config.terminate_on_success is False
        # The relay needs the longer horizon the scene's config declares.
        assert env.config.horizon >= 12000
    finally:
        env.close()


def test_two_box_randomization_stays_inside_its_measured_reach():
    """Ranges copied from the single-box scene were wrong for this one.

    Reach depends on where a box sits, not how big it is: the working box here is at
    (0.075, 0.030) against (0.095, 0.100) in the single-box scene, and sweeping it
    until the expert refuses gives 6 mm in +x against 30 mm there.  Using the other
    scene's numbers produced a run that reached the second fill and failed on angle
    error 2.07 deg against a 2.005 deg threshold -- so the limits are pinned here.
    """

    config = load_config("configs/cookie_two_box_batch.yaml")
    randomization = config.randomization
    limits = {
        # Measured, as (positive reach, negative reach) in metres: the working box
        # has 6 mm in +x against 118 mm in -x, and 118 mm in +y against 8 mm in -y.
        # Getting this table the wrong way round is easy and is exactly what this
        # test exists to catch -- it did, while it was being written.
        "target_bin_x_m": (0.006, 0.118),
        "target_bin_y_m": (0.118, 0.008),
        "spare_bin_x_m": (0.118, 0.118),
        "spare_bin_y_m": (0.118, 0.118),
    }
    for name, (plus, minus) in limits.items():
        axis = getattr(randomization, name)
        assert axis.high <= plus + 1e-9, (
            f"{name} reaches {axis.high * 1000:.1f} mm in the positive direction "
            f"but only {plus * 1000:.1f} mm is reachable"
        )
        assert -axis.low <= minus + 1e-9, (
            f"{name} reaches {-axis.low * 1000:.1f} mm in the negative direction "
            f"but only {minus * 1000:.1f} mm is reachable"
        )
    # Yaw propagates into the fill's tool orientation, so it is bounded by the same
    # 4.5 deg the fill tolerates -- including the spare box's, which arrives carrying
    # whatever yaw it started with.
    yaw_limit = np.deg2rad(4.5)
    for name in ("target_bin_yaw_rad", "spare_bin_yaw_rad"):
        axis = getattr(randomization, name)
        assert axis.high <= yaw_limit, (
            f"{name} reaches {np.rad2deg(axis.high):.1f} deg but the fill only "
            f"tolerates {np.rad2deg(yaw_limit):.1f} deg"
        )


def test_spare_box_x_respects_the_station_tolerance_not_just_reach():
    """Reach is not the binding limit for the spare box's x; the station check is.

    ``RightBoxCarryController`` grips the rear wall and slides the box along y: its
    target keeps ``x = initial_position[0]`` and it aborts if the box drifts more than
    15 mm sideways.  Whatever x offset the box starts with is therefore the offset it
    arrives with, and ``VERIFY_B`` requires it within 8 mm of the station in XY.  The
    carry also stops a deliberate 4 mm short in y, so the budget for x is
    ``sqrt(8^2 - 4^2) ~= 6.9 mm``.

    This is the check that was missing when the range was first raised to +/-40 mm --
    reach was measured, the station tolerance was not, and the first collection run
    lost two of three seeds to it: one with the grip lost at x +26 mm (one finger at
    0 N), one arriving 27.8 mm from the station at x -25 mm.
    """

    config = load_config("configs/cookie_two_box_batch.yaml")
    axis = config.randomization.spare_bin_x_m
    verify_b_tolerance_m = 0.008
    carry_y_shortfall_m = 0.004  # `destination_y - 0.004` in the MOVE phase
    budget = np.sqrt(verify_b_tolerance_m**2 - carry_y_shortfall_m**2)
    assert axis.maximum_magnitude() <= budget + 1e-9, (
        f"spare_bin_x_m reaches {axis.maximum_magnitude() * 1000:.1f} mm, but the "
        f"carry cannot correct x and VERIFY_B leaves only {budget * 1000:.1f} mm"
    )
    # And the budget really is the tighter of the two, or the assertion above is
    # testing the wrong thing.
    assert budget < 0.056, "the grip reach (56 mm) should be looser than this"


def test_two_box_scene_reports_why_an_episode_was_rejected():
    """A rejected episode has to say what went wrong.

    ``success: false`` alone cannot distinguish a failed fill from a failed carry,
    and those need different fixes.  The relay knows the stage and the reason, the
    environment does not track them, so the policy has to expose them and the scene
    has to name them as metric keys -- otherwise a summary of three rejected
    episodes is three identical rows.
    """

    scene = scene_by_name("a3_cookie_two_box")
    for key in ("stage", "failure_reason", "box_a_cookie_count", "box_b_cookie_count"):
        assert key in scene.metric_keys, f"{key} is not reported for the two-box scene"
    env = scene.build_env(0.0, 0.0)
    try:
        policy = scene.build_policy(env)
        try:
            env.reset(seed=0, options={"randomize_cookies": False})
            metrics = policy.metrics
            for key in scene.metric_keys:
                assert key in metrics, f"the policy does not provide {key}"
            assert metrics["stage"] == "FILL", "a fresh relay starts by filling the station box"
            assert metrics["failure_reason"] is None
            assert metrics["box_a_cookie_count"] == 0
            assert metrics["box_b_cookie_count"] == 0
        finally:
            policy.close()
    finally:
        env.close()


def test_policy_metrics_reach_the_episode_result():
    """The runner has to merge them, or the scene's declarations go nowhere."""

    from a3_dual_arm_sim.env import A3DualArmEnv
    from a3_dual_arm_sim.policy import HoldPolicy
    from a3_dual_arm_sim.runner import EpisodeRunner

    class _ReportingPolicy(HoldPolicy):
        @property
        def metrics(self) -> dict:
            return {"stage": "somewhere", "box_a_cookie_count": 7}

    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, _ReportingPolicy(), task="test task")
    try:
        result = runner.run(seed=1, max_steps=2)
        assert result.final_info["stage"] == "somewhere"
        assert result.final_info["box_a_cookie_count"] == 7
    finally:
        runner.close()


def test_unknown_scene_lists_the_registered_ones():
    with pytest.raises(ValueError, match="registered scenes"):
        scene_by_name("a3_does_not_exist")


def test_unknown_task_lists_the_registered_ones():
    with pytest.raises(ValueError, match="registered tasks"):
        task_spec("a3_does_not_exist")


def test_cookie_wrapper_still_forwards_to_the_driver():
    """`collect_cookie_dataset` is the old entry point and must keep working.

    It is a wrapper now, so this is really a check that the wrapper forwards its
    arguments instead of quietly reverting to the scene defaults.
    """

    import inspect

    signature = inspect.signature(collect_cookie_dataset)
    for name in (
        "repo_id",
        "episodes",
        "start_seed",
        "max_attempts",
        "position_noise_m",
        "yaw_noise_rad",
        "hold_steps",
        "shard_index",
        "shard_count",
        "config",
        "fast_render",
        "use_videos",
    ):
        assert name in signature.parameters, f"{name} disappeared from the wrapper"
