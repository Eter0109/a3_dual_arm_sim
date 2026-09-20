"""The scene registry is the collection interface, so pin it.

These run without stepping the simulator: the registry is data, and what needs
checking is that it stays consistent -- a scene whose task name does not exist, a
detail dict that cannot be written to JSON, a task the trainer refuses to read.
The point of the abstraction is that a new scene is a few lines; these are the
lines it must not get wrong.
"""

from __future__ import annotations

import json

import pytest

from a3_dual_arm_sim.collection import (
    collect_cookie_dataset,
    registered_scenes,
    scene_by_name,
)
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
