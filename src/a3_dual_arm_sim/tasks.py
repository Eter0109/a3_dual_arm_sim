"""The task vocabulary shared by collection, recording, and training.

A dataset is only usable if the code that collected it and the code that trains on
it agree on three things: which task it is, what language prompt conditioned it,
and which environment-side success criterion produced the accepted episodes.
Keeping those three in one record is what turns a new scene into a registration
instead of another copy of the collection driver.

Why this module exists separately from :mod:`a3_dual_arm_sim.collection`:
``training`` must import on a host where MuJoCo will not even load (a GPU-only
worker may expose no AVX at all, and the wheels abort on import).  So the
vocabulary lives here, with no simulator import, and ``collection`` binds it to
environments and scripted policies.

Note that a task *name* identifies a data contract, not a goal.  Every cookie
variant shares one prompt -- from a policy's point of view they are the same task,
which is what makes their datasets comparable -- but they carry different success
contracts, because ``a3_cookie_transfer`` requires an exact slot fill while the
batch variants only require ten released, upright, settled, contained Cookies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """What a dataset of one task is, independently of how it was produced."""

    name: str
    #: Language conditioning stored with every frame.
    prompt: str
    #: Identifier of the environment criterion that decides `success`.  It is the
    #: dataset's acceptance rule, so a change here invalidates old datasets and
    #: `training.audit_training_dataset` refuses to train on them.
    success_contract: str


COOKIE_PROMPT = "transfer exactly ten upright square cookie blocks into the 2x5 box"
GRASP_PROMPT = "pick up the red cube"

#: Ten Cookies released by the pad contact chain, then upright, settled and
#: contained in the target box, with seventy left in the source.  This is the
#: criterion both batch experts satisfy (`require_released=True`,
#: `require_exact_slots=False`): the five-by-five grasp places two batches of five
#: and does not aim at the ten slot centres a single-Cookie grip does.
_BATCH_CONTRACT = "released_10_upright_settled_contained_70_remain"

TASKS: dict[str, TaskSpec] = {
    "a3_grasp": TaskSpec(
        name="a3_grasp",
        prompt=GRASP_PROMPT,
        success_contract="dual_contact_centered_clear_lifted_low_motion_1s",
    ),
    "a3_cookie_transfer": TaskSpec(
        name="a3_cookie_transfer",
        prompt=COOKIE_PROMPT,
        success_contract="exact_2x5_fill_touching_all_walls_upright_1s",
    ),
    # Same goal and prompt as `a3_cookie_transfer`, different scripted controller.
    "a3_cookie_batch": TaskSpec(
        name="a3_cookie_batch",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    "a3_cookie_same_column": TaskSpec(
        name="a3_cookie_same_column",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The relay fills two boxes rather than one, so its acceptance rule is its own:
    # ten Cookies in each box with sixty left in the source, box A pushed clear and
    # box B back in the station.  The environment cannot express it -- its task
    # config checks a single target bin -- which is why this task's contract is the
    # only one an expert, rather than the environment, decides.
    "a3_cookie_two_box": TaskSpec(
        name="a3_cookie_two_box",
        prompt=COOKIE_PROMPT,
        success_contract="released_10_each_box_60_remain_boxes_relocated",
    ),
}


def task_spec(name: str) -> TaskSpec:
    """Look up a task, with a message that lists what does exist."""

    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; registered tasks are {sorted(TASKS)}"
        ) from None
