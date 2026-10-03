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


COOKIE_PROMPT = (
    "grasp five yellow blocks together from the source bin, insert them upright "
    "into the marked slots of the 2 by 5 box, then return to the source bin, grasp "
    "five more, and fill all ten slots"
)
"""The instruction every Cookie variant is conditioned on.

Every phrase here is meant to be something the cameras can actually show, because a
word that names nothing visible is not an instruction, it is a guess.  The render
was changed alongside this wording, so the two agree:

- *"yellow blocks"*.  The ``detailed`` style renders two tones of yellow alternating
  down the stack.  A colour word only helps if the picture agrees with it; under the
  older ``plain`` style the blocks are amber, and the word is then an approximation
  rather than a description.
- *"upright"*.  The blocks stand on edge in the source bin and have to end that way
  -- the success contract is ``upright`` too -- but nothing about a rectangle on a
  table says which axis is meant.  The word is the only place that is stated.
- *"the marked slots of the 2 by 5 box"*.  The box interior (126 mm by 70 mm) is much
  larger than the ten-slot lattice inside it (106 mm by 31.7 mm), so "the 2 by 5 box"
  used to name a region the box's own outline did not mark.  The ``detailed`` style
  draws the ten slots on the floor, so "marked slots" now has a referent.  This is
  the phrase that gained the most from the closer camera: at the old pose the lattice
  was 339 pixels and effectively invisible.
- *"five ... then five more ... all ten slots"*.  The bin holds eighty blocks and the
  box needs ten, so "transfer ten" leaves the first grasp undetermined -- ten at once
  is not graspable, and one at a time is not what the scripted experts do.  The
  prompt states the split they actually perform.

"square" was dropped from the older wording: the blocks are 50 mm by 25 mm by 6.3 mm,
so their silhouette is about 2:1, not square.  No camera shows a square face.
"""

GRASP_PROMPT = "pick up the red cube"

#: Ten Cookies released by the pad contact chain, then upright, settled and
#: contained in the target box, with seventy left in the source.  This is the
#: criterion both batch experts satisfy (`require_released=True`,
#: `require_exact_slots=False`): the five-by-five grasp places two batches of five
#: and does not aim at the ten slot centres a single-Cookie grip does.
_BATCH_CONTRACT = "released_10_upright_settled_contained_70_remain"

#: The instruction for the one-Cookie scene, and the only variant whose prompt is not
#: ``COOKIE_PROMPT``.
#:
#: The prompt has to change with the layout, and that is not a stylistic choice: the shared
#: wording asks for *five at a time, twice, all ten slots*, none of which is what this scene
#: is.  Asking for ten while showing one would make the instruction unanswerable from the
#: picture, which is the failure the shared prompt's own docstring is written to avoid.
#: Naming "one" and showing one keeps the sentence and the scene in agreement.
#:
#: It also means a policy conditioned on this prompt is not comparable to one conditioned on
#: ``COOKIE_PROMPT``: the two are different tasks to the model, which is why this is a
#: separate entry in ``TASKS`` rather than another variant sharing the string.
_SINGLE_PIECE_PROMPT = (
    "pick up the one yellow block from the source bin and put it in the target box"
)

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
    # One Cookie, one instruction.  The same physical act -- out of the source bin, into the
    # target box -- reduced to a single piece, so that a result separates two questions the
    # ten-Cookie scene answers at once: whether a policy can approach, grasp and place at
    # all, and whether it can do so ten times without knocking the stack over.  A failure
    # here is about the first; it says nothing about the second.
    #
    # `require_released=True` because a Cookie still held by the pads is not placed, and
    # `require_exact_slots=False` because the 2x5 slot grid is about the ten-piece fill: with
    # one piece, "counted in the box and let go of" is the whole of the task.
    "a3_cookie_single_piece": TaskSpec(
        name="a3_cookie_single_piece",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="released_1_upright_settled_contained_0_remain",
    ),    # The relay fills two boxes rather than one, so its acceptance rule is its own:
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
