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

#: The same instruction for two, which is what the collection actually asks for.  The words it
#: uses -- `block`, `source bin`, `target box` -- occur zero times in LIBERO's vocabulary, so a
#: policy has to be taught them rather than infer them.
_TWO_PIECE_PROMPT = (
    "pick up the two yellow blocks from the source bin and put them in the target box"
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
    # The same job, the same words, the same contract -- with the second batch met from
    # more than the one state the expert's first batch leaves behind.  The prompt and the
    # contract are the baseline's own, character for character: the instruction is the same
    # job (a policy has to learn to read the box's progress from the picture, not from a
    # different sentence), and the contract still asks for ten released and seventy
    # remaining, which the prefilled episodes satisfy by construction.
    "a3_cookie_batch_second_batch": TaskSpec(
        name="a3_cookie_batch_second_batch",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The same job and the same words, with the scene pinned to main's benchmark arrangement
    # instead of this branch's.  Registered under its own name because a dataset's provenance
    # records the task name, and "which arrangement was this collected on" decides whether a
    # number can be compared with main's benchmark at all: the two arrangements differ by 23/42 mm
    # on the bins, which is the difference between a grasp and closing the fingers on air.
    "a3_cookie_batch_second_batch_aligned": TaskSpec(
        name="a3_cookie_batch_second_batch_aligned",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The same task on a different embodiment: the packing, the prompt and the contract are
    # `a3_cookie_batch`'s, and only the arm and the cameras are `libero_props`'.  Registered
    # as its own task rather than aliased because a dataset's provenance records the task
    # name, and "which robot was this collected on" is the first thing a later reader needs.
    "a3_libero_panda_dense_batch": TaskSpec(
        name="a3_libero_panda_dense_batch",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # **The same scene on the other arm**, and registered separately for the same reason: it is a
    # one-variable comparison and the variable is the embodiment.  `a3_libero_panda_dense_batch`
    # is unsolvable by its own scripted expert at *any* batch size -- LIBERO's jaws are ~25 mm
    # thick against a 2.50 mm groove, and the packing leaves 3.3 / 15.1 mm at a column's ends --
    # so asking a policy on that embodiment measures the policy against a scene no controller can
    # finish.  The A3 arm's own gripper is 1.00 mm thick and solves the same packing five at a
    # time (`a3_cookie_batch`, 539 steps, 10 in the target box), which is what makes the prompt
    # and the contract above reachable here rather than decorative.
    "a3_libero_dense_batch": TaskSpec(
        name="a3_libero_dense_batch",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The controlled pair of the entry above: same task, packing, camera and box, with the arm as
    # the only variable -- and both at a placement A3 can actually reach, which the original dense
    # scene's placement is not (measured: its transit pose is 7.73 mm out of A3's reach).
    "a3_libero_panda_dense_batch_a3_placement": TaskSpec(
        name="a3_libero_panda_dense_batch_a3_placement",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The A3 arm back at LIBERO's placement, with the robot slid to meet the task instead of the
    # task slid away from the robot.  Same prompt and same contract as the two entries above --
    # this is the same job in the same frame, and only the mount moved.
    "a3_libero_dense_batch_libero_mount": TaskSpec(
        name="a3_libero_dense_batch_libero_mount",
        prompt=COOKIE_PROMPT,
        success_contract=_BATCH_CONTRACT,
    ),
    # The sparse counterpart: the two-Cookie scene on the A3 arm, with the robot slid into LIBERO's
    # frame instead of the arrangement slid out of the arm's reach.  Its task is the one the
    # reference arm completes, so a behaviour here is a statement about the arm.
    #
    # **The prompt and the contract are the two-Cookie scene's own, character for character.**  The
    # instruction is a policy input, and the shared ten-piece wording ("grasp five ... all ten
    # slots") applied to a scene holding two Cookies would make the sentence unanswerable from the
    # picture -- so a pair that differed here would differ in the question as well as in the arm,
    # and could not be read as an arm comparison at all.
    "a3_libero_two_cookies_libero_mount": TaskSpec(
        name="a3_libero_two_cookies_libero_mount",
        prompt=_TWO_PIECE_PROMPT,
        success_contract="released_2_upright_settled_contained_0_remain",
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
    ),
    # The two behaviour probes are the same task with one layout fact moved, and they carry
    # their own names only because the registry's invariant is that a scene key and its task
    # name agree -- which is what makes a summary row traceable back to the scene that wrote
    # it.  Their *contract* is `a3_cookie_single_piece`'s, unchanged: the probe changes where
    # the box starts or where the arm starts, never what counts as done.  A scene whose
    # success rule differed would be measuring something else and would need a rule of its
    # own rather than a copy of this one.
    #
    # They are registered rather than built inline so that `probe_ladder_object.py --scene
    # <name>` reproduces the run that produced the numbers in
    # `scene-transition-and-bug-triage.md`, and so a sweep that iterates the registry finds
    # them instead of a reader having to know their names.
    "a3_probe_box_in_bin": TaskSpec(
        name="a3_probe_box_in_bin",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="released_1_upright_settled_contained_0_remain",
    ),
    "a3_probe_home_over_cookie": TaskSpec(
        name="a3_probe_home_over_cookie",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="released_1_upright_settled_contained_0_remain",
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
    # The LIBERO prop scenes.  They borrowed `a3_cookie_single_piece`'s spec until now, which
    # is why the registry's own consistency test had been red: a scene keyed `a3_libero_props`
    # was declaring a task named `a3_cookie_single_piece`, and a summary row could not be
    # traced back to the scene that wrote it.  Borrowing the *prompt* is fine and deliberate --
    # these scenes exist to watch a LIBERO-trained policy under LIBERO's own camera, so they
    # have to be asked in words it knows -- but the acceptance rule has to be stated per scene,
    # and for four of the five there is no rule to state.
    #
    # Only `a3_libero_panda_single_cookie` is `scorable` (see the scene's own `details`): the
    # other four have no reachable Cookie, or ask for something no dataset contains, so
    # `success` is False by construction and a success rate there carries no information.  An
    # explicit `unscored_...` contract says that out loud rather than dressing those runs in
    # the single-piece rule and letting a reader divide by the wrong denominator.  `training`
    # derives its accepted task list from this registry, so what is here is exactly what a
    # dataset is allowed to claim.
    "a3_libero_panda_single_cookie": TaskSpec(
        name="a3_libero_panda_single_cookie",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="released_1_upright_settled_contained_0_remain",
    ),
    # Stage 2's scene: the same rule as the single-Cookie task, twice, in one episode.  The
    # contract names two rather than one because that is the denominator a run of this scene
    # would be divided by, and a summary that borrowed the single-piece rule would be wrong.
    "a3_libero_panda_two_cookies": TaskSpec(
        name="a3_libero_panda_two_cookies",
        prompt=_TWO_PIECE_PROMPT,
        success_contract="released_2_upright_settled_contained_0_remain",
    ),
    "a3_libero_props": TaskSpec(
        name="a3_libero_props",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="unscored_no_cookie_in_the_scene",
    ),
    "a3_libero_props_matched": TaskSpec(
        name="a3_libero_props_matched",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="unscored_no_cookie_in_the_scene",
    ),
    "a3_libero_panda": TaskSpec(
        name="a3_libero_panda",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="unscored_no_cookie_in_the_scene",
    ),
    "a3_libero_panda_cookie": TaskSpec(
        name="a3_libero_panda_cookie",
        prompt=_SINGLE_PIECE_PROMPT,
        success_contract="unscored_no_cookie_in_the_source_bin",
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
