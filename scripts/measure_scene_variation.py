"""What the training data randomises, and whether evaluation sits inside it.

Both halves have to be measured rather than read off the config, and the second half was
answered wrongly the first time round for a reason worth stating: this project's
comparison CLI runs the **canonical** layout by default -- ``reset_options`` disables all
three randomisation switches unless ``--randomize`` is passed -- so an evaluation episode
is not a sample from the training distribution at all.  It is one specific layout, and the
seed does not move it.  Asking "are the evaluation seeds inside the training range" and
answering it with ``randomize_scene: True`` forced on measures a configuration the
evaluation never runs.

So three things are reported separately:

``what training varies``
    The layout read out of the environment for the seeds the collection recorded, against
    the config's declared ranges.  A range that is written down but not applied, or
    applied but narrower than claimed, shows up here.

``what training does not vary``
    Stated explicitly, because it bounds what a policy trained on the result can learn.
    A layout that moves in millimetres on a fixed arrangement is not the same kind of
    variation as a scene that re-arranges itself.

``where evaluation sits``
    The canonical layout the CLI actually uses, and separately what ``--randomize`` would
    give.  The first is what every reported number was produced with, so the distance that
    matters is from the canonical layout to the training set.

    MUJOCO_GL=egl /root/autodl-tmp/a3-collect/.venv/bin/python scripts/measure_scene_variation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.collection import registered_scenes  # noqa: E402
from a3_dual_arm_sim.contracts import ARM_JOINTS  # noqa: E402

#: Seeds the collection used.  `collect_cookie.sh` starts at SEED_BASE (2000 in the
#: run that produced the training data) and adds 20 per episode; sharding spreads
#: those over parallel processes, so the stride is 20 across the whole dataset.
TRAINING_SEEDS = [2000 + 20 * index for index in range(200)]
#: Seeds `a3-sim compare` defaults to for a three-episode run.
EVAL_SEEDS = [0, 1, 2]

#: Environment rebuilds per N resets, and the number is **measured**, not guessed.
#:
#: On this scene the environment stops writing the arm's start pose after about 90
#: consecutive resets on one instance: reset 89 (seed 3780) left ``L_SHOULDER_P`` at exactly
#: 0.0, and by reset 200 there were three such samples with the arm-home standard deviation
#: inflated from 0.023 to 0.159 rad.  The first 60 resets reproduce the uniform draw exactly
#: (sd 0.0215 against 0.04/sqrt(3) = 0.0231), which is what makes the corruption visible
#: rather than merely suspicious.
#:
#: This is a property of the scene, not of the measurement: the collection runs are safe
#: because each shard is a fresh process that collects a few dozen episodes and exits, so no
#: shard comes near the limit.  A sweep that resets 200 times in one process does.
#:
#: 40 is chosen to sit well inside the measured 89-reset safe window rather than just
#: under it -- a rebuild costs a second and an unnoticed corrupt sample costs the sweep.
RESETS_PER_ENV = 40

#: The joint names, imported rather than spelled out: a hardcoded copy is what broke the
#: first version of this script, and the names are A3's own vocabulary.
ARM_NAMES = tuple(name for name in ARM_JOINTS)
LEFT_NAMES = tuple(name for name in ARM_JOINTS[:7])


def measure(scene, seeds: list[int], *, randomize: bool) -> dict[str, np.ndarray]:
    """Reset once per seed and record the layout and start pose.

    ``randomize`` selects the two configurations that matter and they are not
    interchangeable: ``True`` is what the training data is a sample of, ``False`` is what
    an evaluation episode actually runs.
    """

    options = (
        {"randomize_scene": True, "randomize_cookies": False}
        if randomize
        else {"randomize_scene": False, "randomize_cookies": False}
    )
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    home: list[np.ndarray] = []

    env = None
    try:
        for index, seed in enumerate(seeds):
            if env is None or (RESETS_PER_ENV and index % RESETS_PER_ENV == 0):
                if env is not None:
                    env.close()
                env = scene.build_env(0.0, 0.0)
            env.reset(seed=seed, options=options)
            source.append(env.data.xpos[env._source_bin_body].copy())
            target.append(env.data.xpos[env._target_bin_body].copy())
            home.append(
                np.asarray([env.data.qpos[env._qpos_ids[name]] for name in ARM_NAMES])
            )

        return {
            "source": np.asarray(source),
            "target": np.asarray(target),
            "home": np.asarray(home),
        }
    finally:
        if env is not None:
            env.close()


def describe(name: str, data: np.ndarray, labels: tuple[str, ...]) -> None:
    print(f"  {name}  (n={len(data)})")
    for index, label in enumerate(labels):
        column = data[:, index]
        print(
            f"    {label:<10} mean {column.mean():+.4f}  sd {column.std():.4f}  "
            f"range [{column.min():+.4f}, {column.max():+.4f}]  "
            f"span {column.max() - column.min():.4f}"
        )


def share_inside(
    eval_values: np.ndarray, training: np.ndarray, labels: tuple[str, ...]
) -> None:
    """For each axis, is the evaluation's value inside the training range?

    Printed per evaluation seed rather than as a count, because the informative answer
    is *which* seed is outside and by how much.  A layout 2 mm outside a 90 mm training
    range is a rounding difference; one 60 mm outside is a different task.
    """

    low = training.min(axis=0)
    high = training.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    print(f"\n  per-seed position relative to the training range, in units of that range")
    for index, values in enumerate(eval_values):
        outside = np.where((values < low) | (values > high))[0]
        worst = float(
            np.max(np.maximum((low - values) / span, (values - high) / span))
        )
        verdict = "inside" if outside.size == 0 else f"outside on {[labels[i] for i in outside]}"
        print(f"    seed {index}: {verdict}, worst overshoot {worst * 100:+.1f}% of a range")


def outside_fraction(values: np.ndarray, training: np.ndarray) -> tuple[int, float]:
    """How many of `values` fall outside the training set's per-axis min/max.

    Reported as a count and a worst-case overshoot *in units of the training span*, which
    is the only form that is interpretable: a layout 2 mm outside a 90 mm range and one
    60 mm outside are both "outside" and are not the same finding.
    """

    low, high = training.min(axis=0), training.max(axis=0)
    span = np.maximum(high - low, 1e-12)
    outside = ((values < low) | (values > high)).any(axis=1)
    worst = float(
        np.max(np.maximum((low - values) / span, (values - high) / span).max(axis=1))
    )
    return int(outside.sum()), worst


def main() -> int:
    scene = registered_scenes()["a3_cookie_same_column"]()
    print("scene: a3_cookie_same_column")

    print("\nsampling the training seeds (randomisation on, as collection ran it) ...",
          flush=True)
    training = measure(scene, TRAINING_SEEDS, randomize=True)
    print("sampling the canonical layout (randomisation OFF, as the CLI evaluates) ...",
          flush=True)
    canonical = measure(scene, EVAL_SEEDS, randomize=False)
    print("sampling the evaluation seeds with randomisation forced on, for contrast ...",
          flush=True)
    sampled = measure(scene, EVAL_SEEDS, randomize=True)

    print("\n" + "=" * 74)
    print("1. WHAT TRAINING VARIES  (measured from the environment)")
    print("=" * 74)
    describe("training layouts (seeds 2000, 2020, ...)", training["source"], ("x", "y", "z"))
    describe("  target bin", training["target"], ("x", "y", "z"))
    describe("  arm start pose", training["home"], LEFT_NAMES)

    randomization = getattr(scene, "env_config", None)
    randomization = getattr(randomization, "randomization", None)
    if randomization is not None:
        print("\n  declared in configs/cookie_same_column.yaml:")
        for name in ("source_bin_x_m", "source_bin_y_m", "source_bin_yaw_rad",
                     "target_bin_x_m", "target_bin_y_m", "target_bin_yaw_rad",
                     "arm_home_rad"):
            axis = getattr(randomization, name, None)
            if axis is not None:
                print(f"    {name:<22} [{axis.low:+.4f}, {axis.high:+.4f}]")

    # The arm-home jitter is the one axis with a closed-form answer, so it is checked
    # rather than described: a uniform draw on +/-h has standard deviation h/sqrt(3).
    spread = training["home"].std(axis=0)
    print(f"\n  arm home sd per joint: {np.round(spread, 4).tolist()}")
    if randomization is not None and randomization.arm_home_rad.high > 0:
        half = float(randomization.arm_home_rad.high)
        expected = half / np.sqrt(3)
        ratio = spread.mean() / expected
        print(f"    a uniform draw on +/-{half} has sd {expected:.4f}"
              f"  ->  measured/theory {ratio:.3f}")
        # The check that would have caught the corrupted sweep.  A uniform draw has a
        # closed-form spread, so a sample that is three times too wide is not noise -- it is
        # resets that did not take, and it is exactly what a too-long single-environment run
        # produces.  Stating it here means a future change to RESETS_PER_ENV fails loudly.
        if not 0.7 < ratio < 1.4:
            print(f"    !! measured spread is {ratio:.2f}x the theoretical value, which means")
            print("       some resets did not apply the jittered home.  Check RESETS_PER_ENV")
            print("       against the environment's safe reset count before trusting anything")
            print("       else on this page.")
        stuck = int((np.abs(training["home"]).max(axis=1) < 1e-6).sum())
        if stuck:
            print(f"    !! {stuck}/{len(training['home'])} samples have every joint at 0.0 --"
                  " those resets left the arm at the model default")

    print("\n" + "=" * 74)
    print("2. WHAT TRAINING DOES NOT VARY")
    print("=" * 74)
    print("  Nothing above measures these, because they are identical in every episode")
    print("  and therefore contribute no variation at all:")
    print("    * the Cookie layout inside each bin: count, spacing, orientation, order")
    print("    * the bins' size, wall height, slot lattice, colour")
    print("    * the camera poses, the lighting, the physics parameters")
    print("    * the visual style -- collection uses the default `plain`")
    print("  `randomize_cookies` is off and has to be for this scene: a 2 mm per-Cookie")
    print("  jitter is wider than the 2.5 mm gaps the batch insertion aims at, and the")
    print("  expert fails immediately with it on.  So the variation is millimetres of")
    print("  rigid translation on a fixed arrangement -- real, but narrow.")

    print("\n" + "=" * 74)
    print("3. WHERE EVALUATION SITS")
    print("=" * 74)
    print("\n  The CLI runs the CANONICAL layout; the seed does not move it:")
    describe("canonical source bin", canonical["source"], ("x", "y", "z"))
    spread_canonical = float(canonical["source"].std(axis=0).max())
    print(f"    -> spread across seeds {list(EVAL_SEEDS)}: {spread_canonical:.2e} m")
    if spread_canonical < 1e-9:
        print("       identical: all three evaluation episodes run the same scene")
    else:
        print("       not identical -- something else in the scene is seeded")

    low, high = training["source"].min(axis=0), training["source"].max(axis=0)
    print(f"\n  training source-bin range : x [{low[0]:+.4f}, {high[0]:+.4f}]  "
          f"y [{low[1]:+.4f}, {high[1]:+.4f}]")
    print(f"  canonical source bin      : x {canonical['source'][0][0]:+.4f}  "
          f"y {canonical['source'][0][1]:+.4f}")
    count, worst = outside_fraction(canonical["source"], training["source"])
    print(f"    -> outside the training range: {count}/{len(canonical['source'])}"
          f"   (worst overshoot {worst * 100:+.1f}% of a range)")

    print(f"\n  For contrast, if `--randomize` were passed:")
    count_r, worst_r = outside_fraction(sampled["source"], training["source"])
    print(f"    seeds {list(EVAL_SEEDS)}: outside {count_r}/{len(sampled['source'])}"
          f"   (worst {worst_r * 100:+.1f}%)")

    # The distance comparison is the one that bears on transfer: how far is an evaluated
    # layout from the training set, against how far two training episodes are apart?
    def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([np.linalg.norm(b - value, axis=1) for value in a])

    within = distances(training["source"][:100], training["source"][100:])
    canonical_gap = distances(canonical["source"][:1], training["source"])
    sampled_gap = distances(sampled["source"], training["source"])
    print("\n  distance from a source-bin position to the training set:")
    print(f"    training episode  vs training episode: mean {within.mean() * 1000:>5.1f} mm  "
          f"p95 {np.percentile(within, 95) * 1000:>5.1f} mm")
    print(f"    canonical layout  vs training        : mean {canonical_gap.mean() * 1000:>5.1f} mm")
    print(f"    sampled eval seeds vs training       : mean {sampled_gap.mean() * 1000:>5.1f} mm")
    ratio = canonical_gap.mean() / within.mean()
    print(f"\n    canonical distance / training spread: {ratio:.2f}x")
    if ratio <= 1.5:
        print("    -> the evaluated layout sits as close to the training set as two training")
        print("       episodes do to each other.")
    else:
        print("    -> the evaluated layout is further from the training set than training")
        print("       episodes are from each other.")

    # The canonical layout is ONE POINT, and with randomisation off nothing about it is
    # redrawn: measured, all three seeds give the nominal home exactly (1.294467, 0.986556,
    # ...) and the same bin positions to 2.8e-17 m.  So a three-seed comparison runs one
    # configuration three times, and the only variation between episodes is the policy's own
    # -- which for a stochastic policy like pi0.5 is real, and for a deterministic one would
    # be nothing at all.
    #
    # That is worth saying because a three-seed table reads like three samples of a scene.
    # It is three draws of one policy in one scene, which is the right setting for comparing
    # models against each other and the wrong one for asking how a policy handles layouts it
    # was not trained on.  ``--randomize`` is what answers the second question.
    print()
    print("  What the three seeds vary: nothing about the scene.  With randomisation off")
    print("  the bins, the Cookie arrangement, the target lattice AND the arm's start pose")
    print("  are identical in every episode -- measured, the three seeds give the nominal")
    print("  home exactly.  A three-seed table is one configuration run three times, and")
    print("  the only variation between episodes is the policy's own randomness (which for")
    print("  a stochastic policy like pi0.5 is real, and for a deterministic one is zero).")
    print()
    print("  That is the right setting for comparing models on one ruler, and the wrong one")
    print("  for asking how a policy handles layouts it was not trained on -- `--randomize`")
    print("  is what answers that, and it lands inside the training range too.")
    return 0
