"""Is the fine-tuned pi0.5's direction flip driven by sampling noise?

The closed-loop traces show the policy reversing direction between every replan: 19 of
24 chunks at ``execute: 16`` and 22 of 23 at ``execute: 50``.  So the oscillation period
is tied to the *replan interval* rather than to the chunk length -- the model turns
around each time it sees a new observation.

This asks the question that decides what to do about it: given **one fixed observation**,
does the model's own 50-step plan have a consistent direction?  Two answers, pointing at
different fixes:

* **Different directions within one observation** -> the sampler decides, and a replan can
turn around without the scene having changed.  Fix: raise ``num_inference_steps``.
* **One direction within an observation** -> the sampler is stable, the policy's map from
  observation to plan is what flips, and inference settings are not the lever.

Note the comparison that makes the answer readable.  A plan's *coherence* is its net
displacement over its own travel (``net/travel``): near 1 means a single deliberate move,
near 0 means it walks out and back.  The closed-loop trace's plans measured 0.04-0.18.
If the same model scores near 1 on observations taken from the demonstration, then the
input distribution, not the sampler, is what differs.

π0.5 is a flow-matching model, so its action head integrates from random noise and is
stochastic by construction.  Two calls on identical input are *expected* to differ a
little; the question is whether they differ in *sign*.

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/probe_pi05_direction.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/a3_dual_arm_sim/scripts")

DATASET = "/root/autodl-tmp/gr00t-runs/b1/merged"
CHECKPOINT = "/root/autodl-tmp/pi05-runs/a3_b1_20k/checkpoints/020000/pretrained_model"
ADAPTER = Path("/root/autodl-tmp/a3-collect/adapters/pi05_a3.py")
#: A few observations from different points in an episode, so the answer is not a
#: property of one pose.  Early, middle and late, where the demonstration is reaching,
#: carrying and placing.
PROBE_STEPS = (20, 200, 340)
#: Samples per observation.  Enough to separate "the direction is random" from "the
#: direction is stable with a little noise"; 12 gives a sign count that is easy to read.
SAMPLES = 12
#: What a plan counts as "moving": below this, the chunk is a hover and its direction is
#: meaningless rather than consistent.
MOVING_THRESHOLD = 0.02


def load_adapter():
    spec = importlib.util.spec_from_file_location("pi05_a3", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Request:
    def __init__(self, arrays, task, chunk):
        self._arrays = arrays
        self.task = task
        self.chunk = chunk
        self.episode_id = "probe"
        self.step = 0

    def image(self, name):
        return self._arrays[name]

    def vector(self, name):
        return self._arrays[name]


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    adapter = load_adapter()
    dataset = LeRobotDataset("local/a3", root=DATASET, video_backend="pyav")

    print("loading checkpoint ...", flush=True)
    model = adapter.load({"ckpt": CHECKPOINT, "device": "cuda"})

    print(f"\n{SAMPLES} samples per observation, identical input each time")
    print(f"a plan counts as moving when its peak net displacement exceeds "
          f"{MOVING_THRESHOLD} rad\n")

    overall_signs: list[int] = []
    per_observation: list[tuple[int, int, int, float]] = []
    for step in PROBE_STEPS:
        frame = dict(dataset[step])
        arrays = {
            "front_view": np.transpose(frame["observation.images.front"].numpy(), (1, 2, 0)) * 255.0,
            "left_wrist_view": np.transpose(frame["observation.images.left_wrist"].numpy(), (1, 2, 0)) * 255.0,
            "right_wrist_view": np.transpose(frame["observation.images.right_wrist"].numpy(), (1, 2, 0)) * 255.0,
            "observation.state": frame["observation.state"].numpy().astype(np.float64),
        }
        arrays = {k: (v.clip(0, 255).astype(np.uint8) if "view" in k else v) for k, v in arrays.items()}
        request = _Request(arrays, "grasp five yellow blocks together from the source bin", 50)

        directions: list[float] = []
        travels: list[float] = []
        for sample in range(SAMPLES):
            actions = adapter.infer(model, request)["actions"]
            # Joint 0 of the left arm, the axis the demonstrations use most.  `net` is the
            # signed end-to-start displacement, so its sign is the plan's direction.
            joint = actions[:, 0]
            directions.append(float(joint[-1] - joint[0]))
            travels.append(float(np.abs(np.diff(joint)).sum()))
            if (sample + 1) % 4 == 0:
                print(f"  step {step:>3}: {sample + 1}/{SAMPLES} sampled", flush=True)

        net = np.asarray(directions)
        trav = np.asarray(travels)
        moving = np.abs(net) > MOVING_THRESHOLD
        if moving.sum() == 0:
            print(f"\n  step {step:>3}  every sample was a hover ({MOVING_THRESHOLD} rad); "
                  "no direction to read")
            per_observation.append((step, 0, 0, float("nan")))
            continue

        forward = int((net[moving] > 0).sum())
        backward = int((net[moving] < 0).sum())
        signs = set(np.sign(net[moving]).astype(int).tolist())
        overall_signs.extend(int(s) for s in signs)
        coherence = float(np.mean(np.abs(net[moving]) / trav[moving]))
        per_observation.append((step, forward, backward, coherence))

        print(f"\n  step {step:>3}  (net displacement of left-arm joint 0 over 50 steps)")
        print(f"    values      {np.round(net, 4).tolist()}")
        print(f"    travel      mean {trav.mean():.3f}")
        print(f"    moving      {int(moving.sum())}/{SAMPLES} samples")
        print(f"    direction   {forward} forward / {backward} backward  -> "
              f"{'CONSISTENT' if len(signs) == 1 else 'BOTH DIRECTIONS'}")
        # Within-sample travel against net displacement: a plan that walks out and back
        # has a wide travel for a small net, whatever its sign.
        print(f"    net/travel  mean {coherence:.3f}  "
              f"({'one deliberate move' if coherence > 0.5 else 'walks out and back'})")

    print("\n" + "=" * 68)
    usable = [row for row in per_observation if row[1] + row[2] > 0]
    inconsistent = [row for row in usable if row[1] and row[2]]
    if not usable:
        print("  Every observation produced only hovers; nothing to conclude.")
        return 1

    print(f"  per observation (direction consistency within ONE fixed observation):")
    for step, forward, backward, coherence in per_observation:
        verdict = "consistent" if not (forward and backward) else "BOTH WAYS"
        print(f"    step {step:>3}  {forward:>2} fwd / {backward:>2} bwd   "
              f"net/travel {coherence:.3f}   {verdict}")
    print()
    print("  NOTE: different observations pointing different ways is expected, and is not")
    print("  a finding -- moving toward a cookie and moving away are opposite plans.  What")
    print("  matters is whether ONE observation gives one answer.")
    print()
    if inconsistent:
        print(f"  {len(inconsistent)}/{len(usable)} observations produced plans in BOTH directions.")
        print("  The sampler decides the direction, so a replan can turn around with the")
        print("  scene unchanged -- which is why the oscillation period tracks the replan")
        print("  interval rather than the chunk length.")
        print("\n  What to try: `num_inference_steps` (config 10 -> 16 or 30).  One run, no")
        print("  retraining, and the same net/travel metric says whether it helped.")
    else:
        coherence = float(np.mean([row[3] for row in usable]))
        print("  EVERY observation gave a single consistent direction, and the plans are")
        print(f"  coherent (mean net/travel {coherence:.3f}).")
        print()
        print("  So the sampler is not the problem: on a fixed input this model reliably")
        print("  produces one deliberate move.  The reversal across chunks is driven by the")
        print("  observation changing, which points away from inference settings.")
        print()
        print(f"  The parts worth comparing: on these demonstration observations the plans")
        print(f"  score net/travel {coherence:.2f}, while the closed-loop trace's plans scored")
        print("  0.04-0.18.  Same model, same sampler -- different input distribution.  That")
        print("  is the hypothesis to test next, not more denoising steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
