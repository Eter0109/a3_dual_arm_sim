"""Did the fine-tuned pi0.5 actually learn to use its cameras?

The question is not whether the training *config* listed the cameras -- it did, and the
checkpoint records them -- but whether the weights depend on them.  A pipeline that
silently fed zeroed or mis-keyed images would produce exactly the behaviour the
closed-loop traces show: a policy that plans coherently from proprioception and the
prompt while being blind to the scene, which cannot finish a task that requires looking.

Two measurements, on one training frame:

``loss``
    `forward()` with the real images against `forward()` with zeros.  If the two losses
    are identical to float noise, the action loss has no gradient path to the pixels and
    the model never learned to look.  This is the strongest statement available, because
    it is the number the fine-tune actually minimised.

``plan``
    The predicted 50-step chunk with real images against zeros, summarised by the same
    ``net/travel`` coherence metric the closed-loop analysis used.  A model that reads its
    cameras should produce a *different* plan when the scene is replaced; one that ignores
    them will produce the same plan.

Why zero and not noise or a shuffled frame
------------------------------------------
Zero is the cleanest null: it is out of distribution for SigLIP (which expects [-1, 1]
after the [-1, 1] rescale, so zero maps to the centre grey), and it cannot accidentally
resemble a real scene.  A shuffled frame would be a stronger perturbation but a muddier
one -- a difference would not distinguish "reads the image" from "reads *which* image".

The comparison is run on several frames, because a single frame could be one where the
cameras genuinely do not matter (a hold) and the answer would be about that frame rather
than about the model.

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/probe_pi05_vision.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

DATASET = "/root/autodl-tmp/gr00t-runs/b1/merged"
CHECKPOINT = "/root/autodl-tmp/pi05-runs/a3_b1_20k/checkpoints/020000/pretrained_model"
ADAPTER = Path("/root/autodl-tmp/a3-collect/adapters/pi05_a3.py")
#: The policy config the dataset's delta timestamps are resolved from.  Read from the same
#: checkpoint that is being probed, so the action horizon under test is the one it trained
#: with rather than a constant that could drift.
POLICY_CONFIG = CHECKPOINT
#: Frames spread across an episode, where the demonstration is reaching, carrying and
#: placing.  A vision test on one pose would be a test of that pose.
FRAMES = (20, 120, 200, 280, 340)
TASK = "grasp five yellow blocks together from the source bin"
#: How many actions the *inference* comparison reads.  The adapter truncates to the
#: manifest's `execute`, and the loss is scored over the model's full horizon regardless --
#: see `main`.
CHUNK = 16


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
        self.episode_id = "vision"
        self.step = 0

    def image(self, name):
        return self._arrays[name]

    def vector(self, name):
        return self._arrays[name]


def coherence(actions: np.ndarray) -> float:
    """net displacement over travel, on the left arm -- the closed-loop metric."""

    arm = actions[:, :7]
    net = float(np.abs(arm[-1] - arm[0]).max())
    travel = float(np.abs(np.diff(arm, axis=0)).sum())
    return net / travel if travel > 1e-9 else 0.0


def to_arrays(frame: dict) -> dict:
    arrays = {}
    for src, dst in (
        ("observation.images.front", "front_view"),
        ("observation.images.left_wrist", "left_wrist_view"),
        ("observation.images.right_wrist", "right_wrist_view"),
    ):
        image = frame[src]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        arrays[dst] = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    state = frame["observation.state"]
    arrays["observation.state"] = (
        state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    ).astype(np.float64)
    return arrays


def loss_for(adapter, model, arrays, *, zero: bool) -> float:
    """`forward()` loss on one frame, optionally with the cameras blanked."""

    import torch as th

    request = _Request(arrays, TASK, CHUNK)
    batch = adapter._images(request, model)
    if zero:
        for key in batch:
            batch[key] = th.zeros_like(batch[key])

    # State and action are assembled the way the adapter's `infer` does it for the
    # observation, and the way a training batch carries the action.  Both must be in place
    # *before* the pipeline runs: the normalizer transforms them, and the action padding
    # to `max_action_dim` happens later, inside `policy.forward` via `prepare_action`.
    # Padding here instead fails with `The size of tensor a (32) must match the size of
    # tensor b (16)` -- the shape-versus-capacity confusion that also made the adapter pad
    # the state, which is why it is named rather than just fixed.
    batch["observation.state"] = th.as_tensor(
        adapter._state_vector(request, model)[None, :], dtype=th.float32
    )
    batch["task"] = [TASK]
    batch["action"] = th.as_tensor(
        np.asarray(arrays["_action"], dtype=np.float32)[None], dtype=th.float32
    )

    ready = model["preprocess"](batch)
    loss, _ = model["policy"].forward(ready)
    return float(loss.item())


def build_dataset():
    """The dataset as the *training run* built it.

    ``delta_timestamps`` is not optional here, and leaving it out was this script's first
    bug.  A bare ``LeRobotDataset`` returns the action **at** a frame, shape ``(16,)``,
    while the trainer passes ``resolve_delta_timestamps(policy_config, meta)`` so each
    sample carries a ``(50, 16)`` chunk -- the checkpoint's ``action_delta_indices`` is
    ``[0..49]``.  The model then builds its attention masks from ``chunk_size``, so a
    16-step action makes the suffix pad mask 16 long against a 50-long att mask and dies
    with ``The size of tensor a (1018) must match the size of tensor b (984)`` -- an error
    that names tensor sizes and not the missing dataset argument that caused them.

    Building it the same way the trainer does is the point: a probe whose data path
    differs from training can report on that difference rather than on the model.
    """

    import lerobot.policies.pi05  # noqa: F401 - registers the `pi05` config subclass
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    config = PreTrainedConfig.from_pretrained(POLICY_CONFIG)
    meta = LeRobotDatasetMetadata("local/a3", root=DATASET)
    delta = resolve_delta_timestamps(config, meta)
    print(f"  action horizon: {len(delta['action'])} steps at {meta.fps} Hz")
    return LeRobotDataset(
        "local/a3", root=DATASET, video_backend="pyav", delta_timestamps=delta
    )

    print("loading checkpoint ...", flush=True)
    model = adapter.load({"ckpt": CHECKPOINT, "device": "cuda"})
    policy = model["policy"]
    policy.train()  # forward() computes the training loss, which needs grad

def main() -> int:
    adapter = load_adapter()
    dataset = build_dataset()

    print("loading checkpoint ...", flush=True)
    model = adapter.load({"ckpt": CHECKPOINT, "device": "cuda"})
    policy = model["policy"]
    policy.train()  # forward() computes the training loss, which needs grad

    # The loss is scored against the full 50-step block, because that is what the model was
    # trained on and its attention masks are built from `chunk_size`: `embed_suffix` makes a
    # 50-long att mask regardless, so a shorter action is rejected inside
    # `make_att_2d_masks` with a tensor-size error that never mentions the horizon.
    chunk = policy.config.chunk_size

    print(f"\n{'frame':>6} | {'loss real':>11} {'loss zero':>11} {'相对差':>9} | "
          f"{'净/行 real':>11} {'净/行 zero':>11} {'计划差':>9}")
    rows = []
    for frame_index in FRAMES:
        frame = dict(dataset[frame_index])
        arrays = to_arrays(frame)
        actions = frame["action"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        arrays["_action"] = np.asarray(actions, dtype=np.float32)[:chunk]

        real = loss_for(adapter, model, arrays, zero=False)
        zero = loss_for(adapter, model, arrays, zero=True)
        relative = abs(zero - real) / max(abs(real), 1e-9)

        request = _Request(arrays, TASK, CHUNK)
        plan_real = adapter.infer(model, request)["actions"]

        # `infer` reads through `_Request`, so blanking has to happen in the arrays.
        blanked = dict(arrays)
        for key in ("front_view", "left_wrist_view", "right_wrist_view"):
            blanked[key] = np.zeros_like(arrays[key])
        plan_zero = adapter.infer(model, _Request(blanked, TASK, CHUNK))["actions"]

        plan_delta = float(np.abs(plan_real - plan_zero).max())
        rows.append((frame_index, real, zero, relative, coherence(plan_real), coherence(plan_zero), plan_delta))
        print(f"{frame_index:>6} | {real:>11.5f} {zero:>11.5f} {relative:>8.2%} | "
              f"{coherence(plan_real):>11.3f} {coherence(plan_zero):>11.3f} {plan_delta:>9.4f}",
              flush=True)

    print()
    relative_mean = float(np.mean([row[3] for row in rows]))
    coherence_real = float(np.mean([row[4] for row in rows]))
    coherence_zero = float(np.mean([row[5] for row in rows]))
    print("=" * 72)
    print(f"  loss with real cameras  {np.mean([row[1] for row in rows]):.5f}")
    print(f"  loss with blank cameras {np.mean([row[2] for row in rows]):.5f}")
    print(f"  relative difference     {relative_mean:.3%}")
    print(f"  plan net/travel   real {coherence_real:.3f}   blank {coherence_zero:.3f}")
    print()
    if relative_mean < 0.01:
        print("  The loss is INSENSITIVE to the cameras.  The action loss has no meaningful")
        print("  gradient path to the pixels, so the fine-tune did not learn to look -- and a")
        print("  policy that cannot see the scene cannot finish a task that requires looking.")
        print("  -> inspect how the images reach the batch, not the training hyperparameters.")
    else:
        print(f"  The loss DEPENDS on the cameras ({relative_mean:.1%} relative change when they")
        print("  are blanked).  The model was trained with vision, so the closed-loop")
        print("  oscillation is not a case of a blind policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
