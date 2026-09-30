"""Does pi0.5's high-level subtask head still work, and what prompt reaches it?

The paper describes a two-level decomposition where one forward pass emits text
(``y^l_1:M``, the subtask) and actions (``y^a_1:H``, the flow-matching expert), and
inference decodes the text first and conditions the actions on it.  lerobot ports
only the low-level half: its prompt is hardcoded to end in ``Action: `` and its loss
has no cross-entropy term at all.

The released ``pi05_base`` weights, though, do carry a text head, and a first probe
saw ``Subtask:`` come out of it.  This script goes after the two things that probe
left open:

*Is the prompt format what decides it?*  The earlier run produced private-use
garbage with the full lerobot prompt (state included, ``Action: `` suffix) and
readable English once the state and suffix were dropped.  One variable was changed
at a time there, so this sweeps the formats systematically and reports what each
one generates.

*Is the state encoding sound?*  The prompt lerobot builds contains a ``-1``, and
``np.digitize(..., linspace(-1, 1, 257)[:-1]) - 1`` can only return 0..255.  A -1
means a normalised value below -1, which is possible whenever the statistics used
are q01/q99 -- those quantiles bound 98% of the data, not all of it -- and
guaranteed when a dimension's q01 equals its q99.  The right gripper in this
dataset does exactly that, because the right arm does not move.

Both parts run on the fine-tuned checkpoint.  Using it rather than the base is not
arbitrary: the backbone is bit-identical between the two, so the text either
produces is the same, and the fine-tuned one is what a deployment would actually
run.

    python probe_pi05_subtask.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path("/root/autodl-tmp/pi05-runs/a3_b1_20k/checkpoints/020000/pretrained_model")
DATASET = Path("/root/autodl-tmp/gr00t-runs/b1/merged")
TOKENIZER_SNAPSHOT = Path("/root/autodl-tmp/hf/hub/models--google--paligemma-3b-pt-224/snapshots")

#: Prompts to try, as (label, template).  `{task}` is the instruction, `{state}` the
#: discretized state string, `{subtask}` a subtask to condition on.  The set spans
#: the three shapes the model could plausibly have been trained on: lerobot's own
#: low-level format, a bare task line, and a high-level format that names the field
#: being asked for.
PROMPT_FORMS = (
    ("lerobot low-level (Task+State+Action)", "Task: {task}, State: {state};\nAction: "),
    ("task only + Action", "Task: {task};\nAction: "),
    ("task only (no suffix)", "Task: {task};\n"),
    ("Task + Subtask field", "Task: {task};\nSubtask: "),
    ("Task + State + Subtask field", "Task: {task}, State: {state};\nSubtask: "),
    ("PaliGemma caption prefix", "caption en\n"),
    ("bare subtask question", "What subtask should the robot do next? "),
    ("task + subtask given, then Action", "Task: {task}, Subtask: {subtask};\nAction: "),
)

#: Instructions to try.  The first is this project's own; the rest are the
#: high-level commands the paper's figures use ("clean the bedroom", "put the
#: dishes in the sink"), which is the level the subtask head was trained to break
#: down.
TASKS = (
    "grasp five yellow blocks together from the source bin",
    "transfer the cookies into the target box",
    "put away the dishes",
    "clean the bedroom",
)

SUBTASK = "pick up the cookies from the bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("pi05_subtask_probe.json"))
    return parser.parse_args()


def load_tokenizer():
    from transformers import AutoTokenizer

    candidates = sorted(p for p in TOKENIZER_SNAPSHOT.glob("*") if (p / "tokenizer.json").is_file())
    if not candidates:
        raise SystemExit(f"no paligemma tokenizer snapshot under {TOKENIZER_SNAPSHOT}")
    return AutoTokenizer.from_pretrained(str(candidates[0]), local_files_only=True)


def quantile_stats(checkpoint: Path) -> dict[str, np.ndarray]:
    """The normalizer's own statistics, straight out of the saved pipeline."""
    from safetensors.torch import load_file

    found = sorted(checkpoint.glob("*normalizer_processor*.safetensors"))
    if not found:
        raise SystemExit(f"no normalizer statistics in {checkpoint}")
    tensors = load_file(str(found[0]))
    out: dict[str, np.ndarray] = {}
    for name in ("observation.state", "action"):
        q01, q99 = tensors.get(f"{name}.q01"), tensors.get(f"{name}.q99")
        if q01 is not None and q99 is not None:
            out[name] = np.stack([q01.numpy().astype(np.float64), q99.numpy().astype(np.float64)])
    return out


def normalise(values: np.ndarray, q01: np.ndarray, q99: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """The same map lerobot's quantile normalizer applies, eps guard included."""
    span = np.where(np.abs(q99 - q01) < eps, eps, q99 - q01)
    return 2.0 * (values - q01) / span - 1.0


def discretize(normalised: np.ndarray) -> np.ndarray:
    """pi0.5's state encoding: 256 uniform bins over [-1, 1].

    No clipping on either side of the division, which is what makes an
    out-of-range normalised value visible: it lands on bin index -1 or 256.
    """
    bins = np.linspace(-1, 1, 257)[:-1]
    return np.digitize(normalised, bins) - 1


def audit_state(dataset_root: Path, stats: dict[str, np.ndarray]) -> dict:
    """How far outside [-1, 1] the state normalisation actually lands.

    Read straight from the parquet files rather than through `LeRobotDataset`.
    The dataset stores state as a column, and indexing it through the dataset
    object also pulls the frame's images, which means decoding video: 4000 frames
    of AV1 took minutes and dominated the probe's runtime for a question that
    needs no pixels at all.
    """
    import glob

    import pandas as pd

    files = sorted(glob.glob(str(dataset_root / "data" / "chunk-*" / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet data under {dataset_root}")
    frames = pd.concat(
        [pd.read_parquet(f, columns=["observation.state"]) for f in files], ignore_index=True
    )
    raw = np.stack(frames["observation.state"].to_numpy())
    q01, q99 = stats["observation.state"]
    normalised = normalise(raw, q01, q99)
    bins = discretize(normalised)

    span = q99 - q01
    degenerate = np.flatnonzero(np.abs(span) < 1e-8)
    below = (normalised < -1.0).sum(axis=0)
    above = (normalised > 1.0).sum(axis=0)
    invalid = (bins < 0) | (bins > 255)
    return {
        "episodes_file_count": len(files),
        "frames": int(len(frames)),
        "degenerate_dims": degenerate.tolist(),
        "spans": span.tolist(),
        "dim_below_minus_one": below.tolist(),
        "dim_above_plus_one": above.tolist(),
        "invalid_bin_tokens": int(invalid.sum()),
        "invalid_dims": np.flatnonzero(invalid.sum(axis=0)).tolist(),
        "normalised_range": [float(normalised.min()), float(normalised.max())],
        "normalised_per_dim": [
            list(np.round(normalised.min(axis=0), 3)),
            list(np.round(normalised.max(axis=0), 3)),
        ],
    }


def generate(policy, batch: dict, prompt_ids: list[int], tokens: int) -> list[int]:
    """Greedy continuation of a prompt, driving the backbone by hand.

    lerobot ships no generation path for pi0.5, so the loop lives here.  Greedy
    because the question is which prompt the model answers coherently; sampling
    would turn one answer into a distribution of them.

    No EOS handling: stopping at EOS would hide the degeneration this probe is
    measuring, since the interesting failure is the model continuing past a
    readable prefix into private-use tokens.
    """
    model = policy.model
    language_model = model.paligemma_with_expert.paligemma.model.language_model
    head = model.paligemma_with_expert.paligemma.lm_head

    images, img_masks = policy._preprocess_images(batch)
    ids = torch.tensor([prompt_ids], device="cuda")
    masks = torch.ones_like(ids, dtype=torch.bool)

    with torch.inference_mode():
        embeddings, pad, _ = model.embed_prefix(images, img_masks, ids, masks)
        out: list[int] = []
        for _ in range(tokens):
            position_ids = torch.cumsum(pad, dim=1) - 1
            hidden = language_model(
                inputs_embeds=embeddings, attention_mask=pad, position_ids=position_ids, use_cache=False
            )
            next_id = int(head(hidden.last_hidden_state)[0, -1].argmax())
            out.append(next_id)
            step = language_model.embed_tokens(torch.tensor([[next_id]], device="cuda"))
            embeddings = torch.cat([embeddings, step * (step.shape[-1] ** 0.5)], dim=1)
            pad = torch.cat([pad, torch.ones((1, 1), dtype=pad.dtype, device="cuda")], dim=1)
    return out


def readable(text: str) -> float:
    """Fraction of characters that are letters, spaces or common punctuation.

    A cheap way to separate "the model is talking" from "the model is emitting
    private-use tokens", which is the distinction this whole probe turns on and
    which a printed string makes hard to judge by eye.
    """
    if not text:
        return 0.0
    okay = sum(1 for c in text if c.isascii() and (c.isalnum() or c in " .,;:'-_/"))
    return okay / len(text)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    import sys

    sys.path.insert(0, "/root/autodl-tmp/a3-collect")
    from adapters import pi05_a3  # type: ignore[import-not-found]

    tokenizer = load_tokenizer()
    stats = quantile_stats(checkpoint)

    print("=== state normalisation audit ===")
    audit = audit_state(args.dataset, stats)
    print(f"  frames             : {audit['frames']}")
    print(f"  degenerate dims    : {audit['degenerate_dims']}  (q01 == q99)")
    print(f"  normalised range   : [{audit['normalised_range'][0]:.3f}, {audit['normalised_range'][1]:.3f}]")
    print(f"  out-of-range tokens: {audit['invalid_bin_tokens']} across dims {audit['invalid_dims']}")
    print(f"  dims below -1      : {[i for i, n in enumerate(audit['dim_below_minus_one']) if n]}")
    print(f"  dims above +1      : {[i for i, n in enumerate(audit['dim_above_plus_one']) if n]}")
    print(flush=True)

    model = pi05_a3.load({"ckpt": str(checkpoint), "device": "cuda"})
    policy = model["policy"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/a3", root=args.dataset, video_backend="pyav")
    frame = dict(dataset[args.step])
    q01, q99 = stats["observation.state"]
    raw_state = np.asarray(frame["observation.state"], dtype=np.float64)
    state_bins = discretize(normalise(raw_state, q01, q99))
    state_str = " ".join(str(v) for v in state_bins)

    images = {
        f"{name}_view": np.transpose(frame[f"observation.images.{name}"].numpy(), (1, 2, 0)) * 255.0
        for name in ("front", "left_wrist", "right_wrist")
    }
    images = {k: v.clip(0, 255).astype(np.uint8) for k, v in images.items()}
    # The adapter reads the state through the same duck-typed request, and it
    # checks the width against the statistics, so it has to be present here or
    # every prompt sweep fails on a KeyError rather than on the thing under test.
    images["observation.state"] = np.asarray(frame["observation.state"], dtype=np.float64)

    class _Req:
        def __init__(self, arrays, task):
            self._arrays, self.task, self.chunk = arrays, task, 50

        def image(self, name):
            return self._arrays[name]

        def vector(self, name):
            return self._arrays[name]

    results = []
    for task in TASKS:
        print(f"=== task: {task!r} ===", flush=True)
        request = _Req(images, task)
        batch = pi05_a3._images(request, model)
        batch["observation.state"] = torch.as_tensor(
            pi05_a3._state_vector(request, model)[None, :], dtype=torch.float32
        )
        batch["task"] = [task]
        batch = model["preprocess"](batch)

        for label, template in PROMPT_FORMS:
            prompt = template.format(task=task, state=state_str, subtask=SUBTASK)
            ids = tokenizer.encode(prompt, add_bos=True)
            generated = generate(policy, batch, ids, args.tokens)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            score = readable(text)
            marker = "OK  " if score > 0.7 else "junk"
            print(f"  [{marker}] {label:<40} {text[:78]!r}", flush=True)
            results.append(
                {
                    "task": task,
                    "form": label,
                    "prompt": prompt,
                    "generated": text,
                    "readable": round(score, 3),
                    "ids": generated,
                }
            )
        print()

    args.out.write_text(json.dumps({"audit": audit, "results": results}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
