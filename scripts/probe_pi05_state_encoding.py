"""How much does the state encoding actually cost? Measure before repairing.

The state reaches pi0.5 only through the prompt, as space-separated bin indices
produced by `np.digitize(normalised, linspace(-1, 1, 257)[:-1]) - 1`.  The
normalisation divides by `q99 - q01`, which bounds 98% of the data, and nothing
clips the result, so values outside those quantiles land on bin `-1` -- 0.87% of all
state tokens, present in 14% of frames.  One dimension is worse: the right gripper
has `q01 == q99`, so lerobot's `eps` guard divides 0.006 rad of sensor noise by
1e-8 and the bin there is decided by the sign of that noise.

The tempting next step is to fix it.  This measures first, because the fix has a
cost that is easy to overlook: **training saw the same out-of-range tokens**.  The
statistics in the checkpoint are the ones the training run used, so the model was
trained with `-1` in the prompt, and repairing the encoding at inference time would
move the input away from what it learned on.  Whether that trade is worth making is
an empirical question, and this answers it before any of it is spent.

The comparison holds the flow-matching noise fixed across variants.  Sampling noise
is stochastic, so the same prompt normally gives a different chunk each call, and
comparing two encodings without pinning it would measure the sampler, not the state.

    python scripts/probe_pi05_state_encoding.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CHECKPOINT = Path("/root/autodl-tmp/pi05-runs/a3_b1_20k/checkpoints/020000/pretrained_model")
DATASET = Path("/root/autodl-tmp/gr00t-runs/b1/merged")
TOKENIZER_SNAPSHOT = Path("/root/autodl-tmp/hf/hub/models--google--paligemma-3b-pt-224/snapshots")
TASK = "grasp five yellow blocks together from the source bin"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--out", type=Path, default=Path("pi05_state_encoding.json"))
    return parser.parse_args()


def load_tokenizer():
    from transformers import AutoTokenizer

    candidates = sorted(p for p in TOKENIZER_SNAPSHOT.glob("*") if (p / "tokenizer.json").is_file())
    if not candidates:
        raise SystemExit(f"no paligemma tokenizer snapshot under {TOKENIZER_SNAPSHOT}")
    return AutoTokenizer.from_pretrained(str(candidates[0]), local_files_only=True)


def state_variants(raw: np.ndarray, q01: np.ndarray, q99: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, np.ndarray]:
    """Bin indices for the same physical state under three normalisations.

    ``current`` reproduces lerobot exactly, including the eps guard, so it is what
    training and today's inference both see.  ``clipped`` changes only the values
    that fall outside [-1, 1], leaving every in-range bin identical -- the smallest
    possible repair.  ``minmax`` rescales everything to the observed extremes, which
    removes out-of-range values entirely but also moves every bin.
    """
    bins = np.linspace(-1, 1, 257)[:-1]

    span = q99 - q01
    denom = np.where(span == 0, EPS, span)
    current_norm = 2.0 * (raw - q01) / denom - 1.0

    span_mm = hi - lo
    denom_mm = np.where(span_mm == 0, EPS, span_mm)
    minmax_norm = 2.0 * (raw - lo) / denom_mm - 1.0

    return {
        "current": np.digitize(current_norm, bins) - 1,
        "clipped": np.digitize(np.clip(current_norm, -1.0, 1.0), bins) - 1,
        "minmax": np.digitize(minmax_norm, bins) - 1,
    }


def prompt_for(task: str, bins: np.ndarray | None) -> str:
    """lerobot's pi0.5 prompt, or openpi's pi0 prompt when there is no state.

    ``None`` means no state at all, which is what openpi's
    ``discrete_state_input=False`` produces.  That path uses the other branch of
    openpi's tokenizer -- a bare instruction and a newline, no state, no
    ``Action:`` suffix -- and building it correctly matters because the two formats
    are not interchangeable: the pi0.5 form in this model's hands produces
    private-use tokens rather than text.
    """
    if bins is None:
        return f"{task}\n"
    return f"Task: {task}, State: {' '.join(map(str, bins))};\nAction: "


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    import sys

    sys.path.insert(0, "/root/autodl-tmp/a3-collect")
    from adapters import pi05_a3  # type: ignore[import-not-found]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from safetensors.torch import load_file

    tokenizer = load_tokenizer()
    stats = load_file(str(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    q01 = stats["observation.state.q01"].numpy().astype(np.float64)
    q99 = stats["observation.state.q99"].numpy().astype(np.float64)

    files = sorted(glob.glob(str(args.dataset / "data" / "chunk-*" / "*.parquet")))
    all_states = np.stack(pd.concat(
        [pd.read_parquet(f, columns=["observation.state"]) for f in files], ignore_index=True
    )["observation.state"].to_numpy())
    lo, hi = all_states.min(axis=0), all_states.max(axis=0)

    model = pi05_a3.load({"ckpt": str(checkpoint), "device": "cuda"})
    policy = model["policy"]
    dataset = LeRobotDataset("local/a3", root=str(args.dataset), video_backend="pyav")

    class _Req:
        def __init__(self, arrays, task):
            self._arrays, self.task, self.chunk = arrays, task, 50

        def image(self, name):
            return self._arrays[name]

        def vector(self, name):
            return self._arrays[name]

    stride = max(1, len(dataset) // args.frames)
    steps = list(range(0, len(dataset), stride))[: args.frames]
    noise = torch.randn(1, 50, 32, device="cuda")

    rows: list[dict] = []
    for index, step in enumerate(steps):
        frame = dict(dataset[step])
        arrays = {
            f"{name}_view": np.transpose(frame[f"observation.images.{name}"].numpy(), (1, 2, 0)) * 255.0
            for name in ("front", "left_wrist", "right_wrist")
        }
        arrays = {k: v.clip(0, 255).astype(np.uint8) for k, v in arrays.items()}
        raw = np.asarray(frame["observation.state"], dtype=np.float64)
        arrays["observation.state"] = raw

        request = _Req(arrays, args.task)
        batch = pi05_a3._images(request, model)
        batch["observation.state"] = torch.as_tensor(
            pi05_a3._state_vector(request, model)[None, :], dtype=torch.float32
        )
        batch["task"] = [args.task]
        batch = model["preprocess"](batch)
        images, img_masks = policy._preprocess_images(batch)

        variant_bins = state_variants(raw, q01, q99, lo, hi)
        # The information-carrying variants.  ``stale`` uses the next sampled frame's
        # state, encoded the same way, so any difference is attributable to the values
        # rather than to the encoding.
        other = np.asarray(dict(dataset[steps[(index + 1) % len(steps)]])["observation.state"], dtype=np.float64)
        variant_bins["zeroed"] = np.zeros_like(variant_bins["current"])
        variant_bins["stale"] = state_variants(other, q01, q99, lo, hi)["current"]
        variant_bins["no_state"] = None

        chunks: dict[str, np.ndarray] = {}
        for name, bins in variant_bins.items():
            prompt = prompt_for(args.task, bins)
            ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device="cuda")
            masks = torch.ones_like(ids, dtype=torch.bool)
            with torch.inference_mode():
                chunk = policy.model.sample_actions(images, img_masks, ids, masks, noise=noise)
            # `sample_actions` returns the padded width; `predict_action_chunk` trims it
            # to the configured action dimension before the unnormalizer sees it, and
            # the action statistics are that same width.
            chunk = chunk[:, :, : policy.config.output_features["action"].shape[0]]
            processed = model["postprocess"](chunk)[0]
            if hasattr(processed, "detach"):
                processed = processed.detach().float().cpu()
            chunks[name] = np.asarray(processed, dtype=np.float64)

        base = chunks["current"]
        # The within-chunk spread is the natural yardstick: a difference smaller than
        # the motion the chunk itself commands is not something the robot can act on.
        spread = float(np.abs(base - base.mean(axis=0)).max())
        row = {"step": step, "negative_tokens": int((variant_bins["current"] < 0).sum())}
        for name in variant_bins:
            if name == "current":
                continue
            delta = np.abs(chunks[name][:, :7] - base[:, :7])
            row[f"{name}_first_max"] = float(delta[0].max())
            row[f"{name}_chunk_max"] = float(delta.max())
        row["chunk_spread"] = spread
        row["first_action_abs"] = float(np.abs(base[0]).max())
        rows.append(row)

    frame = pd.DataFrame(rows)
    print(f"frames compared: {len(frame)}")
    print(f"frames with an out-of-range state token: {int((frame['negative_tokens'] > 0).sum())}/{len(frame)}")
    print(f"max out-of-range tokens in one frame:    {int(frame['negative_tokens'].max())}")
    print()
    print("left-arm difference from the current encoding, in radians:")
    print(f"{'variant':<12} {'first action':>13} {'whole chunk':>12} {'vs chunk spread':>17}")
    print("-" * 58)
    for name in variant_bins:
        if name == "current":
            continue
        first = float(frame[f"{name}_first_max"].mean())
        whole = float(frame[f"{name}_chunk_max"].mean())
        print(f"{name:<12} {first:>13.5f} {whole:>12.5f} {whole / frame['chunk_spread'].mean():>16.2f}x")
    print()
    print(f"for scale: the exported chunk's own internal spread is {frame['chunk_spread'].mean():.5f} rad,")
    print(f"           and the first commanded action reaches {frame['first_action_abs'].mean():.5f} rad")

    args.out.write_text(json.dumps({"rows": rows, "summary": {
        "frames": len(frame),
        "out_of_range_frames": int((frame["negative_tokens"] > 0).sum()),
        "chunk_spread_mean": float(frame["chunk_spread"].mean()),
        **{f"{n}_chunk_max_mean": float(frame[f"{n}_chunk_max"].mean()) for n in variant_bins if n != "current"},
    }}, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
