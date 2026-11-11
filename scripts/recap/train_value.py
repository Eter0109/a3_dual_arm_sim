#!/usr/bin/env python3
"""Train the RECAP value head on cached features, and apply the Spearman gate.

The gate is the point of this file
----------------------------------
A value function that cannot rank states makes every downstream step wrong **silently**: the
advantages are mislabelled, the policy is trained on them, the loss still falls, and nothing
raises.  So this script refuses to declare success below `--min-spearman` (default 0.6) and says
so explicitly rather than printing a number and moving on.

What it trains
--------------
Only the head.  The encoder is frozen and its per-frame summary was cached by
`extract_value_features.py`; the history latent is recomputed here with the checkpoint's own GRU
(0.5 M parameters, negligible) so that the value function sees the same trajectory summary the
policy does.  **`observation.progress` is deliberately not an input** -- it is derived from the
simulator's Cookie count, it would not exist on a real robot, and a head that reads it scores
beautifully in training and is worthless in deployment (§4.1 of the plan).

Two Spearman numbers are reported
---------------------------------
* **overall** -- predicted value against the episode's final count, over all sampled frames.  This
  is the gate, and it is generous: late in an episode the current count nearly determines the
  final count, so a head that merely counts Cookies scores well.
* **early** -- the same correlation restricted to the first quarter of each episode, where the
  outcome is genuinely uncertain.  This is the number that says whether the head learned
  something a counter could not, and it is reported even though the gate is not applied to it.

Usage:
    python scripts/recap/train_value.py --dataset /root/autodl-tmp/datasets/a3_recap_all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "recap"))

from value_model import (  # noqa: E402
    HeadConfig,
    NUM_ATOMS,
    ValueHead,
    count_to_value,
    distributional_loss,
    expected_value,
)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed directly so scipy is not required.

    `np.argsort(np.argsort(x))` gives ranks; ties are broken arbitrarily rather than averaged,
    which biases the coefficient slightly when many values coincide.  That matters here because
    the target takes only 21 distinct values -- reported alongside a Kendall-free note in the
    output so the reader knows the number is mildly pessimistic at the extremes.
    """

    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator else float("nan")


def load_history_encoder(checkpoint: Path, hidden: int = 2048, steps: int = 10, per_step: int = 48):
    """The checkpoint's own history GRU, weights and all.

    Loaded by prefix rather than through the launcher: only `history_encoder` is needed here,
    and pulling the 2B policy in would cost 6 GB to compute a 128-wide latent.
    """

    import torch
    from safetensors.torch import load_file

    import history_encoder

    settings = history_encoder.settings()
    encoder = history_encoder.build(settings, hidden, steps=steps).eval()
    state = load_file(str(checkpoint / "model.safetensors"))
    prefix = "model.history_encoder."
    subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(subset, strict=False)
    # `out` is the dead parameter removed in §9.81; an older checkpoint still carries it.
    real_missing = [k for k in missing if not k.startswith("out.")]
    if real_missing:
        raise SystemExit(f"history encoder weights missing: {real_missing}")
    del unexpected, torch
    return encoder, settings, per_step


def build_frame_table(dataset: Path, rows: np.ndarray, checkpoint: Path, device: str):
    """Everything the head reads, aligned to `rows` (dataset row indices).

    Read straight from the parquet: the images are already cached, and going through
    `LeRobotDataset` would decode three videos per frame for data that lives in a float column.
    That is the difference between seconds and an hour.
    """

    import torch

    data = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index", "observation.state", "observation.history"],
    )
    episode = np.asarray(data.column("episode_index").to_pylist(), dtype=np.int64)[rows]
    frame = np.asarray(data.column("frame_index").to_pylist(), dtype=np.int64)[rows]
    state = np.stack(data.column("observation.state").to_pylist()).astype(np.float32)[rows]
    history = np.stack(data.column("observation.history").to_pylist()).astype(np.float32)[rows]

    encoder, _settings, per_step = load_history_encoder(checkpoint, steps=history.shape[1] // 48)
    encoder = encoder.to(device)
    latents = np.empty((len(rows), 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 4096):
            chunk = torch.from_numpy(history[start : start + 4096]).to(device)
            # The encoder takes the flat `[B, steps * per_step]` window and does its own
            # reshape, so passing an already-reshaped 3-D tensor fails its width check with a
            # message that reads like a dataset mismatch (48 vs 480) rather than a shape error.
            mu, logvar = encoder(chunk, dropout=0.0)
            # Mean, not a sample: the value function should be a function of the observation.
            latents[start : start + 4096] = (
                encoder.latent(mu, logvar, sample=False).float().cpu().numpy()
            )
    return episode, frame, state, latents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--features", type=Path, default=None, help="defaults to <dataset>/features_v9")
    parser.add_argument("--labels", type=Path, default=None, help="defaults to <dataset>/value_labels.parquet")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--output", type=Path, default=None, help="defaults to <dataset>/value_head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--target-column",
        default="value_target",
        help=(
            "which label column to regress.  `value_target` is the layered potential; "
            "`peak_count_target` is the old count-only peak, kept so the two can be fitted by "
            "this same code and compared -- a Spearman that moves when only the target changes "
            "is evidence about the target, while one that moves with the code is not."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of EPISODES held out")
    parser.add_argument("--min-spearman", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = args.dataset
    features_dir = args.features or (dataset / "features_v9")
    labels_path = args.labels or (dataset / "value_labels.parquet")
    output = args.output or (dataset / "value_head")
    output.mkdir(parents=True, exist_ok=True)

    # --- targets -------------------------------------------------------------------
    labels = pq.read_table(labels_path).to_pandas()
    # `value_target` is the peak count (§9.78); the reward is its running-maximum difference, so
    # the head must predict the peak or the two would speak about different quantities.
    #
    # The label file's **row order is the merged episode order**: `merge_parts.py` appends each
    # part's `episode_labels.jsonl` in the same order it appends its episodes, and it was run
    # twice -- once per root, then over the two roots -- so the concatenation is already global.
    # Reading `episode_index` from the labels would be wrong: those restart at 1 inside every
    # part (the first run of this script looked up ("a3_recap_v9roll", 39) and got a KeyError,
    # because part_01's episode 1 is the *39th* episode in the merged set, not label 39).
    episodes_table = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    lengths = np.asarray(episodes_table.column("length").to_pylist(), dtype=np.int64)
    if len(labels) != len(lengths):
        raise SystemExit(
            f"{len(labels)} label rows against {len(lengths)} episodes; the label file and the "
            "dataset are from different runs"
        )
    # Confirm the grouping rather than trusting it: `part` must be constant inside each run of
    # rows whose length equals that part's episode count.
    target = labels[args.target_column].to_numpy(dtype=np.float32)

    # The target's scale has to come from the label file, not from a constant.  When the reward was
    # `change in running-max(cookies_in_target)` the ceiling was always 20, so `count_to_value(x,
    # 20.0)` was right by coincidence.  Under the layered potential the ceiling is
    # `MAX_POTENTIAL` (31.0: 20 delivered + 20 seated at half weight + 1 for wall coverage), and a
    # hard-coded 20 would map every good target above 1.0 on the `[-1, 0]` scale the atoms cover,
    # clamping exactly the episodes the head is supposed to rank highest.
    #
    # Read it from the data rather than importing `MAX_POTENTIAL`: this script must also be able to
    # fit the old count-only labels for a comparison, and `peak_count_target` maxes at 20.
    scale = float(np.ceil(target.max()))
    if scale <= 0:
        raise SystemExit(f"target has no positive range (max {target.max()}); cannot normalise")
    print(f"[value] target scale (ceiling) {scale:.0f}", flush=True)
    groups = []
    for value in labels["part"]:
        if not groups or groups[-1][0] != value:
            groups.append([value, 0])
        groups[-1][1] += 1
    per_part = labels.groupby("part").size()
    if any(count != int(per_part[name]) for name, count in groups):
        raise SystemExit("label rows are not contiguous per part; positional mapping is unsafe")
    print(f"[value] {len(lengths)} episodes, target mean {target.mean():.2f}"
          f" range {target.min():.0f}..{target.max():.0f}", flush=True)

    # --- features ------------------------------------------------------------------
    written = np.load(features_dir / "written.npy")
    # `index.json` is written by the last shard to finish, so a dry run against a partially
    # extracted cache legitimately has no manifest.  Default rather than fail: the only thing
    # this script needs from it is the stride, which is reported not used.
    manifest_path = features_dir / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    rows = np.flatnonzero(written)
    print(f"[value] {len(rows)} cached frames of {written.size}"
          f" (stride {manifest.get('stride')})", flush=True)

    features = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32)
    width = features.shape[1]
    image_dim = (width - 2048) // 3
    image_mean = features[:, : 3 * image_dim].reshape(-1, 3, image_dim)
    language_mean = features[:, 3 * image_dim :]

    data_episode = np.asarray(
        pq.read_table(
            dataset / "data" / "chunk-000" / "file-000.parquet", columns=["episode_index"]
        ).column("episode_index").to_pylist(),
        dtype=np.int64,
    )[rows]
    episode, frame, state, history = build_frame_table(dataset, rows, args.checkpoint, args.device)

    # --- split by episode ----------------------------------------------------------
    unique = np.unique(data_episode)
    rng.shuffle(unique)
    n_holdout = max(1, int(round(args.holdout * len(unique))))
    holdout_episodes = set(unique[:n_holdout].tolist())
    is_holdout = np.isin(data_episode, list(holdout_episodes))
    print(f"[value] split: {len(unique) - n_holdout} train / {n_holdout} holdout episodes"
          f" ({is_holdout.sum()} held-out frames)", flush=True)

    frame_target = target[data_episode]
    value_target = count_to_value(torch.from_numpy(frame_target), scale).numpy()

    head = ValueHead(HeadConfig(image_dim=image_dim, language_dim=image_mean.shape[-1])).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    tensors = {
        "image": torch.from_numpy(image_mean),
        "language": torch.from_numpy(language_mean),
        "state": torch.from_numpy(state),
        "history": torch.from_numpy(history),
        # Held at zero on purpose: `progress` is privileged and must not be an input.
        "progress": torch.zeros(len(rows), 2, dtype=torch.float32),
        "target": torch.from_numpy(value_target.astype(np.float32)),
    }
    train_index = np.flatnonzero(~is_holdout)
    held_index = np.flatnonzero(is_holdout)

    def run(index: np.ndarray, train: bool) -> tuple[float, np.ndarray]:
        order = index.copy()
        if train:
            rng.shuffle(order)
        total_loss, predictions = 0.0, np.empty(len(order), dtype=np.float32)
        batch = args.batch_size
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            gathered = {k: v[chunk].to(args.device) for k, v in tensors.items()}
            with torch.set_grad_enabled(train):
                logits = head(
                    gathered["image"], gathered["language"], gathered["state"],
                    gathered["history"], gathered["progress"],
                )
                loss = distributional_loss(logits, gathered["target"]).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach()) * len(chunk)
            predictions[start : start + len(chunk)] = expected_value(logits).cpu().numpy()
        return total_loss / max(len(order), 1), predictions

    started = time.monotonic()
    best = {"spearman": -1.0, "epoch": -1, "predictions": None}
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run(train_index, True)
        _, held_predictions = run(held_index, False)
        rho = spearman(held_predictions, frame_target[held_index])
        if rho > best["spearman"]:
            best = {"spearman": rho, "epoch": epoch, "predictions": held_predictions.copy()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  train loss {train_loss:.4f}  holdout spearman {rho:.4f}"
                  f"  ({time.monotonic() - started:.0f}s)", flush=True)
    torch.save(head.state_dict(), output / "head.pt")

    # --- the gate ------------------------------------------------------------------
    held_predictions = best["predictions"]
    held_episode = data_episode[held_index]
    held_target = frame_target[held_index]
    held_frame = frame[held_index]

    overall = spearman(held_predictions, held_target)

    # Early frames only: the first quarter of each episode, where the outcome is still open.
    lengths_by_episode = lengths
    early_mask = held_frame < (0.25 * lengths_by_episode[held_episode])
    early = spearman(held_predictions[early_mask], held_target[early_mask]) if early_mask.sum() > 10 else float("nan")

    # A count baseline: `V = current Cookies in target`, which is what a model that only reads
    # the image and learns to count would produce.  The head must beat it on the early subset or
    # it is not a value function, only a counter.
    #
    # Reported on the target's own scale (`* scale`), which is 31 under the layered potential and
    # 20 under the old count-only one; printing a potential as if it were a Cookie count was the
    # kind of unit slip that makes a report unreadable.
    predictions_count = (held_predictions + 1.0) * scale

    report = {
        "holdout_episodes": n_holdout,
        "held_frames": int(len(held_index)),
        "best_epoch": best["epoch"],
        "spearman_overall": overall,
        "spearman_early": early,
        "early_frames": int(early_mask.sum()),
        "min_spearman": args.min_spearman,
        "stride": manifest.get("stride"),
        "target_scale": scale,
        "predicted_mean": float(predictions_count.mean()),
        "target_mean": float(held_target.mean()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"[value] best epoch {best['epoch']}: holdout Spearman {overall:.4f}"
          f" (early-quarter {early:.4f} on {int(early_mask.sum())} frames)")
    print(f"[value] predicted mean {predictions_count.mean():.2f} Cookies"
          f" vs actual {held_target.mean():.2f}")
    print()

    # The overall number is generous by construction: by the last third of an episode the current
    # count nearly determines the peak, so a head that only counts Cookies passes it.  The early
    # quarter is where the outcome is still open, and it is the number that says whether this is
    # a value function or a counter.  It is reported as a warning rather than a second gate
    # because the plan set one threshold (§4.3), and moving the goalposts after seeing the number
    # is how a gate stops meaning anything.
    if overall >= args.min_spearman and early < 0.3:
        print(f"[value] WARNING: overall {overall:.2f} passes but the early quarter is only"
              f" {early:.2f}.")
        print("        That pattern means the head reproduces the current count and adds little")
        print("        about the future.  RECAP would then mostly re-weight states by a quantity")
        print("        already visible in the image, which is a weaker signal than intended.")

    if overall >= args.min_spearman:
        print(f"[value] GATE PASSED ({overall:.3f} >= {args.min_spearman}) -- proceed to advantages")
        return 0
    print(f"[value] GATE FAILED ({overall:.3f} < {args.min_spearman})")
    print("[value] Do not build advantages on this head.  Per the plan (§4.3) the order is:")
    print("          1. run the leakage test: zero `progress` (already zero) and confirm no change")
    print("          2. widen the head, or cache patch features and learn the pool")
    print("          3. only then consider an independent value VLM")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
