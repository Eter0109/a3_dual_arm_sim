#!/usr/bin/env python3
"""Does the history latent actually carry task progress?  Offline, from one checkpoint.

The training log prints the auxiliary MSE every 500 steps, but that number cannot be read
across steps: its target column is the *batch mean* of the label, which swings from -0.09
to -0.75 as sampling changes, so a loss of 0.36 on a batch whose targets sit at -0.67 is a
different thing from 0.36 on one that sits at -0.08.  A run can look like it is not
learning while the head is in fact tracking the label, and vice versa.

What this measures instead is R^2 = 1 - MSE / Var(target) over a large sample: how much of
the label's variance the latent explains.  Two numbers make it interpretable:

    R^2 > 0.5   the latent carries progress -- a history channel is worth having
    R^2 ~ 0     the encoder learned nothing that the auxiliary head can read

It also reports the per-dimension spread of the prediction: a head that has collapsed onto
a constant scores a small MSE whenever the batch happens to sit near that constant, and a
near-zero spread is what that looks like.

Only `history_encoder.*` and `history_proj.*` are loaded, so this runs on CPU in seconds and
does not touch the GPU a training run is using.

    python scripts/check_history_latent.py --checkpoint <pretrained_model> --data <dataset>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import history_encoder as history  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=200_000, help="Frames to sample")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    import torch
    from safetensors import safe_open

    settings = history.settings()
    encoder = history.build(None, 2048)
    projection = history.project(torch.zeros(settings["width"]), 2048)
    wanted = {k: v for k, v in encoder.state_dict().items()}
    wanted = {**{f"model.history_encoder.{k}": k for k in wanted},
              **{f"model.history_proj.{k}": k for k in projection.state_dict()}}
    loaded = 0
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as handle:
        available = set(handle.keys())
        missing = sorted(set(wanted) - available)
        if missing:
            raise SystemExit(f"checkpoint has no history weights (missing {missing[:3]})")
        state = {}
        encoder_state, projection_state = {}, {}
        for full, short in wanted.items():
            tensor = handle.get_tensor(full).to(args.device)
            if full.startswith("model.history_encoder."):
                encoder_state[short] = tensor
            else:
                projection_state[short] = tensor
            loaded += 1
    encoder.load_state_dict(encoder_state)
    projection.load_state_dict(projection_state)
    encoder.to(args.device).eval()
    projection.to(args.device).eval()
    print(f"loaded {loaded} history tensors from {args.checkpoint.name}")

    files = sorted((args.data / "data").glob("chunk-*/*.parquet"))
    # Subsample while streaming, never with a whole-column read.  Arrow indexes a column's
    # child buffers with 32-bit offsets, so `pq.read_table` on `observation.history` fails with
    # `OSError: List index overflow` once rows x width passes 2^31 -- 312470 x 480 = 1.5e8 is
    # fine, 312470 x 9600 = 3.0e9 is not.  Only `--limit` frames are needed, so this costs
    # nothing and works at any window width.  Same limit, same fix as the writer in
    # `add_history_window.py`.
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    step = max(1, total // args.limit)

    def as_matrix(column) -> np.ndarray:
        """A `fixed_size_list<float>[w]` column as an `(n, w)` array.

        `np.asarray` on a FixedSizeListArray raises "setting an array element with a
        sequence"; the child values have to be flattened explicitly.  `flatten()` applies the
        list offsets and returns the underlying child buffer, so this stays close to free.
        """
        width = column.type.list_size
        return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), width)

    history_chunks: list[np.ndarray] = []
    progress_chunks: list[np.ndarray] = []
    row = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=2048, columns=["observation.history", "observation.progress"]
        ):
            index = [i for i in range(batch.num_rows) if (row + i) % step == 0]
            row += batch.num_rows
            if index:
                history_chunks.append(as_matrix(batch.column(0))[index].astype(np.float32))
                progress_chunks.append(as_matrix(batch.column(1))[index].astype(np.float32))
    window = torch.tensor(np.concatenate(history_chunks), device=args.device)
    target = torch.tensor(np.concatenate(progress_chunks), device=args.device)
    print(f"{len(window)} frames sampled from {total} (every {step})")

    with torch.no_grad():
        mu, logvar = encoder(window, dropout=0.0)
        prediction = encoder.progress(mu)
        token = projection(mu)

    prediction_np = prediction.cpu().numpy()
    target_np = target.cpu().numpy()
    mse = float(np.mean((prediction_np - target_np) ** 2))
    variance = float(np.mean(np.var(target_np, axis=0)))
    r2 = 1.0 - mse / variance if variance > 0 else float("nan")

    print(f"\nMSE           {mse:.5f}")
    print(f"Var(target)   {variance:.5f}")
    print(f"R^2           {r2:+.3f}   (1.0 = perfect, 0.0 = no better than the mean)")
    print(f"\n{'dim':>4} {'target mean':>12} {'pred mean':>10} {'target std':>11} "
          f"{'pred std':>9} {'corr':>7}")
    for index, name in enumerate(("batches", "cookies")):
        t = target_np[:, index]
        p = prediction_np[:, index]
        correlation = float(np.corrcoef(t, p)[0, 1]) if p.std() > 1e-8 else float("nan")
        print(f"{name:>10} {t.mean():>12.4f} {p.mean():>10.4f} {t.std():>11.4f} "
              f"{p.std():>9.4f} {correlation:>7.3f}")

    print(f"\nlatent: |mu| mean {float(mu.abs().mean()):.4f}, "
          f"per-dim std {float(mu.std(dim=0).mean()):.4f}")
    print(f"prefix token (after projection and sqrt scaling): "
          f"std {float(history.scale(token).std()):.4f}  "
          f"(image tokens enter at ~4.2, language at ~9.6)")

    # A latent that only ever occupies a few distinct values is not encoding a continuum;
    # this is the cheapest way to see that from the numbers alone.
    unique = len(np.unique(np.round(prediction_np[:, 0], 3)))
    print(f"distinct predicted batch values (3 dp): {unique}")

    if r2 > 0.5:
        print("\nVERDICT: the latent carries progress.")
    elif r2 > 0.2:
        print("\nVERDICT: partial -- the latent carries some progress, not cleanly.")
    else:
        print("\nVERDICT: the latent does not carry progress; the auxiliary head is at chance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
