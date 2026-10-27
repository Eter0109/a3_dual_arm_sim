#!/usr/bin/env python3
"""Is the KL term the ceiling on the latent's progress information?

The joint run reaches R^2 = +0.310 at 3000 steps and then flattens: the latent's per-dim
std goes 0.0638 -> 0.0489 and its mean magnitude 0.0968 -> 0.0720 between 1500 and 3000
steps, i.e. KL is winning and compressing the channel while the progress head stops
improving.  The same encoder trained alone with *only* the progress term reaches 0.954, so
the gap is either KL or the flow gradient competing for the same weights.

This separates the two.  It trains the encoder on its own -- no VLA, no flow loss -- under
exactly the joint run's auxiliary objective, and sweeps `kl_weight`.  If a smaller weight
recovers most of the 0.954, the joint run is over-regularised and the fix is a number, not
an architecture.  If it does not, the flow gradient is the cost and no weight will fix it.

    python scripts/probe_kl_ceiling.py --data .../a3_multitask_v6 --steps 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import history_encoder as history  # noqa: E402


def load(data: Path, limit: int):
    import torch

    files = sorted((data / "data").glob("chunk-*/*.parquet"))
    table = pq.read_table(files, columns=["observation.history", "observation.progress"])
    total = len(table)
    step = max(1, total // limit)
    window = np.asarray(table["observation.history"].to_pylist()[::step], dtype=np.float32)
    target = np.asarray(table["observation.progress"].to_pylist()[::step], dtype=np.float32)
    print(f"{len(window)} frames sampled from {total} (every {step})")
    return torch.tensor(window), torch.tensor(target)


def r2_of(pred, target) -> float:
    p, t = pred.cpu().numpy(), target.cpu().numpy()
    variance = float(np.mean(np.var(t, axis=0)))
    return 1.0 - float(np.mean((p - t) ** 2)) / variance


def train_once(window, target, *, kl_weight, progress_weight, lr, steps, seed, batch):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    encoder = history.build(None, 2048).to(window.device)
    optimiser = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad], lr=lr
    )

    n = len(window)
    # Evaluated on a fixed subsample: the whole 40k through the encoder every 200 steps
    # costs more than the training does, and the comparison only needs one consistent
    # measurement.  Training-set R^2 on purpose -- it is the same number the joint run's
    # check_history_latent.py reports, so the two are comparable.
    probe_index = torch.linspace(0, n - 1, min(6000, n)).long().to(window.device)
    for step in range(steps):
        index = torch.randint(0, n, (batch,), device=window.device)
        mu, logvar = encoder(window[index], dropout=0.1)
        latent = encoder.latent(mu, logvar, sample=True)
        progress = F.mse_loss(encoder.progress(latent), target[index])
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = progress_weight * progress + kl_weight * kl
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                mu_eval, _ = encoder(window[probe_index], dropout=0.0)
                held_out = r2_of(encoder.progress(mu_eval), target[probe_index])
                spread = float(mu_eval.std())
                mean_abs = float(mu_eval.abs().mean())
            print(
                f"  step {step:4d}  progress={float(progress.detach()):.4f} "
                f"kl={float(kl.detach()):.4f} R^2={held_out:+.3f} "
                f"mu_std={spread:.4f} |mu|={mean_abs:.4f}",
                flush=True,
            )
    with torch.no_grad():
        mu_eval, _ = encoder(window[probe_index], dropout=0.0)
        return r2_of(encoder.progress(mu_eval), target[probe_index]), float(mu_eval.std())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--kl", type=float, nargs="+", default=[0.0, 0.005, 0.05])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    window, target = load(args.data, args.limit)
    window, target = window.to(device), target.to(device)

    results = []
    for kl_weight in args.kl:
        print(f"\nkl_weight={kl_weight}  (progress_weight=0.1, lr={args.lr}, batch={args.batch})",
              flush=True)
        r2, spread = train_once(
            window, target,
            kl_weight=kl_weight, progress_weight=0.1, lr=args.lr,
            steps=args.steps, seed=args.seed, batch=args.batch,
        )
        results.append((kl_weight, r2, spread))

    print("\nkl_weight   R^2      mu_std")
    for kl_weight, r2, spread in results:
        print(f"{kl_weight:9.3f}  {r2:+.3f}   {spread:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
