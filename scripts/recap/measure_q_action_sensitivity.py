"""Does Q read the action, or only the state?

The question this answers
------------------------
`train_q_value.py` established that `Q(s, a)` ranks frame-level outcomes at 85.6 % pairwise
accuracy on held-out episodes, against a 50 % chance baseline.  But those pairs differed in
**both** state and action, because every frame carries exactly one action.  DSRL needs something
stronger: at one fixed state, does Q prefer one action over another?

If Q's output barely moves when the action is changed and the state is held fixed, then it is a
state-value function wearing a Q's name.  Steering the sampler with it would do nothing -- the
actor's gradient would be flat in the noise, and any apparent improvement would be noise.

How it is measured, without counterfactual data
-----------------------------------------------
There is no recorded data of "one state, several actions", so this constructs the comparison
synthetically, which is legitimate for one specific question: **how much of Q's variance is
attributable to the action term.**  It says nothing about whether the substituted actions are
sensible, only about whether the network's output is sensitive to its second argument.

    sensitivity = fraction of Q's total variance that survives when the state is held fixed

Reported three ways, because they fail differently:

* **action-driven fraction** -- `Var(Q | fixed state)` over `Var(Q)` across everything.  High
  means the action matters; low means Q is a state function.
* **within-state spread relative to its own mean** -- how far apart two actions at one state are.
  This is the number a steering actor can actually exploit.
* **state-vs-action variance decomposition** -- a two-way split, so the two components are
  reported on the same scale instead of being inferred from two separate runs.

What "changing the action" means here
------------------------------------
Actions are taken from *other* frames, so they are in-distribution individually but paired with a
state they never co-occurred with.  A policy that only ever sees matched pairs may score
out-of-distribution pairs arbitrarily.  The mitigation is to also report a **near-state**
control: actions from nearby frames of the same episode, which is a mild perturbation, and
actions from a random frame, which is a strong one.  If the sensitivity is only visible for the
strong perturbation, the effect is likely distribution shift rather than a learned preference.

Usage
-----
    python scripts/recap/measure_q_action_sensitivity.py \
        --q-head /root/autodl-tmp/datasets/a3_recap_all/q_head_v3/q.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--q-head", type=Path, default=None,
                        help="defaults to <dataset>/q_head_v3/q.pt")
    parser.add_argument("--states", type=int, default=200, help="states to probe")
    parser.add_argument("--actions-per-state", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    from torch import nn

    dataset = args.dataset
    head_path = args.q_head or (dataset / "q_head_v3" / "q.pt")
    if not head_path.is_file():
        raise SystemExit(f"no Q head at {head_path}")

    features_dir = dataset / "features_v9"
    written = np.load(features_dir / "written.npy")
    rows = np.flatnonzero(written)
    manifest = json.loads((features_dir / "index.json").read_text())
    stride = int(manifest["stride"])
    features = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32)

    frame_table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["action", "episode_index", "frame_index"],
    ).to_pandas()
    actions = np.stack(frame_table["action"].values).astype(np.float32)[rows][:, :16]
    episodes = frame_table["episode_index"].to_numpy(dtype=np.int64)[rows]
    frames = frame_table["frame_index"].to_numpy(dtype=np.int64)[rows]

    class QNet(nn.Module):
        """Same shape as the trained head; `train_q_value.py` must be kept in step with this."""

        def __init__(self, feature_dim: int, action_dim: int, hidden: int = 1024):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden // 2), nn.SiLU(),
            )
            self.head = nn.Linear(hidden // 2, 1)

        def forward(self, state, action):
            return self.head(self.trunk(torch.cat([state, action], dim=-1))).squeeze(-1)

    device = args.device
    q = QNet(features.shape[1], actions.shape[1]).to(device)
    q.load_state_dict(torch.load(head_path, map_location=device))
    q.eval()

    rng = np.random.default_rng(0)
    probe = rng.choice(len(rows), size=min(args.states, len(rows)), replace=False)

    results = {}
    for label, sampler in (
        ("near", "same_episode_nearby"),   # mild perturbation: same episode, few frames away
        ("far", "random_frame"),           # strong perturbation: any frame
    ):
        within_state_variance = []
        within_state_spread = []
        for index in probe:
            state = torch.from_numpy(features[index])[None].to(device)
            candidate_actions = []
            if sampler == "same_episode_nearby":
                # Frames within +/- 10 cached rows: a real trajectory's local neighbourhood, where
                # the state barely moved, so the action is the main thing that differs.
                local = np.flatnonzero((episodes == episodes[index]) & (np.abs(frames - frames[index]) <= 10 * stride))
                local = local[local != index]
                if len(local) < 2:
                    continue
                pick = rng.choice(local, size=min(args.actions_per_state, len(local)), replace=False)
            else:
                pick = rng.choice(len(rows), size=args.actions_per_state, replace=False)
            candidate_actions = torch.from_numpy(actions[pick]).to(device)
            with torch.no_grad():
                values = q(state.expand(len(pick), -1), candidate_actions).cpu().numpy()
            within_state_variance.append(float(np.var(values)))
            within_state_spread.append(float(values.max() - values.min()))
        results[label] = {
            "within_state_variance": float(np.mean(within_state_variance)) if within_state_variance else float("nan"),
            "within_state_spread": float(np.mean(within_state_spread)) if within_state_spread else float("nan"),
            "probes": len(within_state_variance),
        }

    # Total variance over random (state, action) pairs, for the denominator.
    with torch.no_grad():
        sample = rng.choice(len(rows), size=4096, replace=False)
        sample_actions = rng.choice(len(rows), size=4096, replace=False)
        total = q(torch.from_numpy(features[sample]).to(device),
                  torch.from_numpy(actions[sample_actions]).to(device)).cpu().numpy()
    results["total_variance"] = float(np.var(total))
    results["q_mean"] = float(np.mean(total))
    results["q_std"] = float(np.std(total))

    print(f"[sens] Q head {head_path.name}, {results['near']['probes']} probes")
    print(f"[sens] Q over random pairs: mean {results['q_mean']:+.4f} sd {results['q_std']:.4f}"
          f" (variance {results['total_variance']:.6f})")
    print()
    print(f"{'扰动强度':10s} {'状态内方差':>12s} {'占总体方差':>11s} {'状态内极差':>11s}")
    for label, chinese in (("near", "近邻状态"), ("far", "随机动作")):
        entry = results[label]
        fraction = entry["within_state_variance"] / max(results["total_variance"], 1e-12)
        print(f"{chinese:10s} {entry['within_state_variance']:12.6f} {100 * fraction:10.2f} %"
              f" {entry['within_state_spread']:11.4f}")
        results[label]["fraction_of_total"] = fraction

    print()
    near_fraction = results["near"]["fraction_of_total"]
    far_fraction = results["far"]["fraction_of_total"]
    # The decision.  `near` is the honest number: it perturbs the action while keeping the state
    # in the same neighbourhood, which is what a sampler-steering actor actually does.
    if near_fraction > 0.30:
        verdict = "action_sensitive"
        print("→ 动作敏感度足够（近邻 >30%）。DSRL 的 actor 有可优化的信号，可以做。")
    elif near_fraction > 0.10:
        verdict = "weak"
        print("→ 动作敏感度弱（10–30%）。Q 主要读状态；DSRL 可能只有很小收益，建议停。")
    else:
        verdict = "state_only"
        print("→ Q 几乎是状态函数（<10%）。DSRL 无效：改噪声不会改变它的输出。止步。")
    if far_fraction > 3 * max(near_fraction, 1e-9):
        print("   注意：随机动作的敏感度远高于近邻，说明很大一部分来自分布外位移，"
              "不是学到的偏好。上一条判断应以近邻那列为准。")
        verdict += "_ood_dominated"

    results["verdict"] = verdict
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
