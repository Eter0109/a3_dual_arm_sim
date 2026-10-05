"""Does the observation contain enough information to do this task?

The question this answers is the one that decides every downstream choice: when the
policy's error sits at 0.0185 rad and the task needs 0.0075, is that because the *model*
failed to extract what is in the pixels, or because the pixels do not contain it?

The instrument is deliberately not a VLA.  It is a **linear probe and a tiny MLP** on
frozen features, evaluated on episodes the probe never saw:

    features            error (rad, max-abs)
    state only          <- the y of a lookup table on the joint angles
    state + vision      <- if this is much lower, the pixels carry the missing half

Comparison against the numbers already measured on this scene:

    hold still (predict the state)          0.0215
    kNN on the state, cross-episode         0.0156 .. 0.0185
    best fine-tuned pi0.5, teacher forced   0.0185
    expert fails completely at injected     0.0075

The features come from the **vision tower of the fine-tuned checkpoint itself** -- the
exact tensors the policy's action head attends to.  Anything the tower can see, the
policy could in principle see; a probe that decodes the action better than the policy
does therefore localises the failure in the action head's use of the features rather
than in the images.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

# π0.5 resizes with aspect-preserving padding and normalises [0,1] -> [-1,1] before the
# tower.  Both steps are copied here rather than called, because the point of the probe is
# to see what the tower is handed, and a call into the policy would hide it behind the
# policy's own image bookkeeping.
TOWER_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/datasets/a3_cookie_one_pi05"))
    parser.add_argument("--ckpt", default="/root/autodl-tmp/pi05-runs/a3_one_base/checkpoints/004000/pretrained_model")
    parser.add_argument("--cache", type=Path, default=Path("/root/autodl-tmp/logs/vision_features.npz"))
    parser.add_argument("--stride", type=int, default=1, help="frame stride while encoding")
    parser.add_argument("--test-episodes", type=int, default=60)
    parser.add_argument("--split-seed", type=int, default=0,
                        help="which episodes become the held-out set; report more than one")
    parser.add_argument("--pickle-device", default="cuda")
    parser.add_argument("--skip-vision", action="store_true", help="reuse the cache and probe only")
    parser.add_argument("--mlp-epochs", type=int, default=400)
    return parser.parse_args()


def _tower_step(tower, pending: list, device: str, torch, F) -> np.ndarray:
    """One batch through the tower.

    The resize and the [0,1] -> [-1,1] normalisation happen here, on the GPU and for the
    whole batch at once, rather than per frame on the host.  Doing it per frame with
    ``antialias=True`` was costing more than the forward pass -- 36 frames/s, which is
    forty minutes per camera over this dataset -- and the arithmetic is identical.
    """

    batch = torch.stack(pending).to(device).float().div_(255.0)
    if batch.shape[-2:] != TOWER_SIZE:
        batch = F.interpolate(batch, size=TOWER_SIZE, mode="bilinear",
                              align_corners=False, antialias=True)
    batch = batch.mul_(2.0).sub_(1.0)
    hidden = tower(pixel_values=batch).last_hidden_state
    return hidden.mean(dim=1).float().cpu().numpy()


def encode(root: Path, ckpt: str, stride: int, device: str) -> dict[str, np.ndarray]:
    """Mean-pooled vision-tower features for every frame of both cameras."""

    import torch
    import torch.nn.functional as F
    import av
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(ckpt).to(device).eval()
    tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    # The action head is irrelevant here and holding a second copy of 3B weights costs
    # memory for nothing, so everything except the tower is dropped.
    del policy
    torch.cuda.empty_cache()

    out: dict[str, np.ndarray] = {}
    for key in ("base_0_rgb", "left_wrist_0_rgb"):
        files = sorted(glob.glob(str(root / "videos" / f"observation.images.{key}" / "chunk-*" / "*.mp4")))
        feats: list[np.ndarray] = []
        started = time.time()
        pending: list = []
        # A running counter rather than `enumerate` per file: a camera's frames are split
        # across several mp4 files, and an index that restarts at zero would mis-phase the
        # stride on every file after the first -- silently pairing frames with the wrong
        # labels whenever `stride > 1`.
        index = 0
        with torch.no_grad():
            for path in files:
                for frame in av.open(path).decode(video=0):
                    index += 1
                    if (index - 1) % stride:
                        continue
                    rgb = frame.to_ndarray(format="rgb24")
                    pending.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
                    if len(pending) >= 256:
                        feats.append(_tower_step(tower, pending, device, torch, F))
                        pending.clear()
                        if index % 20000 < 256:
                            print(f"    {key} {index} 帧  {time.time()-started:.0f}s", flush=True)
            if pending:
                feats.append(_tower_step(tower, pending, device, torch, F))
        out[key] = np.concatenate(feats)
        print(f"  {key}: {out[key].shape} in {time.time()-started:.0f}s", flush=True)
    return out


def standardise(train: np.ndarray, test: np.ndarray):
    mean, std = train.mean(0), train.std(0) + 1e-6
    return (train - mean) / std, (test - mean) / std


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge with an intercept, returning the pieces needed to score a test set.

    The target needs centring as much as the features do, and for a reason that is easy to
    miss: an action is an absolute joint position whose dataset mean is 1.39 rad, not 0.
    Fitting only standardised features leaves the prediction centred on zero and the whole
    probe off by the mean of the thing it is predicting -- which reads as "the observation
    carries no information" (error 1.39 rad, worse than the 0.363 of predicting the dataset
    mean) rather than as a missing intercept.
    """

    x_mean, y_mean = X.mean(0), y.mean(0)
    centred = X - x_mean
    gram = centred.T @ centred
    gram.flat[:: len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, centred.T @ (y - y_mean))
    return weights, x_mean, y_mean


def score(fitted, X: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    weights, x_mean, y_mean = fitted
    pred = (X - x_mean) @ weights + y_mean
    e = np.abs(pred - y).max(axis=1)
    return float(e.mean()), float(np.median(e)), pred


def evaluate(name: str, Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
             alphas: tuple[float, ...]) -> tuple[float, np.ndarray, float]:
    """Pick alpha on a held-out slice of the *training* episodes, then report test error.

    Tuning on the test set would make every number here flattering, and the whole point is
    a lower bound on what the observation supports.
    """

    cut = int(len(Xtr) * 0.85)
    Xa, Xb, ya, yb = Xtr[:cut], Xtr[cut:], ytr[:cut], ytr[cut:]
    best, chosen = np.inf, alphas[0]
    for alpha in alphas:
        value, _, _ = score(fit_ridge(Xa, ya, alpha), Xb, yb)
        if value < best:
            best, chosen = value, alpha
    mean, median, pred = score(fit_ridge(Xtr, ytr, chosen), Xte, yte)
    print(f"  {name:<34} alpha={chosen:<10.3g} 误差 mean {mean:.4f}  median {median:.4f}")
    return mean, pred, chosen


def main() -> int:
    args = parse_args()

    if args.cache.exists() and args.skip_vision:
        saved = np.load(args.cache)
        vision = {k: saved[k] for k in saved.files}
        print(f"从缓存读取视觉特征: {args.cache}")
    else:
        print("用微调检查点自己的视觉塔编码所有帧 ...", flush=True)
        vision = encode(args.root, args.ckpt, args.stride, args.pickle_device)
        np.savez(args.cache, **vision)
        print(f"已缓存到 {args.cache}")

    import pandas as pd

    table = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
         for f in sorted(args.root.glob("data/**/*.parquet"))],
        ignore_index=True,
    )
    episodes = table["episode_index"].to_numpy()
    action = np.stack([np.asarray(v, dtype=np.float64) for v in table["action"]])[:, :8]
    state = np.stack([np.asarray(v, dtype=np.float64) for v in table["observation.state"]])[:, :8]

    # The stride was applied while encoding, so the labels have to be strided identically
    # or every row is compared against the wrong action -- which would look like a
    # catastrophic result rather than like an off-by-one.
    keep = np.arange(0, len(action), args.stride)
    vision = {k: v[: len(keep)] for k, v in vision.items()}
    episodes, action, state = episodes[keep], action[keep], state[keep]

    rng = np.random.default_rng(args.split_seed)
    pool = np.unique(episodes)
    rng.shuffle(pool)
    test_episodes = set(pool[: args.test_episodes].tolist())
    te = np.array([e in test_episodes for e in episodes])
    tr = ~te
    print(f"\n划分种子 {args.split_seed}   训练 {tr.sum()} 帧 / {len(pool)-args.test_episodes} 集    "
          f"测试 {te.sum()} 帧 / {args.test_episodes} 集")

    front, wrist = vision["base_0_rgb"], vision["left_wrist_0_rgb"]
    print(f"视觉特征维度 前视 {front.shape[1]}  腕部 {wrist.shape[1]}\n")

    blocks: dict[str, np.ndarray] = {}
    for name, raw in (("state", state), ("front", front), ("wrist", wrist)):
        a, b = standardise(raw[tr], raw[te])
        full = np.empty_like(raw)
        full[tr], full[te] = a, b
        blocks[name] = full

    alphas = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)

    print("【线性探针 — 岭回归，跨 episode】")
    results: dict[str, np.ndarray] = {}
    for name, parts in (
        ("只有状态 (8)", ("state",)),
        ("只有前视视觉塔特征", ("front",)),
        ("只有腕部视觉塔特征", ("wrist",)),
        ("状态 + 前视", ("state", "front")),
        ("状态 + 腕部", ("state", "wrist")),
        ("状态 + 两路视觉", ("state", "front", "wrist")),
    ):
        X = np.hstack([blocks[p] for p in parts])
        _, pred, _ = evaluate(name, X[tr], X[te], action[tr], action[te], alphas)
        results[name] = pred

    hold = np.abs(state[te] - action[te]).max(axis=1)
    print(f"\n  {'基线「原地不动」':<33} 误差 mean {hold.mean():.4f}  median {np.median(hold):.4f}")
    print(f"  {'基线「训练集均值动作」':<31} 误差 mean "
          f"{np.abs(action[tr].mean(0)-action[te]).max(axis=1).mean():.4f}")
    print("\n  参照：")
    print(f"    kNN 只用状态（跨 episode）          0.0156 .. 0.0185")
    print(f"    最好 VLA（教师强制）                0.0185")
    print(f"    专家失败门槛（注入噪声即 0/4）      0.0075")

    # ---- MLP：岭回归只能看到线性可解码的信息，若图像信息是非线性的会漏掉
    if args.mlp_epochs:
        import torch

        device = args.pickle_device
        def mlp(Xtr, Xte, ytr, yte, seed=0):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(
                torch.nn.Linear(Xtr.shape[1], 512), torch.nn.ReLU(),
                torch.nn.Linear(512, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, ytr.shape[1]),
            ).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            xa = torch.from_numpy(Xtr).float().to(device)
            ya = torch.from_numpy(ytr).float().to(device)
            xb = torch.from_numpy(Xte).float().to(device)
            n = len(xa)
            for epoch in range(args.mlp_epochs):
                perm = torch.randperm(n, device=device)
                for start in range(0, n, 4096):
                    idx = perm[start:start+4096]
                    opt.zero_grad()
                    loss = torch.nn.functional.mse_loss(net(xa[idx]), ya[idx])
                    loss.backward(); opt.step()
            with torch.no_grad():
                pred = net(xb).float().cpu().numpy()
            return pred

        # Held-out slice of the training episodes for early-stopping-free reporting: the
        # test episodes stay untouched, so the comparison between "state" and "state+vision"
        # is still the thing being measured.
        cut = int(tr.sum() * 0.9)
        print(f"\n【小 MLP — 2 隐层，{args.mlp_epochs} 轮，跨 episode】")
        for name, parts in (("只有状态 (8)", ("state",)),
                            ("状态 + 两路视觉", ("state", "front", "wrist")),
                            ("只有两路视觉", ("front", "wrist"))):
            X = np.hstack([blocks[p] for p in parts])
            Xtr, ytr = X[tr], action[tr]
            pred = mlp(Xtr[:cut], X[te], ytr[:cut], action[te])
            e = np.abs(pred - action[te]).max(axis=1)
            print(f"  {name:<34} 误差 mean {e.mean():.4f}  median {np.median(e):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
