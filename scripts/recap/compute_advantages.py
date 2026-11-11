#!/usr/bin/env python3
"""Turn value predictions into per-frame advantage labels for RECAP's conditioned training.

The formula, and the one place the units matter
----------------------------------------------
    A_t = sum(r_{t..t+N-1}) + V(o_{t+N}) - V(o_t)

The reward is a **Cookie count** (the running-maximum difference, §9.78) and the head predicts a
value on the `[-1, 0]` atom scale.  Adding them without converting is a unit error that produces
a number with the right shape and no meaning, so the value is converted back to Cookies
(`(v + 1) * required`) before the sum.  `A_t > 0` then reads directly as "this window gained
Cookies", which is what a reader can check by eye.

N is in **cached frames, not control steps.**  The feature cache is strided (§extract), so a
10-frame lookahead at stride 5 spans 50 control steps -- about three action chunks.  The plan
specified N = 10 against `execute = 16`, which is the same order of magnitude, and this script
prints the real span so the number is never in doubt.

Labels
------
`positive` if `A_t` is above the batch's 70th percentile, which puts 30 % of frames in the
positive class.  A percentile rather than an absolute threshold because the scale of `A` shifts
with episode quality: an absolute cut would label "the best step in a bad episode" negative.

Usage:
    python scripts/recap/compute_advantages.py --dataset /root/autodl-tmp/datasets/a3_recap_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "recap"))

from value_model import HeadConfig, ValueHead, expected_value, value_to_count  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--head", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"))
    parser.add_argument("--lookahead", type=int, default=10, help="N, in cached frames")
    parser.add_argument(
        "--advantage-mode",
        choices=("hindsight", "recap"),
        default="hindsight",
        help=(
            "`recap` is the value-bootstrapped `sum(r) + V(t+N) - V(t)` used by the reference "
            "implementations, kept for comparison.  `hindsight` uses the episode's own future, "
            "`Phi_peak - Phi_running(t)`, which is available because every trace runs to "
            "completion -- and which measures +76pp of success/failure separation where `recap` "
            "measures -4.6pp (backwards) on this data."
        ),
    )
    parser.add_argument(
        "--time-buckets",
        type=int,
        default=10,
        help="normalized-time buckets for the per-bucket percentile in `hindsight` mode",
    )
    parser.add_argument(
        "--time-guard",
        type=float,
        default=0.9,
        help=(
            "no negative labels past this normalized time.  A successful episode ends by holding "
            "still at its peak, and without the guard those frames are labelled negative -- i.e. "
            "the policy is told that finishing the task is a mistake."
        ),
    )
    parser.add_argument("--positive-quantile", type=float, default=0.7)
    parser.add_argument(
        "--deadband-fraction",
        type=float,
        default=0.025,
        help=(
            "fraction of the target scale below which `|A|` is called neutral.  Expressed as a "
            "fraction rather than an absolute because the scale is not fixed: 20 under the old "
            "count-only target and 31 under the layered potential, and a constant would mean "
            "different things on each.  0.025 reproduces the 0.5-Cookie deadband that measured "
            "correct in both directions on the count target (positive: 19.3 %% success vs "
            "13.7 %% failure; negative: 0.8 %% vs 4.8 %%)."
        ),
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=None,
        help="absolute override for --deadband-fraction, in the target's own units",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_value import build_frame_table, load_history_encoder  # noqa: F401  (loader reuse)

    dataset = args.dataset
    features_dir = args.features or (dataset / "features_v9")
    labels_path = args.labels or (dataset / "value_labels.parquet")
    head_path = args.head or (dataset / "value_head" / "head.pt")
    output = args.output or (dataset / "advantages.parquet")

    labels = pq.read_table(labels_path).to_pandas()
    required = float(labels["required_cookies"].iloc[0])
    # The value head was trained on `value_target / scale - 1`, and with the layered potential the
    # ceiling is `MAX_POTENTIAL` (31.0) rather than the Cookie count (20).  Deriving it from the
    # labels is what `train_value.py` does, so doing it the same way keeps the two commensurable:
    # `rewards` are potential differences, and a value term scaled by 20/31 would under-weight
    # them by a third -- which silently distorts every advantage rather than failing.
    scale = float(np.ceil(labels["value_target"].to_numpy(dtype=np.float32).max()))
    if scale <= 0:
        raise SystemExit(f"value_target has no positive range; cannot rescale the value term")
    print(f"[adv] target scale {scale:.0f} (rewards and values are both in these units)", flush=True)
    rewards_by_episode = {i: np.asarray(r, dtype=np.float64) for i, r in enumerate(labels["rewards"])}
    target_by_episode = labels["value_target"].to_numpy(dtype=np.float32)

    written = np.load(features_dir / "written.npy")
    rows = np.flatnonzero(written)
    manifest_path = features_dir / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    stride = int(manifest.get("stride", 1) or 1)

    features = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32)
    width = features.shape[1]
    image_dim = (width - 2048) // 3
    image_mean = features[:, : 3 * image_dim].reshape(-1, 3, image_dim)
    language_mean = features[:, 3 * image_dim :]

    episode, frame, state, history = build_frame_table(dataset, rows, args.checkpoint, args.device)

    head = ValueHead(HeadConfig(image_dim=image_dim, language_dim=language_mean.shape[-1])).to(args.device)
    head.load_state_dict(torch.load(head_path, map_location=args.device))
    head.eval()

    predictions = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 8192):
            stop = start + 8192
            logits = head(
                torch.from_numpy(image_mean[start:stop]).to(args.device),
                torch.from_numpy(language_mean[start:stop]).to(args.device),
                torch.from_numpy(state[start:stop]).to(args.device),
                torch.from_numpy(history[start:stop]).to(args.device),
                torch.zeros(min(stop, len(rows)) - start, 2, device=args.device),
            )
            predictions[start:stop] = expected_value(logits).cpu().numpy()

    # Back to the label-file's own scale, so the reward term and the value term are commensurable.
    values = value_to_count(torch.from_numpy(predictions), scale).numpy()
    print(f"[adv] {len(rows)} cached frames, stride {stride}, lookahead {args.lookahead}"
          f" (= {args.lookahead * stride} control steps)")

    # Position of each cached row inside its own episode, so the lookahead resolves within the
    # episode instead of running into the next one.  `order` sorts by (episode, frame); the rank
    # of a row among its episode's rows is then `index - first_index_of_that_episode`.
    order = np.lexsort((frame, episode))
    episode_sorted = episode[order]
    _, first_index, counts = np.unique(episode_sorted, return_index=True, return_counts=True)
    rank = np.arange(len(rows)) - np.repeat(first_index, counts)
    position = np.empty(len(rows), dtype=np.int64)
    position[order] = rank

    rows_of_episode: dict[int, list[int]] = {}
    for idx in order:
        rows_of_episode.setdefault(int(episode[idx]), []).append(int(idx))

    advantages = np.full(len(rows), np.nan, dtype=np.float64)
    for i in range(len(rows)):
        ep = int(episode[i])
        ep_rewards = rewards_by_episode.get(ep)
        if ep_rewards is None:
            continue
        sequence = rows_of_episode[ep]
        # Clamp the lookahead to the episode's last cached frame; a window that would run past the
        # end has no `V(o_{t+N})` and using the next episode's would be a fabricated future.
        j = sequence[min(int(position[i]) + args.lookahead, len(sequence) - 1)]
        start = int(frame[i])
        stop = int(frame[j])
        reward_sum = float(ep_rewards[start:stop].sum())
        advantages[i] = reward_sum + values[j] - values[i]

    finite = np.isfinite(advantages)

    # --- hindsight advantages -------------------------------------------------------
    # The value-bootstrapped advantage above is measurably degenerate on this task, and the
    # measurement is worth recording because the arithmetic looks fine:
    #
    #   * rewards are non-zero on 2557 of 552789 frames -- 6.4 per episode;
    #   * `V` predicts the episode's *final* potential, so two frames 50 steps apart receive
    #     nearly the same prediction and `V(t+N) - V(t) ~ 0`;
    #   * hence `A ~ (reward in the next 2.5 s)`, which is zero for ~80 % of frames.
    #
    # Measured discrimination at the 70th percentile: success 26.3 % vs failure 30.9 %, i.e.
    # **backwards**.  A policy conditioned on that label learns to prefer failing frames.
    #
    # The fix uses what the traces already contain.  Every episode was recorded to completion, so
    # its future is known -- there is no need to bootstrap an estimate of it:
    #
    #   headroom(t) = Phi_peak(episode) - Phi_running(t)
    #
    # "how much more potential this trajectory went on to reach".  Measured: success 88.0 % vs
    # failure 12.2 % in the positive class, and it survives the obvious confound -- headroom
    # falls monotonically with time, but `Spearman(trel, success) = +0.000`, and within **every**
    # time quintile headroom separates success from failure by +5 to +13 potential points.  The
    # partial correlation controlling for normalized time (+0.599) is higher than the raw one
    # (+0.417), so time is not what is being measured.
    #
    # A per-time-bucket percentile is used for the positive class rather than a global cut, for
    # the same reason the count-based threshold had to become a deadband: a global threshold on a
    # quantity that decays with time just selects early frames.  Bucketing makes the label mean
    # "for this phase of the task, this frame had better-than-typical prospects".
    hindsight_positive = np.zeros(len(rows), dtype=bool)
    hindsight_negative = np.zeros(len(rows), dtype=bool)
    headroom = np.full(len(rows), np.nan, dtype=np.float64)
    if args.advantage_mode == "hindsight":
        potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]
        lengths = np.array([len(p) for p in potentials], dtype=np.int64)
        for i in range(len(rows)):
            ep = int(episode[i])
            if ep >= len(potentials) or len(potentials[ep]) == 0:
                continue
            path = potentials[ep]
            headroom[i] = float(path[-1]) - float(path[min(int(frame[i]), len(path) - 1)])

        relative_time = frame / np.maximum(lengths[episode] - 1, 1)
        buckets = np.minimum((relative_time * args.time_buckets).astype(int), args.time_buckets - 1)
        for bucket in range(args.time_buckets):
            members = (buckets == bucket) & np.isfinite(headroom)
            if members.sum() < 20:
                continue
            cut = float(np.percentile(headroom[members], 100 * args.positive_quantile))
            hindsight_positive[members] = headroom[members] >= cut
        # A frame already at its trajectory's peak has nothing ahead of it.  Mid-episode that is a
        # stall; in the final tenth it is the hold that a successful episode is *supposed* to
        # perform, so it is left neutral rather than taught as bad -- otherwise the policy would
        # be told that finishing the task is a mistake.
        hindsight_positive &= headroom > 0
        hindsight_negative = (headroom <= 0) & (relative_time < args.time_guard)
        advantages = np.where(np.isfinite(headroom), headroom, advantages)
        finite = np.isfinite(advantages)
        print(f"[adv] mode hindsight: headroom = Phi_peak - Phi_running(t),"
              f" positive cut at the {100 * args.positive_quantile:.0f}th percentile **within each"
              f" of {args.time_buckets} time buckets**, negative = at the peak, guarded to"
              f" relative time < {args.time_guard}", flush=True)

    # Two labellings, because the first one is measuring the wrong thing and the second is the
    # fix.  Both are written so the difference is inspectable rather than asserted.
    #
    # **Global percentile** -- the obvious implementation, and the one both upstream projects
    # use.  It is biased by episode length: a successful episode spends most of its frames in a
    # "everything is placed, nothing new" state where `A_t ~ 0`, so those frames fall below a
    # global cut and are labelled negative.  Measured on this data, the global rule put **23.3 %
    # of successful-episode frames** in the positive class against **31.7 % of failed-episode
    # frames** -- backwards, even though the per-episode mean advantage was correctly higher for
    # successes (+0.98 vs +0.44).  Conditioning a policy on that label teaches "frames spent
    # carrying are good", which is a property of the trajectory's phase, not of its quality.
    #
    # **Per-episode percentile** -- take the top `positive_quantile` *within each episode* and
    # union them.  Every episode then contributes the same fraction of positive frames, so the
    # label can no longer reward an episode for being long or penalise it for finishing early.
    # The threshold is no longer a single number, which is why it is written per frame.
    threshold = float(np.percentile(advantages[finite], 100 * args.positive_quantile))
    positive_global = advantages >= threshold

    positive_episode = np.zeros_like(positive_global)
    per_episode_threshold = np.full(len(advantages), np.nan, dtype=np.float64)
    for ep in np.unique(episode[finite]):
        members = np.flatnonzero(finite & (episode == ep))
        if len(members) < 2:
            # A single-frame episode has no ranking to take a percentile of; call it positive
            # so it is not silently dropped from training.
            positive_episode[members] = True
            per_episode_threshold[members] = advantages[members]
            continue
        cut = float(np.percentile(advantages[members], 100 * args.positive_quantile))
        per_episode_threshold[members] = cut
        positive_episode[members] = advantages[members] >= cut

    for name, mask in (("global", positive_global), ("per-episode", positive_episode)):
        print(f"[adv] {name:12s} threshold"
              + (f" {threshold:+.3f}" if name == "global" else " (per episode)")
              + f" -> {int(mask.sum())}/{int(finite.sum())} positive"
              f" ({100 * mask.sum() / max(finite.sum(), 1):.1f} %)")

    # **Deadband** -- the one to train on.  Neither percentile rule survives inspection:
    #
    # `A` is bimodal.  Half the frames sit at `A ~ 0` because the value model predicts the same
    # final cookie count 50 control steps later, and the tail is quantised in whole cookies
    # (3, 6, 9 ...).  Both percentile cuts land *inside the zero spike* -- the median per-episode
    # threshold measured +0.0005 cookie -- so they are ranking numerical noise.  The symptom is
    # visible without any model: between frames 5 apart (0.25 s), the sign of the label flips
    # 14.1 % of the time, and frames the rule calls positive are *more* common in failed episodes
    # (32.2 %) than in successful ones (23.4 %).
    #
    # A deadband fixes both by refusing to label frames whose advantage is smaller than the
    # scoring can resolve.  Those frames are not "negative", they are uninformative, and
    # leaving them unconditional is the honest encoding.  The cost is that most frames end up
    # neutral, which is a property of the signal rather than a defect of the threshold.
    deadband = (
        float(args.deadband)
        if args.deadband is not None
        else float(args.deadband_fraction) * scale
    )
    if args.advantage_mode == "hindsight":
        # The headroom labelling already carries its own positive/negative sets, and a deadband on
        # its scale would be a second, unrelated threshold.  Keep the deadband masks populated for
        # the report and the parquet's compatibility columns, but train on the hindsight ones.
        positive_deadband = hindsight_positive
        negative_deadband = hindsight_negative
    else:
        positive_deadband = finite & (advantages > deadband)
        negative_deadband = finite & (advantages < -deadband)
    neutral = finite & ~positive_deadband & ~negative_deadband
    print(f"[adv] positive {int(positive_deadband.sum())}"
          f" ({100 * positive_deadband.sum() / max(finite.sum(), 1):.1f} %), negative"
          f" {int(negative_deadband.sum())}"
          f" ({100 * negative_deadband.sum() / max(finite.sum(), 1):.1f} %), neutral"
          f" {int(neutral.sum())} ({100 * neutral.sum() / max(finite.sum(), 1):.1f} %)")

    # The check that separates a correct label from a merely plausible one: successful episodes
    # must carry a strictly higher positive rate than failed ones.  The global rule fails this.
    success_by_episode = labels["success"].to_numpy()
    episode_success = success_by_episode[episode]
    for name, mask in (
        ("global", positive_global),
        ("per-episode", positive_episode),
        ("deadband", positive_deadband),
    ):
        rate_ok = 100 * mask[episode_success].mean()
        rate_bad = 100 * mask[~episode_success].mean()
        verdict = "OK" if rate_ok > rate_bad else "BACKWARDS"
        print(f"[adv] {name:12s} positive rate: success {rate_ok:.1f} % vs failure"
              f" {rate_bad:.1f} %  {verdict}")
    # The negative class has to point the other way, or the two conditions say the same thing.
    rate_ok = 100 * negative_deadband[episode_success].mean()
    rate_bad = 100 * negative_deadband[~episode_success].mean()
    verdict = "OK" if rate_ok < rate_bad else "BACKWARDS"
    print(f"[adv] {'deadband -':12s} negative rate: success {rate_ok:.1f} % vs failure"
          f" {rate_bad:.1f} %  {verdict}")

    table = pa.table(
        {
            "dataset_row": rows.astype(np.int64),
            "episode_index": episode.astype(np.int64),
            "frame_index": frame.astype(np.int64),
            "value": values.astype(np.float32),
            "advantage": advantages.astype(np.float32),
            "positive_global": positive_global,
            # The two to train on.  `neutral` is implied: neither flag is set, so the frame keeps
            # its unconditioned sentence and acts as the CFG dropout branch.  The percentile
            # variants are kept beside them so a future run can check this reasoning rather than
            # trust it.
            "positive": positive_deadband,
            "negative": negative_deadband,
            "positive_episode": positive_episode,
            "headroom": headroom.astype(np.float32),
            "episode_threshold": per_episode_threshold.astype(np.float32),
            "value_target": target_by_episode[episode].astype(np.float32),
        }
    )
    pq.write_table(table, output)
    (output.parent / "advantages.json").write_text(
        json.dumps(
            {
                "frames": int(len(rows)),
                "stride": stride,
                "lookahead_frames": args.lookahead,
                "lookahead_control_steps": args.lookahead * stride,
                "threshold": threshold,
                "positive_quantile": args.positive_quantile,
                "positive_frames": int(positive_episode.sum()),
                "positive_frames_global_rule": int(positive_global.sum()),
                "rate_success_percent": float(100 * positive_episode[episode_success].mean()),
                "rate_failure_percent": float(100 * positive_episode[~episode_success].mean()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[adv] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
