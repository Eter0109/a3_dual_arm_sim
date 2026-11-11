"""Score N independent attempts per scene, and report what choosing the best would give.

Why retrying works, stated precisely
------------------------------------
The policy is a *stochastic* map: `predict_action_chunk` starts its flow-matching ODE from
`x0 ~ N(0, I)`, freshly drawn on every forward pass.  An episode needs 1500/16 = 94 forwards, so
a rollout consumes 94 independent noise draws, and the controller feeds each action back into the
next observation.  The result is that **an episode's score is a random variable conditioned on the
scene**, not a function of it.

Measured on this task (v9, seeds 300-307, three independent repeats):
within-scene standard deviation **3.34 Cookies**, range on a single scene 3-20, and **0 of 8
scenes repeated identically**.  The per-step noise is small -- 0.43 % of the action range -- but
contact dynamics have a positive Lyapunov exponent, so a sub-millimetre difference in where a
Cookie is first touched decides which one is brushed out later.  Same reason a pinball machine is
unpredictable: tiny input error, large output error, deterministic system.

Given that, choosing the best of N attempts has strictly higher expectation than one attempt:

    E[max(R_1 .. R_N)] >= E[R_1],  with equality only if R is degenerate

and it needs no training, no gradient and no critic -- just a way to *score* a finished attempt.

Why this is not cheating
------------------------
The verifier here is the task's own success criterion: Cookies in the target, seated in slots,
touching the walls.  A deployed system can observe exactly that, and "try, check, try again" is
what a human does on a contact-rich task.  So this is a legitimate decision-time procedure rather
than a benchmark trick.

Two things it does NOT do, stated here because they bound the claim:

* **It does not raise the policy's ceiling.**  A scene where the policy never scores above 10
  will still score 10 after N retries.  The gain comes only from scenes whose *upper tail* is
  reachable, so it is worth reporting what fraction of scenes are "improvable" at all.
* **The reset is exact only in simulation.**  Retrying means replaying the same scene from the
  same seed, which is what the harness does.  On hardware the physical reset is approximate, so
  the realisable gain is smaller than what this measures.

How the numbers are reported
----------------------------
Both, always: the per-attempt score (the policy's own quality, unchanged by this) and the
best-of-N score (what a deployment that retries would see).  Reporting only the second would make
"success rate" meaningless -- with N independent attempts, a per-attempt rate p becomes
`1-(1-p)^N`, which says more about N than about the policy.

Usage
-----
Analyse attempts that have already been run:

    python scripts/recap/best_of_n.py --roots div1 div2 div3

Run N fresh attempts per scene and then analyse:

    bash scripts/recap/run_retry_suite.sh 56 3
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path


def load_attempt(tag: str, root: Path) -> dict[int, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / "part_*" / "episode_labels.jsonl"))):
        rows += [json.loads(line) for line in open(path)]
    return {int(row["seed"]): row for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/datasets"))
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="attempt tags; each must be a separate evaluation over the SAME scene seeds",
    )
    parser.add_argument(
        "--score",
        default="cookies_in_target",
        help="label field to maximise; `cookies_in_target` is the task's own count",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    attempts = [load_attempt(tag, args.root / f"eval_{tag}") for tag in args.roots]
    if any(not attempt for attempt in attempts):
        missing = [tag for tag, attempt in zip(args.roots, attempts) if not attempt]
        raise SystemExit(f"no labels for {missing}")

    seeds = sorted(set.intersection(*[set(attempt) for attempt in attempts]))
    if not seeds:
        raise SystemExit("the attempts share no seeds; they are not comparable")
    dropped = {tag: len(attempt) - len(seeds) for tag, attempt in zip(args.roots, attempts)}
    if any(dropped.values()):
        print(f"[retry] common seeds only ({len(seeds)}); per-attempt extras: "
              f"{ {t: d for t, d in dropped.items() if d} }")

    n = len(attempts)
    print(f"[retry] {n} attempt(s) x {len(seeds)} scene(s), maximising `{args.score}`\n")

    # --- per scene ------------------------------------------------------------------
    print(f"{'seed':>5} " + " ".join(f"{'a' + str(i + 1):>5}" for i in range(n)) + f" {'best':>5} {'mean':>6} {'range':>6}")
    per_seed_best, per_seed_mean, per_seed_range = [], [], []
    improvable = 0
    for seed in seeds:
        values = [attempt[seed][args.score] for attempt in attempts]
        best, mean = max(values), st.mean(values)
        spread = best - min(values)
        # "Improvable" means the upper tail is reachable: some attempt did better than the worst.
        # A scene where every attempt scores the same cannot benefit from retrying at all.
        improvable += int(spread > 0)
        per_seed_best.append(best)
        per_seed_mean.append(mean)
        per_seed_range.append(spread)
        print(f"{seed:5d} " + " ".join(f"{v:5.1f}" for v in values) + f" {best:5.1f} {mean:6.2f} {spread:6.1f}")

    # --- distributions ---------------------------------------------------------------
    # `best-of-n` is the mean over scenes of the per-scene maximum, which is exactly what a
    # deployment that runs n attempts and keeps the highest-scoring one would average.
    single_attempt = st.mean(per_seed_mean)
    best_of_n = st.mean(per_seed_best)
    mean_within_std = st.mean([st.stdev([a[s][args.score] for a in attempts]) for s in seeds]) if n > 1 else 0.0

    print()
    print(f"单次尝试均值        {single_attempt:.2f}")
    print(f"best-of-{n} 均值      {best_of_n:.2f}   增益 {best_of_n - single_attempt:+.2f}")
    print(f"场景内标准差        {mean_within_std:.2f}")
    print(f"可改善场景          {improvable}/{len(seeds)}"
          f" ({100 * improvable / len(seeds):.0f} %)  — 其余场景所有尝试同分，重试无用")
    print(f"场景极差 中位/最大   {st.median(per_seed_range):.1f} / {max(per_seed_range):.1f}")

    # --- strict success --------------------------------------------------------------
    # Reported per attempt AND under retry, because the retry number is a function of n as much
    # as of the policy: a per-attempt rate p becomes 1-(1-p)^n.
    if "success" in attempts[0][seeds[0]]:
        per_attempt = st.mean([sum(1 for s in seeds if a[s]["success"]) for a in attempts]) / len(seeds)
        any_success = st.mean([any(a[s]["success"] for a in attempts) for s in seeds])
        print()
        print(f"严格成功率 单次     {100 * per_attempt:.1f} %")
        print(f"严格成功率 best-of-{n} {100 * any_success:.1f} %"
              f"   ({'1-(1-p)^n 的独立近似 = '}{100 * (1 - (1 - per_attempt) ** n):.1f} %)")

    # --- what the gain costs ----------------------------------------------------------
    print()
    print(f"代价：{n} 倍评估时间（每集 ~12 分钟，{n} 次 = {12 * n} 分钟/场景）")
    print("注意：这是用算力换稳定性，不改变策略本身；单次尝试的质量没有提升。")

    # Marginal value of one more attempt, from the observed tail.  With only n attempts the
    # estimate is coarse, and it is stated as such rather than extrapolated confidently.
    if n >= 2:
        print(f"\n（仅 {n} 次重复，best-of-{n} 的估计有约 ±{mean_within_std / n ** 0.5:.2f} 的不确定度）")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "attempts": n,
            "scenes": len(seeds),
            "single_attempt_mean": single_attempt,
            "best_of_n_mean": best_of_n,
            "gain": best_of_n - single_attempt,
            "mean_within_scene_std": mean_within_std,
            "improvable_scenes": improvable,
            "per_scene_best": per_seed_best,
            "per_scene_mean": per_seed_mean,
        }, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
