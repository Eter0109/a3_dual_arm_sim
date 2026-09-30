"""Score complete, fixed episode grids; never infer unobserved skill labels."""

from collections import Counter, defaultdict


def summarize(report, protocol, split="development"):
    spec = protocol[split]
    expected = {
        (case["profile"], str(case["source_column"]), seed)
        for case in protocol["cases"]
        for seed in range(spec["seed_start"], spec["seed_start"] + spec["episodes"])
    }
    observed = set()
    groups = defaultdict(list)
    failures = Counter()
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Evaluation report has no episode list")  # noqa: TRY004
    for row in episodes:
        key = (row.get("profile"), str(row.get("column_mode")), row.get("seed"))
        if key not in expected or key in observed:
            raise ValueError("Evaluation contains unexpected or duplicate episodes")
        observed.add(key)
        if type(row.get("success")) is not bool:
            raise ValueError("Episode success must be boolean")
        for field in ("score", "steps", "cookies_in_source"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"Episode {field} must be a nonnegative integer")
        if row["score"] > 10 or not 1 <= row["steps"] <= protocol["max_steps"]:
            raise ValueError("Episode score/steps outside benchmark limits")
        if row["success"] and (row["score"] != 10 or row["cookies_in_source"] != 70):
            raise ValueError("Success violates the existing ten-cookie task contract")
        groups[f"{key[0]}/column_{key[1]}"].append(row)
        if not row["success"]:
            failures[row.get("failure_reason") or "unclassified"] += 1
    if observed != expected:
        raise ValueError("Incomplete evaluation; missing episodes must not improve a score")
    n = len(episodes)
    successes = sum(r["success"] for r in episodes)
    return {
        "synthetic": bool(report.get("synthetic", False)),
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "mean_cookies": sum(r["score"] for r in episodes) / n,
        "mean_steps": sum(r["steps"] for r in episodes) / n,
        "by_case": {
            key: {
                "episodes": len(rows),
                "successes": sum(r["success"] for r in rows),
                "success_rate": sum(r["success"] for r in rows) / len(rows),
                "mean_cookies": sum(r["score"] for r in rows) / len(rows),
            }
            for key, rows in sorted(groups.items())
        },
        "failure_types": dict(failures),
        "zero_transfer_failures": sum(not r["success"] and r["score"] == 0 for r in episodes),
        "partial_transfer_failures": sum(not r["success"] and r["score"] > 0 for r in episodes),
        "representative_failures": [
            {
                key: row.get(key)
                for key in (
                    "profile",
                    "source_column",
                    "seed",
                    "score",
                    "steps",
                    "failure_reason",
                    "trace",
                )
            }
            for row in sorted(episodes, key=lambda r: (r["success"], r["score"]))[:5]
            if not row["success"]
        ],
        "unavailable_metrics": ["skill_success", "pose_error_mm", "grasp_slip"],
    }


def is_better(candidate, best):
    # Exact aggregate counts, with placed cookies as the predeclared tiebreaker.
    # Runtime is reported but is deliberately not used to keep a candidate.
    return (candidate["successes"], candidate["mean_cookies"]) > (
        best["successes"],
        best["mean_cookies"],
    )
