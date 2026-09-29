"""Summarize count transitions and expert phases without inventing policy phases."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for path in args.root.rglob("trace.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            continue
        events = []
        previous = None
        for row in rows:
            state = (
                row["after"]["cookies_in_source"],
                row["after"]["cookies_in_target"],
                row["expert_phase"],
            )
            if state != previous:
                events.append(
                    {
                        "step": row["step"],
                        "source": state[0],
                        "target": state[1],
                        "expert_phase": state[2],
                    }
                )
                previous = state
        source = [r["after"]["cookies_in_source"] for r in rows]
        target = [r["after"]["cookies_in_target"] for r in rows]
        summary[str(path.relative_to(args.root))] = {
            "frames": len(rows),
            "events": events,
            "final_source": source[-1],
            "final_target": target[-1],
            "max_target": max(target),
            "phase_limit": "Policy phases require video review; counts are not a phase classifier.",
        }
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        axes[0].plot(source, label="source count")
        axes[0].plot(target, label="target count")
        axes[0].legend()
        axes[1].plot(
            [r["observation_before"]["observation.state"][7] for r in rows],
            label="measured gripper",
        )
        axes[1].plot([r["applied_action"][7] for r in rows], label="applied gripper")
        axes[1].legend()
        axes[1].set_xlabel("control step")
        fig.tight_layout()
        fig.savefig(path.parent / "counts_gripper.png")
        plt.close(fig)
    (args.root / "trace_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
