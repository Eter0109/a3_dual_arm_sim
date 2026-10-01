"""Run the 2x10 expert without recording or loading a neural policy."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.sim.model import write_generated_xml
from a3_dual_arm_sim.sim.randomization import choose_column
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan, parse_first_grasp
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.workflows.cookie_2x10_execution import Cookie2x10EpisodeExecution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--randomization-config",
        type=Path,
        help="Independent 2x10 YAML; defaults to configs/randomization_2x10.yaml",
    )
    parser.add_argument(
        "--first-grasp",
        type=parse_first_grasp,
        default=None,
        help="Optional override of YAML first_grasp",
    )
    parser.add_argument("--source-column", choices=("1", "2", "3", "4", "random"), default=None)
    parser.add_argument("--profile", default=None, help="Optional override of YAML profile")
    parser.add_argument("--dry-run", action="store_true", help="Show settings and prompt only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-model", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    settings, first_grasp = load_cookie_2x10_settings(args.randomization_config)
    if args.first_grasp is not None:
        first_grasp = args.first_grasp
    source_column = args.source_column or settings["source_column"]
    profile = args.profile or settings["profile"]
    if profile not in settings["profiles"]:
        parser.error(f"profile {profile!r} is not defined in the configuration")
    if args.dry_run:
        column = choose_column(source_column, args.seed)
        plan = Cookie2x10Plan.from_seed(first_grasp, args.seed)
        print(
            json.dumps(
                {
                    "profile": profile,
                    "source_column": column,
                    "source_column_mode": source_column,
                    "seed": args.seed,
                    **plan.metadata(),
                    "prompt": plan.prompt(column),
                },
                indent=2,
            )
        )
        return 0
    runner = Cookie2x10EpisodeExecution(
        args.config,
        first_grasp=first_grasp,
        source_column=source_column,
        profile=profile,
        randomization_settings=settings,
        max_steps=args.max_steps,
        render=args.render,
        randomize_boxes=not args.no_randomization,
        randomize_cookies=not args.no_randomization,
    )
    if args.write_model:
        write_generated_xml(args.write_model, runner.config, scene="cookie_transfer")
        print(f"Saved independent 2x10 model to {args.write_model}")
        return 0
    result = runner.run_episode("variable_grasp", args.seed)
    payload = asdict(result)
    print(json.dumps(payload, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
