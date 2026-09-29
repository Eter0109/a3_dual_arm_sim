"""Lazy, grouped command entry point for A3 simulation and learning workflows."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    help: str
    prefix: tuple[str, ...] = ()


COMMANDS: dict[str, dict[str, Command]] = {
    "sim": {
        "batch": Command("simulation.batch", "Run same/cross/random-column batch experts"),
        "same-column": Command("simulation.same_column", "Run the single-box same-column expert"),
        "transfer": Command("simulation.transfer", "Run the experimental cooperative expert"),
        "control": Command("simulation.control", "Open the manual control panel", ("control",)),
        "model": Command("simulation.control", "Inspect or export the scene model", ("model",)),
        "smoke": Command("simulation.control", "Check a bounded simulation rollout", ("smoke",)),
        "run": Command("simulation.control", "Run a policy plugin", ("run",)),
        "replay": Command("simulation.control", "Replay recorded joint actions", ("replay",)),
    },
    "agent": {"run": Command("agents.run", "Run Planner, Executor and temporal Verifier")},
    "data": {
        "actions": Command("data.actions", "Collect whole-task expert action episodes"),
        "skills": Command("data.skills", "Collect labeled action skill episodes"),
        "verifier": Command("data.verifier", "Collect independently labeled visual windows"),
        "prepare-verifier": Command("data.prepare_verifier", "Audit and split verifier datasets"),
    },
    "train": {
        "skills": Command("training.skills", "Prepare or train a LeRobot skill policy"),
        "verifier": Command("training.verifier", "Train a temporal Verifier LoRA adapter"),
    },
    "eval": {
        "experts": Command("evaluation.experts", "Evaluate batch experts over seeded episodes"),
        "transfer": Command("evaluation.transfer", "Evaluate the cooperative transfer expert"),
        "verifier": Command("evaluation.verifier", "Evaluate visual verification on saved windows"),
    },
    "inspect": {
        "cameras": Command("diagnostics.cameras", "Preview the three RGB cameras"),
        "depth": Command("diagnostics.depth", "Save aligned RGB and metric depth previews"),
        "actions": Command("diagnostics.actions", "Compare recorded expert and policy actions"),
        "environment": Command("diagnostics.environment", "Report installed package compatibility"),
        "planner": Command("diagnostics.planner", "Probe a planner without robot motion"),
        "verifier-sample": Command("diagnostics.verifier_sample", "Describe one saved visual window"),
    },
    "setup": {"models": Command("setup.models", "Download model weights using a chosen endpoint")},
}

GROUP_HELP = {
    "sim": "Simulation, expert demonstrations and manual control",
    "agent": "Planner, action executor and visual verification",
    "data": "Action, skill and temporal verification datasets",
    "train": "Skill-policy and verifier-adapter training",
    "eval": "Independent expert and visual verification evaluation",
    "inspect": "Camera, action, model and environment diagnostics",
    "setup": "Model and dependency preparation",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="a3-sim", description="A3 cookie packing: simulation, data and agent workflows.",
    )
    groups = result.add_subparsers(dest="group", required=True)
    for group_name, commands in COMMANDS.items():
        group = groups.add_parser(group_name, help=GROUP_HELP[group_name])
        subcommands = group.add_subparsers(dest="command", required=True)
        for name, command in commands.items():
            # Command-specific help belongs to the selected module, so top-level
            # and group help never import a simulator, renderer or model stack.
            entry = subcommands.add_parser(name, help=command.help, add_help=False)
            entry.set_defaults(_entry=command)
    return result


def main(argv: list[str] | None = None) -> int:
    args, remaining = parser().parse_known_args(argv)
    command: Command = args._entry
    module = importlib.import_module(f"a3_dual_arm_sim.cli.{command.module}")
    result = module.main([*command.prefix, *remaining])
    return 0 if result is None else int(result)
