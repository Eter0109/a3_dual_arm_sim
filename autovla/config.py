"""Strict experiment/protocol validation and atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

BOUNDS = {
    "lr": (1e-6, 1e-3, float),
    "decay_lr": (1e-7, 1e-4, float),
    "warmup_steps": (0, 2000, int),
    "batch_size": (1, 32, int),
    "n_action_steps": (1, 50, int),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config, steps):
    if not isinstance(config, dict) or set(config) != set(BOUNDS):
        raise ValueError(f"Experiment must contain exactly {list(BOUNDS)}")
    for key, (low, high, kind) in BOUNDS.items():
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
            or (kind is int and type(value) is not int)
        ):
            raise ValueError(f"{key} must be {kind.__name__} in [{low}, {high}]")
    if config["decay_lr"] > config["lr"]:
        raise ValueError("decay_lr must be <= lr")
    if config["warmup_steps"] >= steps:
        raise ValueError("warmup_steps must be less than the fixed training steps")
    return dict(config)


def validate_protocol(protocol):
    expected = {
        "dataset_root",
        "repo_id",
        "base_model",
        "device",
        "train_seed",
        "steps",
        "save_freq",
        "num_workers",
        "offline",
        "scene_config",
        "randomization_config",
        "cases",
        "development",
        "acceptance",
        "max_steps",
        "trace_episodes",
        "max_experiments",
        "max_hours",
        "experiment_timeout_seconds",
        "agent_timeout_seconds",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected:
        raise ValueError(f"Protocol must contain exactly {sorted(expected)}")
    for key in (
        "steps",
        "save_freq",
        "max_steps",
        "max_experiments",
        "experiment_timeout_seconds",
        "agent_timeout_seconds",
    ):
        positive_int(protocol[key], key)
    for key in ("num_workers", "trace_episodes", "train_seed"):
        if type(protocol[key]) is not int or protocol[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if protocol["device"] not in ("cuda", "cpu") or type(protocol["offline"]) is not bool:
        raise ValueError("device must be cuda/cpu and offline must be boolean")
    hours = protocol["max_hours"]
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise ValueError("max_hours must be a finite positive number")  # noqa: TRY004
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("max_hours must be a finite positive number")
    for key in ("dataset_root", "repo_id", "base_model", "scene_config", "randomization_config"):
        if not isinstance(protocol[key], str) or not protocol[key]:
            raise ValueError(f"{key} must be a nonempty string")
    seed_sets = []
    for split in ("development", "acceptance"):
        spec = protocol[split]
        if not isinstance(spec, dict) or set(spec) != {"seed_start", "episodes"}:
            raise ValueError(f"{split} requires seed_start and episodes")
        positive_int(spec["episodes"], f"{split}.episodes")
        if type(spec["seed_start"]) is not int or spec["seed_start"] < 0:
            raise ValueError("seed_start must be a nonnegative integer")
        seed_sets.append(set(range(spec["seed_start"], spec["seed_start"] + spec["episodes"])))
    if seed_sets[0] & seed_sets[1]:
        raise ValueError("Development and acceptance seeds must be disjoint")
    cases = protocol["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"profile", "source_column"}:
            raise ValueError("Each case requires profile and source_column")
        if not isinstance(case["profile"], str) or str(case["source_column"]) not in (
            "1",
            "2",
            "3",
            "4",
            "random",
        ):
            raise ValueError("Invalid profile or source_column")
        key = (case["profile"], str(case["source_column"]))
        if key in seen:
            raise ValueError("Duplicate evaluation case")
        seen.add(key)
    return protocol


def apply_proposal(best, proposal, steps):
    required = {"hypothesis", "change", "expected_effect", "finding", "stop"}
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError(f"Proposal must contain exactly {sorted(required)}")
    for key in ("hypothesis", "expected_effect", "finding"):
        if not isinstance(proposal[key], str) or len(proposal[key]) > 1500:
            raise ValueError(f"{key} must be a string of at most 1500 characters")
    if type(proposal["stop"]) is not bool:
        raise ValueError("stop must be boolean")
    if proposal["stop"]:
        return None
    if not proposal["hypothesis"].strip():
        raise ValueError("A candidate needs a hypothesis")
    change = proposal["change"]
    if not isinstance(change, dict) or set(change) != {"parameter", "value"}:
        raise ValueError("change requires parameter and value")
    parameter = change["parameter"]
    if parameter not in BOUNDS:
        raise ValueError("Parameter is outside the experiment allowlist")
    value = change["value"]
    if BOUNDS[parameter][2] is int and type(value) is float and value.is_integer():
        value = int(value)
    config = validate_config({**best, parameter: value}, steps)
    if config == best:
        raise ValueError("Proposal makes no change")
    return config
