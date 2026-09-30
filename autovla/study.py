"""Immutable study inputs and recoverable experiment journals."""

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from .config import digest, read_json, validate_config, validate_protocol, write_json

TEMPLATES = Path(__file__).parent


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_files(root, contract):
    repo = Path(contract["repo"])
    files = {}
    for directory in ("src", "autovla", "configs", "assets", "models"):
        for path in sorted((repo / directory).rglob("*")):
            if path.is_file() and (
                path.suffix in (".py", ".yaml", ".xml", ".urdf", ".STL", ".stl")
                or path.name == "program.md"
            ):
                # Downloaded model folders are fingerprinted separately below.
                if directory == "models" and path.parent != repo / "models":
                    continue
                if "__pycache__" not in path.parts:
                    files["repo/" + path.relative_to(repo).as_posix()] = file_hash(path)
    for name in (
        "contract.json",
        "baseline_config.json",
        "randomization.json",
        "scene.yaml",
        "program.md",
        "proposal.schema.json",
    ):
        files["study/" + name] = file_hash(root / name)
    if contract["mode"] == "live":
        for name, directory in (
            ("dataset", contract["protocol"]["dataset_root"]),
            ("base_model", contract["protocol"]["base_model"]),
        ):
            directory = Path(directory)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and ".cache" not in path.relative_to(directory).parts:
                    files[name + "/" + path.relative_to(directory).as_posix()] = file_hash(path)
    return files


def verify(root):
    contract = read_json(root / "contract.json")
    expected = read_json(root / "seal.json")
    current = protected_files(root, contract)
    if current != expected:
        changed = sorted(
            k for k in current.keys() | expected.keys() if current.get(k) != expected.get(k)
        )
        raise RuntimeError(f"Frozen study inputs changed: {changed[:8]}; start a new study")
    return contract


def init_study(root, *, repo, protocol_path=None, config_path=None, demo=False):
    import yaml

    root, repo = Path(root).resolve(), Path(repo).resolve()
    protocol = validate_protocol(read_json(protocol_path or TEMPLATES / "protocol.json"))
    baseline = validate_config(
        read_json(config_path or TEMPLATES / "experiment.json"), protocol["steps"]
    )
    for key in ("dataset_root", "base_model", "scene_config", "randomization_config"):
        protocol[key] = str((repo / Path(protocol[key]).expanduser()).resolve())
    protected_roots = [repo / name for name in ("src", "autovla", "configs", "assets", "models")]
    protected_roots += [Path(protocol["dataset_root"]), Path(protocol["base_model"])]
    if (
        root == repo
        or repo.is_relative_to(root)
        or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected_roots)
    ):
        raise ValueError("Study output must be separate from code, dataset and base-model inputs")
    randomization = yaml.safe_load(
        Path(protocol["randomization_config"]).read_text(encoding="utf-8")
    )
    for case in protocol["cases"]:
        if case["profile"] not in randomization["profiles"]:
            raise ValueError(f"Undefined randomization profile: {case['profile']}")
    if not demo:
        dataset = Path(protocol["dataset_root"])
        model = Path(protocol["base_model"])
        for path in (
            dataset / "meta/info.json",
            dataset / "a3_episode_metadata.jsonl",
            model / "config.json",
            model / "model.safetensors",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Download/prepare local training inputs first: {path}")
        seeds = set()
        for line in (
            (dataset / "a3_episode_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            if line.strip():
                row = json.loads(line)
                if type(row.get("seed")) is not int:
                    raise ValueError(
                        "Training sidecars need explicit seeds to verify split separation"
                    )
                seeds.add(row["seed"])
        for split in ("development", "acceptance"):
            spec = protocol[split]
            evaluation = set(range(spec["seed_start"], spec["seed_start"] + spec["episodes"]))
            if seeds & evaluation:
                raise ValueError(f"Training and {split} seeds overlap")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "randomization.json", randomization)
    (root / "scene.yaml").write_bytes(Path(protocol["scene_config"]).read_bytes())
    protocol["randomization_config"] = str(root / "randomization.json")
    protocol["scene_config"] = str(root / "scene.yaml")
    for name in ("program.md", "proposal.schema.json"):
        (root / name).write_bytes((TEMPLATES / name).read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    contract = {
        "repo": str(repo),
        "git_commit": commit,
        "mode": "demo" if demo else "live",
        "protocol": protocol,
    }
    write_json(root / "contract.json", contract)
    write_json(root / "baseline_config.json", baseline)
    print("Fingerprinting fixed code, scene, dataset and base model...", flush=True)
    write_json(root / "seal.json", protected_files(root, contract))
    refresh(root)
    return contract


def records(root):
    return [read_json(path) for path in sorted((root / "experiments").glob("exp_*/result.json"))]


def refresh(root):
    rows = records(root)
    kept = [row for row in rows if row["status"] == "keep"]
    best = kept[-1] if kept else None
    tested = [row["config"] for row in rows if row.get("config") is not None]
    state = {
        "best": best,
        "tested": tested,
        "spent_seconds": sum(row.get("seconds", 0) for row in rows),
        "experiments": len(rows),
        "best_public": {"id": best["id"], "config": best["config"], "metrics": best["metrics"]}
        if best
        else None,
        "recent": [
            {k: row.get(k) for k in ("id", "status", "config", "metrics", "hypothesis", "error")}
            for row in rows[-8:]
        ],
        "findings": [row.get("finding", "") for row in rows[-8:] if row.get("finding")],
        "failure_public": None,
    }
    evaluated = [row for row in rows if row.get("metrics")]
    if evaluated:
        latest = evaluated[-1]["metrics"]
        state["failure_public"] = {
            k: v for k, v in latest.items() if k != "representative_failures"
        }
        write_json(root / "latest_failure_report.json", latest)
    write_json(root / "research_state.json", state)
    if best:
        write_json(root / "best_config.json", best["config"])
        write_json(
            root / "best.json",
            {"id": best["id"], "checkpoint": best["checkpoint"], "metrics": best["metrics"]},
        )
    stream = io.StringIO(newline="")
    fields = [
        "id",
        "status",
        "successes",
        "episodes",
        "success_rate",
        "mean_cookies",
        "seconds",
        "config_hash",
        "hypothesis",
        "error",
    ]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{k: row.get(k, "") for k in fields},
                **{
                    k: row.get("metrics", {}).get(k, "")
                    for k in fields
                    if k
                    in (
                        "successes",
                        "episodes",
                        "success_rate",
                        "mean_cookies",
                    )
                },
            }
        )
    temporary = root / "results.csv.tmp"
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    temporary.replace(root / "results.csv")
    return state


def check_budget(protocol, state):
    if state["experiments"] >= protocol["max_experiments"]:
        raise RuntimeError("Study experiment budget exhausted (failed runs also count)")
    if state["spent_seconds"] >= protocol["max_hours"] * 3600:
        raise RuntimeError("Study time budget exhausted")


def config_seen(config, state):
    return digest(config) in {digest(c) for c in state["tested"]}
