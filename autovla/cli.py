"""Run bounded, restartable SmolVLA experiments from the repository root."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

from .agent import api_settings, make_prompt, propose
from .config import apply_proposal, digest, read_json, write_json
from .metrics import is_better, summarize
from .process import run_job, study_lock
from .study import check_budget, config_seen, init_study, records, refresh, verify

REPO = Path(__file__).resolve().parents[1]
FINISHED = {"keep", "discard", "error", "interrupted", "stopped"}


def doctor(root, contract):
    if contract["mode"] == "demo":
        return {"mode": "demo", "synthetic": True}
    for module in ("torch", "lerobot", "mujoco", "pyarrow"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Missing {module}; install the project's [train] dependencies")
    import torch

    if contract["protocol"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    versions = {
        key: importlib.metadata.version(key)
        for key in (
            "torch",
            "lerobot",
            "mujoco",
            "numpy",
            "transformers",
        )
    }
    if versions["lerobot"] != "0.5.1":
        raise RuntimeError("This training adapter requires the project's pinned LeRobot 0.5.1")
    path = root / "environment.json"
    if path.exists() and read_json(path) != versions:
        raise RuntimeError("Training environment changed; start a new study")
    write_json(path, versions)
    return versions


def recover_interrupted(root):
    for path in sorted((root / "experiments").glob("exp_*/result.json")):
        row = read_json(path)
        if row["status"] not in FINISHED:
            row.update(
                status="interrupted",
                error="Controller exited before finalizing this run",
                seconds=max(0, time.time() - row["started_at"]),
            )
            write_json(path, row)


def new_record(root, protocol, *, config=None, hypothesis="Baseline"):
    state = refresh(root)
    check_budget(protocol, state)
    directory = root / "experiments" / f"exp_{state['experiments'] + 1:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    row = {
        "id": directory.name,
        "status": "proposing",
        "config": config,
        "hypothesis": hypothesis,
        "started_at": time.time(),
        "seconds": 0,
    }
    write_json(directory / "result.json", row)
    return directory, row


def remaining(protocol, state, row):
    elapsed = time.time() - row["started_at"]
    return min(
        protocol["experiment_timeout_seconds"] - elapsed,
        protocol["max_hours"] * 3600 - state["spent_seconds"] - elapsed,
    )


def worker_job(contract, request, directory, stage, timeout):
    request_path = directory / f"{stage}_request.json"
    output = directory / f"{stage}.json"
    write_json(request_path, request)
    run_job(
        [
            sys.executable,
            "-m",
            "autovla.worker",
            stage,
            "--request",
            str(request_path),
            "--output",
            str(output),
        ],
        cwd=contract["repo"],
        log=directory / f"{stage}.log",
        timeout=timeout,
    )
    return read_json(output)


def synthetic_eval(protocol, config, split):
    spec = protocol[split]
    ratio = {8: 0.5, 4: 0.75, 16: 0.25}.get(config["n_action_steps"], 0.5)
    episodes = []
    for case in protocol["cases"]:
        for index in range(spec["episodes"]):
            success = index < int(spec["episodes"] * ratio)
            episodes.append(
                {
                    "profile": case["profile"],
                    "column_mode": str(case["source_column"]),
                    "source_column": case["source_column"],
                    "seed": spec["seed_start"] + index,
                    "success": success,
                    "score": 10 if success else 2,
                    "steps": min(100, protocol["max_steps"]),
                    "cookies_in_source": 70 if success else 78,
                    "failure_reason": None if success else "synthetic_timeout",
                }
            )
    return {"synthetic": True, "episodes": episodes}


def execute(root, contract, directory, row, state):
    protocol, config = contract["protocol"], row["config"]
    verify(root)
    request = {"protocol": protocol, "config": config, "model_output": str(directory / "model")}
    write_json(directory / "config.json", config)
    row.update(config_hash=digest(config), status="training")
    write_json(directory / "result.json", row)
    print(f"{row['id']}: training ({contract['mode']})", flush=True)
    best = state["best"]
    training_keys = [key for key in config if key != "n_action_steps"]
    reuse = best is not None and all(config[k] == best["config"][k] for k in training_keys)
    if reuse:
        checkpoint = best["checkpoint"]
        row["reused_checkpoint_from"] = best["id"]
        write_json(
            directory / "train.json",
            {
                "checkpoint": checkpoint,
                "reused_from": best["id"],
                "synthetic": contract["mode"] == "demo",
            },
        )
        print(f"{row['id']}: reused {best['id']} weights for an inference-only change", flush=True)
    elif contract["mode"] == "demo":
        checkpoint = str(directory / "synthetic_model")
        write_json(Path(checkpoint) / "SYNTHETIC.json", {"config": config})
        write_json(directory / "train.json", {"checkpoint": checkpoint, "synthetic": True})
    else:
        checkpoint = worker_job(
            contract, request, directory, "train", remaining(protocol, state, row)
        )["checkpoint"]
    row.update(checkpoint=checkpoint, status="evaluating")
    write_json(directory / "result.json", row)
    print(f"{row['id']}: fixed development evaluation", flush=True)
    if contract["mode"] == "demo":
        report = synthetic_eval(protocol, config, "development")
        write_json(directory / "evaluate.json", report)
    else:
        expected_hashes = None
        if reuse:
            expected_hashes = read_json(root / "experiments" / best["id"] / "evaluate.json")[
                "checkpoint_hashes"
            ]
        report = worker_job(
            contract,
            {
                **request,
                "checkpoint": checkpoint,
                "split": "development",
                "expected_checkpoint_hashes": expected_hashes,
            },
            directory,
            "evaluate",
            remaining(protocol, state, row),
        )
    if bool(report.get("synthetic")) != (contract["mode"] == "demo"):
        raise ValueError("Synthetic and real evaluation results cannot be mixed")
    metrics = summarize(report, protocol)
    verify(root)
    write_json(directory / "failure_report.json", metrics)
    keep = state["best"] is None or is_better(metrics, state["best"]["metrics"])
    row.update(metrics=metrics, status="keep" if keep else "discard")
    print(
        f"{row['id']}: {row['status'].upper()} — {metrics['successes']}/{metrics['episodes']} "
        f"complete tasks; mean cookies {metrics['mean_cookies']:.2f}",
        flush=True,
    )


def finish_record(root, directory, row, error=None):
    if error is not None:
        row.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            error=str(error)[:1000] or type(error).__name__,
        )
    row["seconds"] = max(0, time.time() - row["started_at"])
    write_json(directory / "result.json", row)
    refresh(root)


def baseline(root, contract):
    state = refresh(root)
    if state["best"]:
        return
    if records(root):
        raise RuntimeError(
            "Baseline failed or was interrupted; inspect logs and initialize a new study"
        )
    config = read_json(root / "baseline_config.json")
    directory, row = new_record(root, contract["protocol"], config=config)
    try:
        execute(root, contract, directory, row, state)
    except BaseException as exc:
        finish_record(root, directory, row, exc)
        raise
    finish_record(root, directory, row)


def candidate(root, contract, kind, proposal_path=None):
    state = refresh(root)
    if state["best"] is None:
        raise RuntimeError("Run the baseline first: loop --rounds 0")
    protocol = contract["protocol"]
    verify(root)
    directory, row = new_record(root, protocol)
    try:
        if proposal_path:
            proposal = read_json(proposal_path)
            write_json(directory / "proposal.json", proposal)
        else:
            prompt = make_prompt(
                (root / "program.md").read_text(encoding="utf-8"),
                {**state, "training_steps": protocol["steps"]},
                read_json(root / "proposal.schema.json"),
            )
            (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
            proposal = propose(
                kind,
                prompt=prompt,
                output=directory / "proposal.json",
                repo=contract["repo"],
                schema=root / "proposal.schema.json",
                tested=state["tested"],
                timeout=min(protocol["agent_timeout_seconds"], remaining(protocol, state, row)),
            )
        config = apply_proposal(state["best"]["config"], proposal, protocol["steps"])
        row.update(hypothesis=proposal["hypothesis"], finding=proposal["finding"])
        if config is None:
            row["status"] = "stopped"
            finish_record(root, directory, row)
            print("Agent requested a stop", flush=True)
            return False
        if config_seen(config, state):
            raise ValueError("This configuration was already tested; no training started")
        row["config"] = config
        execute(root, contract, directory, row, state)
    except BaseException as exc:
        finish_record(root, directory, row, exc)
        raise
    finish_record(root, directory, row)
    return True


def acceptance(root, contract):
    state = refresh(root)
    if state["best"] is None:
        raise RuntimeError("No completed baseline")
    path = root / "acceptance.json"
    if path.exists():
        raise RuntimeError("Acceptance was already started; these held-out seeds are consumed")
    protocol = contract["protocol"]
    if state["spent_seconds"] >= protocol["max_hours"] * 3600:
        raise RuntimeError("Study time budget exhausted")
    directory = root / "acceptance"
    directory.mkdir(exist_ok=False)
    best = state["best"]
    row = {
        "status": "running",
        "best_id": best["id"],
        "checkpoint": best["checkpoint"],
        "config": best["config"],
        "started_at": time.time(),
    }
    # Consumed before launch; a crash must not enable repeated holdout selection.
    write_json(path, row)
    try:
        if contract["mode"] == "demo":
            report = synthetic_eval(protocol, best["config"], "acceptance")
            write_json(directory / "evaluate.json", report)
        else:
            development = read_json(root / "experiments" / best["id"] / "evaluate.json")
            from .study import file_hash

            if (
                file_hash(Path(best["checkpoint"]) / "model.safetensors")
                != development["weights_sha256"]
            ):
                raise RuntimeError("Best checkpoint weights changed after development evaluation")
            report = worker_job(
                contract,
                {
                    "protocol": protocol,
                    "config": best["config"],
                    "checkpoint": best["checkpoint"],
                    "split": "acceptance",
                    "expected_checkpoint_hashes": development["checkpoint_hashes"],
                },
                directory,
                "evaluate",
                remaining(protocol, state, row),
            )
        if bool(report.get("synthetic")) != (contract["mode"] == "demo"):
            raise ValueError("Synthetic and real results cannot be mixed")
        row.update(status="complete", metrics=summarize(report, protocol, "acceptance"))
        verify(root)
    except BaseException as exc:
        row.update(status="failed", error=str(exc)[:1000] or type(exc).__name__)
        raise
    finally:
        row["seconds"] = time.time() - row["started_at"]
        write_json(path, row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("init", "doctor", "loop", "submit", "status", "accept"))
    result.add_argument("--study", type=Path, default=Path("outputs/autovla"))
    result.add_argument("--protocol", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--demo", action="store_true")
    result.add_argument("--agent", choices=("api", "codex", "mock"), default="api")
    result.add_argument(
        "--rounds", type=int, default=3, help="Candidate rounds; baseline is additional"
    )
    result.add_argument("--proposal", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.study.expanduser().resolve()
    try:
        if args.rounds < 0:
            raise ValueError("rounds must be nonnegative")
        if args.command == "init":
            init_study(
                root,
                repo=REPO,
                protocol_path=args.protocol,
                config_path=args.config,
                demo=args.demo,
            )
            print(f"Initialized {'SYNTHETIC DEMO' if args.demo else 'live'} study: {root}")
            return 0
        if not (root / "contract.json").is_file():
            raise ValueError("Study not initialized; run init first")
        with study_lock(root):
            contract = verify(root)
            recover_interrupted(root)
            state = refresh(root)
            if args.command == "status":
                print(
                    json.dumps(
                        {
                            "mode": contract["mode"],
                            "experiments": state["experiments"],
                            "best": state["best_public"],
                            "acceptance": read_json(root / "acceptance.json")
                            if (root / "acceptance.json").exists()
                            else None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            if args.command == "doctor":
                print(json.dumps(doctor(root, contract), indent=2))
                return 0
            if args.command in ("loop", "submit") and (root / "acceptance.json").exists():
                raise RuntimeError(
                    "Study frozen for acceptance; further tuning requires a new study"
                )
            if args.agent == "mock" and contract["mode"] != "demo":
                raise ValueError("The mock agent is restricted to --demo studies")
            doctor(root, contract)
            if args.command == "accept":
                acceptance(root, contract)
            elif args.command == "submit":
                if args.proposal is None:
                    raise ValueError("submit requires --proposal")
                candidate(root, contract, args.agent, args.proposal)
            else:
                if args.agent == "api" and args.rounds:
                    api_settings()  # Fail before GPU work if provider settings are missing.
                baseline(root, contract)
                for _ in range(args.rounds):
                    state = refresh(root)
                    if state["experiments"] >= contract["protocol"]["max_experiments"]:
                        print("Study experiment budget reached", flush=True)
                        break
                    if not candidate(root, contract, args.agent):
                        break
        return 0
    except KeyboardInterrupt:
        print("Interrupted; child jobs stopped and run journal preserved", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"AutoVLA: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
