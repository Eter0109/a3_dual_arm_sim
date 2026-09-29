"""Evaluate a visual verifier on saved, independently labeled RGB windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.agents.backends import ChatBackendConfig, make_chat_backend
from a3_dual_arm_sim.agents.verification import TemporalSkillVerifier
from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.evaluation.verifier import evaluate_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim eval verifier", description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, default=project_root() / "configs/agents/agentic_single_box.yaml")
    parser.add_argument("--parent-seeds", type=int, nargs="+")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=8)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--per-class-limit", type=int)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cookie_verifier_evaluation.json"))
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.per_class_limit is not None and args.per_class_limit < 1:
        parser.error("--per-class-limit must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("use a new verifier evaluation output file")
    import yaml

    settings = yaml.safe_load(args.agent_config.read_text(encoding="utf-8")) or {}
    config = ChatBackendConfig(**settings["verifier"])
    backend = make_chat_backend(config)
    try:
        result = evaluate_samples(
            args.data_root, TemporalSkillVerifier(backend), limit=None if args.all else args.limit,
            parent_seeds=args.parent_seeds,
            per_class_limit=args.per_class_limit, selection_seed=args.selection_seed,
        )
    finally:
        adapter_loaded = bool(getattr(backend, "adapter_loaded", False))
        backend.close()
    result.update(
        backend=config.kind, model=config.model, model_family=config.model_family,
        configured_fine_tuned=config.fine_tuned,
        visual_verifier_fine_tuned=adapter_loaded if config.kind == "local_qwen" else None,
        adapter_loaded=adapter_loaded, adapter_path=config.adapter_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "samples"},
        ensure_ascii=False, indent=2,
    ), flush=True)
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
