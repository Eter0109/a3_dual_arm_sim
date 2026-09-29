"""Fine-tune a local Qwen temporal verifier with answer-only language LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.training.verifier import VerifierTrainingConfig, train_verifier_lora


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim train verifier", description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared parent-seed-grouped train/holdout dataset")
    parser.add_argument("--output", type=Path, required=True, help="New training output directory")
    parser.add_argument("--base-model", default=str(project_root() / "models/Qwen3.5-4B"))
    parser.add_argument("--init-adapter", help="Local LoRA weight warm-start; optimizer resets, not resume")
    parser.add_argument("--model-family", choices=("qwen3_5", "qwen2_5_vl"), default="qwen3_5")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-max-edge", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args(argv)
    settings = vars(args).copy()
    data_root = settings.pop("data_root")
    output = settings.pop("output")
    settings["gradient_checkpointing"] = not settings.pop("no_gradient_checkpointing")
    try:
        config = VerifierTrainingConfig(**settings)
    except ValueError as exc:
        parser.error(str(exc))
    train_verifier_lora(data_root, output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
