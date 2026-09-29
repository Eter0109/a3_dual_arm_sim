"""Inspect one real planner response against a saved camera image, without moving the robot."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim inspect planner", description=__doc__)
    parser.add_argument("--agent-config", type=Path,
                        default=project_root() / "configs/agents/agentic_single_box.yaml")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/verifier_smoke_server_seed0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/agentic_planner_probe.json"))
    args = parser.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import yaml
    from PIL import Image

    from a3_dual_arm_sim.agents.backends import (
        AgenticModelError,
        ChatBackendConfig,
        make_chat_backend,
    )
    from a3_dual_arm_sim.agents.planning import MultimodalSkillPlanner

    settings = yaml.safe_load(args.agent_config.read_text(encoding="utf-8"))
    config = ChatBackendConfig(**settings["planner"])
    with (args.dataset_root / "samples.jsonl").open(encoding="utf-8") as stream:
        sample = json.loads(next(stream))
    image_path = args.dataset_root / sample["input"]["frames"][0]["images"]["front"]
    with Image.open(image_path) as saved:
        front = np.asarray(saved.convert("RGB"))
    backend = make_chat_backend(config)
    captured = []

    class RecordingBackend:
        def complete(self, prompt, images):
            output = backend.complete(prompt, images)
            captured.append(output[:16384])
            return output

        def close(self):
            backend.close()

    planner = MultimodalSkillPlanner(RecordingBackend(), allowed_source_columns=(1,))
    result = {
        "backend": config.kind,
        "model": config.model,
        "image": str(image_path),
        "robot_moved": False,
        "simulator_truth_used_as_model_input": False,
        "success": False,
    }
    try:
        plan = planner.plan(
            "Transfer 10 cookies from source column 1 into the target box in two batches of five.",
            {"front": front},
        )
        result.update(success=True, plan=[asdict(item) for item in plan])
    except AgenticModelError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["raw_response"] = captured[-1] if captured else None
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
