"""Describe a saved RGB window without running a robot or changing its verifier.

This diagnostic sends only four camera images and their stored subgoal to the
configured visual model. The independent completion label is appended to the
report after inference, never included in the prompt or camera content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim inspect verifier-sample", description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("datasets/verifier_smoke_server_seed0")
    )
    parser.add_argument(
        "--agent-config", type=Path,
        default=project_root() / "configs/agents/agentic_qwen35.yaml",
    )
    parser.add_argument("--sample-index", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample-index must not be negative")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new file; existing paths will not be overwritten")

    # Set the headless backend before any import that initializes MuJoCo.
    # No environment, renderer, executor, or robot is instantiated by this probe.
    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"

    import yaml

    from a3_dual_arm_sim.agents.backends import ChatBackendConfig, make_chat_backend
    from a3_dual_arm_sim.data.verifier import read_verifier_samples
    from a3_dual_arm_sim.evaluation.verifier import load_visual_frames

    root = args.data_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--data-root must be a directory")
    samples_path = (root / "samples.jsonl").resolve(strict=True)
    if not samples_path.is_relative_to(root) or not samples_path.is_file():
        raise ValueError("sample metadata path must stay inside the verifier dataset")
    matches = [
        sample for sample in read_verifier_samples(root)
        if sample.get("sample_index") == args.sample_index
    ]
    if len(matches) != 1:
        raise ValueError("requested sample_index must identify exactly one saved sample")
    sample = matches[0]
    subgoal = sample["input"]["subgoal"]
    if not isinstance(subgoal, str) or not subgoal.strip():
        raise ValueError("saved sample subgoal must be nonempty text")
    # Reuse the evaluation loader's path containment, RGB conversion, and strict
    # chronological step checks. It opens only front and left-wrist image files.
    frames = load_visual_frames(root, sample)
    if len(frames) != 2:
        raise ValueError("this diagnostic requires exactly two saved camera frame pairs")
    images = [frame[camera] for frame in frames for camera in ("front", "left_wrist")]
    prompt = (
        "Describe the attached four RGB images of a cookie-packing robot. "
        "They are two chronological frame pairs, oldest first: image 1 is the "
        "earlier front/global view; image 2 is its left-wrist view; image 3 is the "
        "later front/global view; image 4 is its left-wrist view. "
        f"Stored subgoal: {subgoal}\n"
        "For each of the four images, describe the visible cookies and give the "
        "number visibly distinguishable, or a defensible range when uncertain. "
        "State whether they appear held by the gripper, resting in a tray or box, "
        "or whether this cannot be determined. Describe the gripper position, "
        "occlusion, and uncertainty in each view. Then describe observable changes "
        "between the earlier and later pair. Do not infer hidden cookies or assume "
        "the subgoal has been achieved. Do not output a Yes/No completion decision; "
        "this is an open-ended visual description, not a success test."
    )
    settings = yaml.safe_load(args.agent_config.read_text(encoding="utf-8")) or {}
    config = ChatBackendConfig(**settings["verifier"])
    backend = make_chat_backend(config)
    raw_response = None
    model_error = None
    try:
        raw_response = backend.complete(prompt, images)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        # Public backend exceptions omit credentials; do not introspect transport
        # objects, environment variables, or the full backend configuration.
        model_error = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    # Independent annotation is read only after the model call. Neither this
    # label nor sample metadata/audit files are passed to the chat backend.
    expected = sample["label"]["completion"]
    if expected not in ("Yes", "No"):
        raise ValueError("stored completion label must be exactly Yes or No")
    result = {
        "diagnostic": "open-ended description of one saved temporal RGB window",
        "sample_index": args.sample_index,
        "dataset_root": str(root),
        "backend": config.kind,
        "model": config.model,
        "model_family": config.model_family,
        "visual_verifier_fine_tuned": config.fine_tuned,
        "model_input": {"subgoal": subgoal, "prompt": prompt},
        "input_path_identifiers": [
            {
                "step": entry["step"],
                "front": entry["images"]["front"],
                "left_wrist": entry["images"]["left_wrist"],
            }
            for entry in sample["input"]["frames"]
        ],
        "raw_response": raw_response,
        "model_error": model_error,
        "independent_stored_expected_label": {"completion": expected},
        "model_input_contains_labels": False,
        "simulator_truth_used_as_model_input": False,
        "robot_motion_performed": False,
        "completion_accuracy_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a concurrent writer from being replaced.
    with args.output.open("x", encoding="utf-8") as report:
        report.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if model_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
