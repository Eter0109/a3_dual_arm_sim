"""Run the single-box Planner -> LeRobot/Expert -> temporal Verifier SAP loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim agent run", description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project_root() / "configs/envs/cookie_same_column.yaml")
    parser.add_argument("--agent-config", type=Path,
                        default=project_root() / "configs/agents/agentic_single_box.yaml")
    parser.add_argument("--planner", choices=("fixed", "vlm"), default="fixed")
    parser.add_argument("--verifier", choices=("oracle", "vlm"), default="oracle")
    parser.add_argument("--executor", choices=("expert", "lerobot"), default="expert")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repo-id", default="local/a3-single-box-skills")
    parser.add_argument("--policy-type", choices=("smolvla", "pi05"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-column", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", help="Optional user instruction within the single-box skill library")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--profile", choices=("fixed", "diverse"), default="fixed")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/agentic_single_box.json"))
    parser.add_argument("--verification-data", type=Path)
    args = parser.parse_args(argv)
    if args.executor == "lerobot" and (args.checkpoint is None or args.dataset_root is None):
        parser.error("--executor lerobot requires --checkpoint and --dataset-root")
    if not args.render and os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")

    import yaml

    from a3_dual_arm_sim.agents.backends import (
        ChatBackendConfig,
        make_chat_backend,
    )
    from a3_dual_arm_sim.agents.controller import (
        AgenticController,
        AgentLoopConfig,
    )
    from a3_dual_arm_sim.agents.executors import ExpertSkillExecutor, PolicySkillExecutor
    from a3_dual_arm_sim.agents.planning import FixedSkillPlanner, MultimodalSkillPlanner
    from a3_dual_arm_sim.agents.verification import OracleSkillVerifier, TemporalSkillVerifier
    from a3_dual_arm_sim.data.verifier_recording import TemporalVerificationRecorder
    from a3_dual_arm_sim.envs.config import load_config
    from a3_dual_arm_sim.envs.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
    from a3_dual_arm_sim.evaluation.benchmark import DIVERSE_RANDOMIZATION, column_task_instruction
    from a3_dual_arm_sim.evaluation.skill_truth import GroundTruthSkillEvaluator

    settings = yaml.safe_load(args.agent_config.read_text(encoding="utf-8")) or {}
    loop_options = dict(settings.get("loop", {}))
    if args.max_steps is not None:
        loop_options["max_steps"] = args.max_steps
    cfg = AgentLoopConfig(**loop_options)
    backends = []
    if args.planner == "fixed":
        planner = FixedSkillPlanner(args.source_column)
    else:
        backend = make_chat_backend(ChatBackendConfig(**settings["planner"]))
        backends.append(backend)
        planner = MultimodalSkillPlanner(backend, allowed_source_columns=(args.source_column,))
    if args.verifier == "oracle":
        verifier = OracleSkillVerifier()
    else:
        # Reuse one local weight copy if both roles have identical settings.
        if args.planner == "vlm" and settings["planner"] == settings["verifier"]:
            backend = backends[0]
        else:
            backend = make_chat_backend(ChatBackendConfig(**settings["verifier"]))
            backends.append(backend)
        verifier = TemporalSkillVerifier(backend)

    env = A3CookieTransferEnv(
        load_config(args.config),
        task_config=CookieTransferTaskConfig(
            require_exact_slots=False, require_released=True, terminate_on_success=False,
        ),
        render_mode="human" if args.render else None,
        render_cameras=True,
    )
    executor = None
    recorder = None
    try:
        reset_options = {"randomize_cookies": False, "randomize_boxes": False}
        if args.profile == "diverse":
            reset_options = {"randomize_cookies": True, "randomize_boxes": True,
                             **DIVERSE_RANDOMIZATION}
        observation, _ = env.reset(seed=args.seed, options=reset_options)
        if args.executor == "expert":
            executor = ExpertSkillExecutor(env, seed=args.seed,
                                           source_column_number=args.source_column)
        else:
            from a3_dual_arm_sim.policies.lerobot import LeRobotPolicyPlugin

            policy_options = {"device": args.device}
            if args.policy_type:
                policy_options["policy_type"] = args.policy_type
            policy_options.update(settings.get("executor", {}))
            policy = LeRobotPolicyPlugin(
                args.checkpoint, args.dataset_root, args.repo_id, **policy_options,
            )
            executor = PolicySkillExecutor(policy, seed=args.seed)
        if args.verification_data:
            recorder = TemporalVerificationRecorder(
                args.verification_data, max_frames=cfg.window_frames, frame_stride=cfg.frame_interval,
            )
        result = AgenticController(planner, verifier, config=cfg).run(
            env, executor, task=args.task or column_task_instruction(args.source_column - 1),
            observation=observation, seed=args.seed,
            truth_evaluator=GroundTruthSkillEvaluator(env), verification_recorder=recorder,
        )
        result.update(executor=args.executor, planner=args.planner, profile=args.profile,
                      planner_response=getattr(planner, "last_response", None),
                      checkpoint=str(args.checkpoint) if args.checkpoint else None,
                      visual_verifier_fine_tuned=(
                          bool(getattr(backend, "adapter_loaded", False))
                          if args.verifier == "vlm" else False
                      ),
                      verifier_adapter=(settings["verifier"].get("adapter_path")
                                        if args.verifier == "vlm" else None),
                      action_training_data_recorded=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("events", "plan")},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if result["success"] else 1
    finally:
        if executor is not None:
            executor.close()
        if recorder is not None:
            recorder.close()
        env.close()
        for backend in backends:
            close = getattr(backend, "close", None)
            if close is not None:
                close()


if __name__ == "__main__":
    raise SystemExit(main())
