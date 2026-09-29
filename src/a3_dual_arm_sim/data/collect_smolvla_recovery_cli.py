"""One human-reviewed takeover episode in a new, separate dataset directory."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.controllers.recovery import RecoveryPolicy
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
from a3_dual_arm_sim.workflows.runner import EpisodeRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2500)
    args = parser.parse_args()
    if 3000 <= args.seed <= 3019:
        parser.error("Acceptance seeds are reserved")
    bench = CookieBatchBenchmark(render=True)
    policy = RecoveryPolicy(SmolVLAPolicyAdapter(args.checkpoint))
    env = bench.create_env(render_cameras=True)
    recorder = None
    try:
        env.set_key_callback(policy.handle_key)
        recorder = LeRobotV3Recorder(
            args.root, repo_id="local/a3-human-recovery", fps=20, image_height=256, image_width=256
        )
        runner = EpisodeRunner(
            env,
            policy,
            task="transfer 10 cookies into target box",
            recorder=recorder,
            realtime=True,
            save_failed_episodes=False,
        )
        print(
            "T toggles model/human control. Existing teleop movement keys apply. Space: emergency stop.",
            flush=True,
        )
        result = runner.run(seed=args.seed, max_steps=1000)
        (args.root / "takeover.json").write_text(
            json.dumps(
                {
                    "result": asdict(result),
                    "events": policy.events,
                    "initial_controller": "model",
                    "review_required": True,
                    "training_eligible": False,
                },
                indent=2,
            )
        )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        policy.close()


if __name__ == "__main__":
    main()
