"""Overnight Guardian Pipeline for A3 Dual Arm Sim.

------------------------------------------------
Automates the full pipeline:
  Stage 1: Expert Data Collection (Cookie Transfer Benchmark, with auto-resume)
  Stage 2: Dataset Verification & Audit
  Stage 3: Policy Imitation Learning Training (ACT / SmolVLA / Diffusion)
  Stage 4: Multi-seed Closed-Loop MuJoCo Evaluation

Features:
- Windows Sleep Prevention (SetThreadExecutionState) & Process Priority Boost (High + EcoQoS disabled)
- Automatic retry on transient subprocess crashes (up to max_retries)
- Real-time status reporting to outputs/pipeline_status.json
- Clean log output to both console and outputs/overnight_pipeline.log
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Win32 APIs for Sleep Prevention & Process Boost
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
HIGH_PRIORITY_CLASS = 0x00000080

ProcessPowerThrottling = 0x26
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


def set_sleep_prevention(enable: bool = True) -> bool:
    """Keep Windows system and display awake."""
    if sys.platform != "win32":
        return True
    try:
        flags = ES_CONTINUOUS
        if enable:
            flags |= ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        res = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        return res != 0
    except Exception:
        return False


def boost_process(pid: int) -> bool:
    """Set process priority to High and disable Windows Power Throttling / EcoQoS."""
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        pid,
    )
    if not handle:
        return False
    try:
        kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
        throttling = PROCESS_POWER_THROTTLING_STATE()
        throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
        throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        throttling.StateMask = 0  # 0 means turn off throttling
        kernel32.SetProcessInformation(
            handle,
            ProcessPowerThrottling,
            ctypes.byref(throttling),
            ctypes.sizeof(throttling),
        )
        return True
    except Exception:
        return False
    finally:
        kernel32.CloseHandle(handle)


class PipelineLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, prefix: str = "[INFO]"):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now_str}] {prefix} {msg}"
        print(line, flush=True)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def get_current_dataset_stats(dataset_root: Path) -> dict:
    """Read episode counts and frame counts from dataset metadata."""
    info_path = dataset_root / "meta" / "info.json"
    metadata_path = dataset_root / "a3_episode_metadata.jsonl"
    attempts_path = dataset_root / "attempts.jsonl"

    saved_count = 0
    total_attempts = 0
    frames = 0

    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            saved_count = int(info.get("total_episodes", 0))
            frames = int(info.get("total_frames", 0))
        except Exception:
            pass

    if metadata_path.is_file():
        try:
            lines = [l for l in metadata_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            saved_count = max(saved_count, len(lines))
        except Exception:
            pass

    if attempts_path.is_file():
        try:
            lines = [l for l in attempts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            total_attempts = len(lines)
        except Exception:
            pass

    return {
        "saved_episodes": saved_count,
        "total_attempts": max(total_attempts, saved_count),
        "total_frames": frames,
    }


def update_status_file(status_path: Path, data: dict):
    data["updated_at"] = datetime.datetime.now().isoformat()
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def run_command_with_boost(
    cmd: list[str],
    cwd: Path,
    logger: PipelineLogger,
    status_path: Path,
    status_data: dict,
    env: dict | None = None,
) -> int:
    """Run a subprocess, apply performance boost, and stream its output."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Windows UTF-8 & PyAv / MuJoCo sanity; Linux headless EGL
    merged_env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "win32" and merged_env.get("MUJOCO_GL") == "egl":
        del merged_env["MUJOCO_GL"]
    elif sys.platform != "win32":
        merged_env.setdefault("MUJOCO_GL", "egl")
        merged_env.setdefault("PYOPENGL_PLATFORM", "egl")

    logger.log(f"Launching command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=merged_env,
    )

    # Boost child process
    time.sleep(0.5)
    boosted = boost_process(proc.pid)
    logger.log(f"Child process PID={proc.pid} boosted (High priority + EcoQoS disabled): {boosted}")

    status_data["current_pid"] = proc.pid
    update_status_file(status_path, status_data)

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            logger.log(line, prefix="[SUB]")
            if line.startswith("{") and "saved" in line:
                try:
                    payload = json.loads(line)
                    status_data["dataset_stats"] = {
                        "saved_episodes": payload.get("saved"),
                        "last_score": payload.get("score"),
                        "last_success": payload.get("success"),
                    }
                    update_status_file(status_path, status_data)
                except Exception:
                    pass
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        logger.log("Interrupt received! Terminating child process...", prefix="[WARN]")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def run_pipeline(args: argparse.Namespace):
    workspace = Path(__file__).resolve().parents[2]
    python_exe = sys.executable

    log_path = workspace / "outputs" / "overnight_pipeline.log"
    status_path = workspace / "outputs" / "pipeline_status.json"
    dataset_root = workspace / args.dataset_root
    output_dir = workspace / args.output_dir
    if args.policy_type == "act" and "smolvla" in str(output_dir):
        output_dir = workspace / "outputs" / "train" / "a3_act_25k"
    elif args.policy_type == "diffusion" and "smolvla" in str(output_dir):
        output_dir = workspace / "outputs" / "train" / "a3_diffusion_50k"

    logger = PipelineLogger(log_path)
    logger.log("=" * 60)
    logger.log(f"STARTING OVERNIGHT GUARDIAN PIPELINE ({args.policy_type.upper()})")
    logger.log(f"Workspace: {workspace}")
    logger.log(f"Target Episodes: {args.target_episodes}")
    logger.log(f"Training Steps: {args.train_steps}")
    logger.log(f"Batch Size: {args.batch_size}")
    logger.log(f"Device: {args.device}")
    logger.log("=" * 60)

    # 1. Enable Sleep Prevention & Boost Current Process
    set_sleep_prevention(True)
    boost_process(os.getpid())
    logger.log("Sleep prevention active (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)")

    status_data = {
        "status": "RUNNING",
        "stage": "INIT",
        "start_time": datetime.datetime.now().isoformat(),
        "target_episodes": args.target_episodes,
        "train_steps": args.train_steps,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
    }
    update_status_file(status_path, status_data)

    try:
        # ==========================================
        # STAGE 1: DATA COLLECTION
        # ==========================================
        status_data["stage"] = "COLLECTION"
        stats = get_current_dataset_stats(dataset_root)
        current_saved = stats["saved_episodes"]
        logger.log(f"Current saved episodes in dataset: {current_saved} / {args.target_episodes}")

        retry_count = 0
        while current_saved < args.target_episodes:
            status_data["stage_info"] = f"Collection: {current_saved}/{args.target_episodes}"
            update_status_file(status_path, status_data)

            has_existing = (dataset_root / "collection_summary.json").is_file() and current_saved > 0
            collect_cmd = [
                python_exe,
                "examples/collect_cookie_benchmark.py",
                "--root",
                str(dataset_root),
                "--repo-id",
                str(args.repo_id),
                "--episodes",
                str(args.target_episodes),
            ]
            if has_existing:
                collect_cmd.append("--resume")

            logger.log(f"Starting data collection step (resume={has_existing})...")
            exit_code = run_command_with_boost(
                collect_cmd,
                cwd=workspace,
                logger=logger,
                status_path=status_path,
                status_data=status_data,
            )

            stats = get_current_dataset_stats(dataset_root)
            current_saved = stats["saved_episodes"]
            logger.log(f"Collection round finished with exit code {exit_code}. Saved: {current_saved}/{args.target_episodes}")

            if current_saved >= args.target_episodes:
                logger.log("Target episodes achieved successfully!")
                break

            if exit_code != 0:
                retry_count += 1
                if retry_count > args.max_retries:
                    err_msg = f"Data collection failed after {args.max_retries} retries."
                    logger.log(err_msg, prefix="[ERROR]")
                    status_data["status"] = "FAILED"
                    status_data["error"] = err_msg
                    update_status_file(status_path, status_data)
                    return 1
                logger.log(f"Subprocess crashed, auto-recovering in 5 seconds (retry {retry_count}/{args.max_retries})...", prefix="[WARN]")
                time.sleep(5)

        logger.log("Stage 1 completed: All required episodes collected.")

        # ==========================================
        # STAGE 2: DATASET AUDIT
        # ==========================================
        status_data["stage"] = "AUDIT"
        update_status_file(status_path, status_data)
        logger.log("Auditing dataset before starting training...")

        from a3_dual_arm_sim.learning.training import audit_training_dataset
        audit_res = audit_training_dataset(dataset_root, repo_id=args.repo_id)
        logger.log(f"Dataset audit PASSED: {json.dumps(audit_res, indent=2)}")

        # ==========================================
        # STAGE 3: TRAINING
        # ==========================================
        status_data["stage"] = "TRAINING"
        status_data["stage_info"] = f"Training {args.policy_type.upper()} {args.train_steps} steps on {args.device}"
        update_status_file(status_path, status_data)

        effective_output = output_dir
        if effective_output.exists() and not getattr(args, "resume", False):
            timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_output = output_dir.parent / f"{output_dir.name}_{timestamp_suffix}"
            logger.log(f"Output directory existed; adjusted to {effective_output}")

        if args.policy_type == "act":
            train_cmd = [
                python_exe,
                "-m",
                "a3_dual_arm_sim.cli",
                "train-act",
                "--root",
                str(dataset_root),
                "--repo-id",
                args.repo_id,
                "--output",
                str(effective_output),
                "--steps",
                str(args.train_steps),
                "--batch-size",
                str(args.batch_size),
                "--chunk-size",
                str(args.chunk_size),
                "--device",
                args.device,
            ]
            if getattr(args, "num_workers", None) is not None:
                train_cmd.extend(["--num-workers", str(args.num_workers)])
            if getattr(args, "resume", False):
                train_cmd.append("--resume")
                if getattr(args, "checkpoint_path", None):
                    train_cmd.extend(["--checkpoint-path", str(args.checkpoint_path)])
        elif args.policy_type == "diffusion":
            train_cmd = [
                python_exe,
                "-m",
                "a3_dual_arm_sim.cli",
                "train-diffusion",
                "--root",
                str(dataset_root),
                "--repo-id",
                args.repo_id,
                "--output",
                str(effective_output),
                "--steps",
                str(args.train_steps),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            if getattr(args, "num_workers", None) is not None:
                train_cmd.extend(["--num-workers", str(args.num_workers)])
        else:
            train_cmd = [
                python_exe,
                "-m",
                "a3_dual_arm_sim.cli",
                "train-smolvla",
                "--root",
                str(dataset_root),
                "--repo-id",
                args.repo_id,
                "--output",
                str(effective_output),
                "--steps",
                str(args.train_steps),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]

        logger.log(f"Starting {args.policy_type.upper()} training for {args.train_steps} steps...")
        train_exit_code = run_command_with_boost(
            train_cmd,
            cwd=workspace,
            logger=logger,
            status_path=status_path,
            status_data=status_data,
        )

        if train_exit_code != 0:
            err_msg = f"Training exited with code {train_exit_code}"
            logger.log(err_msg, prefix="[ERROR]")
            status_data["status"] = "FAILED"
            status_data["error"] = err_msg
            update_status_file(status_path, status_data)
            return train_exit_code

        # ==========================================
        # STAGE 4: EVALUATION
        # ==========================================
        if not args.skip_eval:
            status_data["stage"] = "EVALUATION"
            status_data["stage_info"] = f"Evaluating {args.policy_type.upper()} on seeds {args.eval_seeds}"
            update_status_file(status_path, status_data)

            latest_checkpoint = effective_output / "checkpoints" / "last" / "pretrained_model"
            if not latest_checkpoint.exists():
                all_cps = sorted((effective_output / "checkpoints").glob("*/pretrained_model"))
                if all_cps:
                    latest_checkpoint = all_cps[-1]

            if args.policy_type == "act":
                eval_script = "examples/evaluate_cookie_act.py"
            elif args.policy_type == "diffusion":
                eval_script = "examples/evaluate_cookie_diffusion.py"
            else:
                eval_script = "examples/evaluate_cookie_smolvla.py"

            eval_report = workspace / "artifacts" / f"cookie_{args.policy_type}_evaluation.json"

            eval_cmd = [
                python_exe,
                eval_script,
                "--checkpoint",
                str(latest_checkpoint),
                "--dataset-root",
                str(dataset_root),
                "--repo-id",
                args.repo_id,
                "--seeds",
                args.eval_seeds,
                "--max-steps",
                str(getattr(args, "eval_max_steps", 1800)),
                "--device",
                args.device,
                "--output",
                str(eval_report),
            ]

            logger.log(f"Starting closed-loop evaluation on seeds {args.eval_seeds}...")
            eval_exit_code = run_command_with_boost(
                eval_cmd,
                cwd=workspace,
                logger=logger,
                status_path=status_path,
                status_data=status_data,
            )
            if eval_exit_code == 0:
                logger.log(f"Evaluation completed successfully! Report: {eval_report}")
                status_data["evaluation_report"] = str(eval_report)
            else:
                logger.log(f"Evaluation finished with code {eval_exit_code}", prefix="[WARN]")

        # ==========================================
        # COMPLETION
        # ==========================================
        logger.log("=" * 60)
        logger.log("OVERNIGHT PIPELINE COMPLETED SUCCESSFULLY!")
        logger.log(f"Model saved to: {effective_output}")
        logger.log("=" * 60)

        status_data["status"] = "SUCCESS"
        status_data["stage"] = "FINISHED"
        status_data["completed_at"] = datetime.datetime.now().isoformat()
        status_data["final_model_dir"] = str(effective_output)
        update_status_file(status_path, status_data)
        return 0

    except KeyboardInterrupt:
        logger.log("Pipeline interrupted by user.", prefix="[WARN]")
        status_data["status"] = "CANCELLED"
        update_status_file(status_path, status_data)
        return 130
    except Exception as e:
        logger.log(f"Pipeline encountered unexpected exception: {e}", prefix="[FATAL]")
        status_data["status"] = "FAILED"
        status_data["error"] = str(e)
        update_status_file(status_path, status_data)
        return 1
    finally:
        set_sleep_prevention(False)
        logger.log("Sleep prevention released. Guardian terminated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 Dual Arm Overnight Pipeline Guardian")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/a3_single_box_same_column_100"),
        help="Path to LeRobot v3 dataset root",
    )
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=100,
        help="Target number of successful episodes to collect",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/a3-single-box-same-column-100",
        help="Dataset repo-id",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/overnight_smolvla"),
        help="Model output directory",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=20000,
        help="Training iterations",
    )
    parser.add_argument(
        "--policy-type",
        choices=("act", "smolvla", "diffusion"),
        default="act",
        help="Policy architecture to train (default: act)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="ACT chunk size and action horizon (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8 for ACT, 4 for SmolVLA)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default: 0 on Windows, 4 on Linux/servers)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Hardware accelerator to train on",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum auto-retries on transient collection crashes",
    )
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="0-9",
        help="Evaluation seeds range or list (default: 0-9)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to checkpoint directory to resume from (e.g. checkpoints/005000)",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=1800,
        help="Maximum simulator steps during evaluation (default: 1800)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training closed-loop evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_pipeline(parse_args()))
