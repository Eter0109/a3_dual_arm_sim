#!/usr/bin/env python3
"""5-Minute Full-Stack Smoke Test for Server / RTX 4090D.

Verifies:
  1. CUDA / PyTorch GPU functionality (RTX 4090D detection & tensor operations)
  2. MuJoCo headless EGL 3-camera rendering (front, left_wrist, right_wrist)
  3. LeRobot v3 dataset recording (1 full expert rollout via collect_cookie_benchmark)
  4. Dataset audit & metadata validation (via audit_training_dataset)
  5. GPU ACT training smoke run (50 steps on CUDA via train_act)
  6. Closed-loop policy inference on CUDA (via ACTPolicyPlugin)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure headless EGL on Linux
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["PYTHONUNBUFFERED"] = "1"

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def step_print(step_num: int, title: str):
    print(f"\n[{step_num}/6] >>> {title}...", flush=True)


def test_gpu() -> str:
    step_print(1, "Checking CUDA & GPU Device")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch! Please install torch with CUDA support.")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"  [+] PyTorch Version : {torch.__version__}")
    print(f"  [+] CUDA Version    : {torch.version.cuda}")
    print(f"  [+] GPU Device      : {device_name}")
    print(f"  [+] Compute Cap.    : {capability[0]}.{capability[1]}")
    print(f"  [+] VRAM Available  : {vram_gb:.2f} GB")

    # Quick tensor compute test
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    del a, b, c
    print("  [+] CUDA Matmul Tensor Compute: PASS")
    return device_name


def test_mujoco_headless_egl():
    step_print(2, "Checking MuJoCo Headless 3-Camera Rendering")
    from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

    env = A3CookieTransferEnv(render_cameras=True)
    try:
        obs, _ = env.reset(seed=42)
        for cam_key in ("observation.images.front", "observation.images.left_wrist", "observation.images.right_wrist"):
            if cam_key not in obs:
                raise KeyError(f"Missing camera observation: {cam_key}")
            img = obs[cam_key]
            if img.shape != (256, 256, 3):
                raise ValueError(f"Invalid shape for {cam_key}: {img.shape}, expected (256, 256, 3)")
            if img.max() == 0 and img.min() == 0:
                raise ValueError(f"Rendered image for {cam_key} is completely black! EGL context failed.")
            print(f"  [+] Camera {cam_key:<32}: shape={img.shape}, mean_val={img.mean():.1f}")
        print("  [+] MuJoCo Headless EGL 3-Camera Rendering: PASS")
    finally:
        env.close()


def test_dataset_collection(smoke_dir: Path) -> Path:
    step_print(3, "Collecting 1 Real Expert Episode with examples/collect_cookie_benchmark.py")
    smoke_data_dir = smoke_dir / "dataset"
    if smoke_data_dir.exists():
        shutil.rmtree(smoke_data_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        str(ROOT / "examples" / "collect_cookie_benchmark.py"),
        "--root",
        str(smoke_data_dir),
        "--repo-id",
        "local/smoke-test",
        "--episodes",
        "1",
    ]
    t0 = time.time()
    env = os.environ.copy()
    if sys.platform != "win32":
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    duration = time.time() - t0
    for line in proc.stdout.splitlines():
        if "saved" in line or "Dataset audit" in line:
            print(f"  [+] {line.strip()}")
    print(f"  [+] Real Expert Data Collection Pipeline: PASS in {duration:.1f}s")
    return smoke_data_dir


def test_dataset_audit(data_dir: Path):
    step_print(4, "Auditing Dataset Integrity & Feature Alignments")
    from a3_dual_arm_sim.learning.training import audit_training_dataset

    audit_res = audit_training_dataset(data_dir, repo_id="local/smoke-test")
    print(f"  [+] Total Episodes : {audit_res['episodes']}")
    print(f"  [+] Total Frames   : {audit_res['frames']}")
    print(f"  [+] State Dim      : {audit_res['state_dim']}")
    print(f"  [+] Action Dim     : {audit_res['action_dim']}")
    print("  [+] Dataset Audit: PASS")


def test_act_training_smoke(data_dir: Path, smoke_dir: Path) -> Path:
    step_print(5, "Running ACT Training on GPU (50 Steps Smoke Run)")
    from a3_dual_arm_sim.learning.training import train_act

    train_out_dir = smoke_dir / "act_smoke_model"
    if train_out_dir.exists():
        shutil.rmtree(train_out_dir, ignore_errors=True)

    t0 = time.time()
    res = train_act(
        dataset_root=data_dir,
        repo_id="local/smoke-test",
        output_dir=train_out_dir,
        steps=50,
        batch_size=2,
        seed=1000,
        device="cuda",
        chunk_size=50,
        n_action_steps=50,
        num_workers=0 if sys.platform == "win32" else 2,
    )
    duration = time.time() - t0
    ckpt = res["checkpoint"]
    print(f"  [+] 50 steps trained in {duration:.1f}s ({duration/50:.3f}s/step)")
    print(f"  [+] Checkpoint saved at: {ckpt}")
    return Path(ckpt)


def test_act_inference(checkpoint_dir: Path, data_dir: Path):
    step_print(6, "Testing ACT Closed-Loop Policy Rollout on CUDA")
    from a3_dual_arm_sim.policies.act import ACTPolicyPlugin
    from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

    policy = ACTPolicyPlugin(
        checkpoint=checkpoint_dir,
        dataset_root=data_dir,
        repo_id="local/smoke-test",
        device="cuda",
        temporal_ensemble_coeff=0.01,
    )
    env = A3CookieTransferEnv(render_cameras=True)
    try:
        obs, _ = env.reset(seed=1)
        policy.reset()
        for _ in range(10):
            action = policy.act(obs, "transfer 10 cookies into target box")
            if action.shape != (16,):
                raise ValueError(f"Expected action shape (16,), got {action.shape}")
            obs, _, _, _, _ = env.step(action)
        print("  [+] ACT 16D Closed-Loop Action Inference: PASS")
    finally:
        env.close()
        policy.close()


def main():
    print("=" * 68)
    print("  A3 Dual-Arm Sim: Server / Full-Stack Smoke Test")
    print("=" * 68, flush=True)

    t_start = time.time()
    smoke_dir = ROOT / "outputs" / "server_smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    try:
        device_name = test_gpu()
        test_mujoco_headless_egl()
        data_dir = test_dataset_collection(smoke_dir)
        test_dataset_audit(data_dir)
        ckpt_dir = test_act_training_smoke(data_dir, smoke_dir)
        test_act_inference(ckpt_dir, data_dir)

        # Cleanup temporary smoke test data
        shutil.rmtree(smoke_dir, ignore_errors=True)

        elapsed = time.time() - t_start
        print("\n" + "=" * 68)
        print("  SMOKE TEST PASSED SUCCESSFULLY IN " f"{elapsed:.1f}s!")
        print(f"  Target Accelerator : {device_name}")
        print("  Status             : ALL SYSTEMS READY FOR PRODUCTION TRAINING!")
        print("=" * 68 + "\n", flush=True)
        return 0

    except Exception as e:
        print(f"\n[!] SMOKE TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
