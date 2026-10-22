"""Adapter to the pinned, previously validated A3 + LeRobot stack."""
import hashlib
import importlib.metadata
import os
import random
import sys
from pathlib import Path
import numpy as np
from .protocol import prompt


def setup(job):
    job = Path(job)
    commit = (job/"benchmark_main_29fe90e/SOURCE_COMMIT.txt").read_text().strip()
    if commit != "29fe90e423259f388ebc061926c3ec860b9edc74":
        raise ValueError("Native benchmark source commit is not the pinned research baseline")
    # The server's default ~/.cache path can be a stale symlink after restart.
    # Keep all runtime caches on the data disk, as in the original training run.
    os.environ.setdefault("HF_HOME", str(job/"hf"))
    os.environ["HF_DATASETS_CACHE"] = str(job/"cache/continuability_datasets")
    (job/"cache/continuability_datasets").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(job / "benchmark_main_29fe90e" / "src"))
    sys.path.insert(0, str(job / "benchmark_research_v3"))
    from determinism import configure_determinism
    configure_determinism()
    import torch
    torch.set_num_threads(4)
    return job


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_environment(job, task, *, render=True, max_steps=None, render_protocol='native',
                     profile='basic', camera_profile='dataset'):
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from camera_alignment import apply_camera_profile
    options = {"first_grasp": 5} if task == "2x10" else {}
    bench = make_cookie_benchmark(task, profile=profile, source_column="1", **options,
                                  max_steps=max_steps or {"2x5": 1500, "2x10": 3000}[task])
    apply_camera_profile(bench, camera_profile, Path(job)/"artifacts/training_camera_profile.json")
    env = bench.create_env(render_cameras=render)
    from .rendering import configure_rendering
    configure_rendering(env, render_protocol)
    return bench, env


def reset_environment(bench, env, seed):
    return env.reset(seed=seed, options={
        "randomize_cookies": bench.randomize_cookies, "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes, "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m, "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic", "appearance_seed": seed,
        "randomization_settings": bench.randomization_settings,
    })


def load_policy(job, checkpoint):
    import torch
    from pi05_adapter import Pi05BenchmarkPolicy
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 14 * 1024**3:
        raise RuntimeError("At least 14 GiB free GPU memory is required")
    policy = Pi05BenchmarkPolicy(checkpoint, job)
    policy.config.n_action_steps = 8
    return policy


def reset_policy(policy, seed, task):
    from a3_dual_arm_sim.contracts import EpisodeContext
    policy.reset(EpisodeContext(seed=seed, task=prompt(task), action_mode="joint_position"))


def checkpoint_hash(checkpoint):
    digest = hashlib.sha256()
    for path in sorted(Path(checkpoint).iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def dependencies(job):
    job = Path(job)
    files = [job/"repo/scripts/train_pi05_launcher.py", job/"artifacts/training_camera_profile.json"]
    files += [job/"benchmark_research_v3"/name for name in
              ("pi05_adapter.py", "camera_alignment.py", "segment_resync.py", "determinism.py")]
    files += [job/"benchmark_main_29fe90e"/name for name in
              ("SOURCE_COMMIT.txt", "src/a3_dual_arm_sim/config.py",
               "src/a3_dual_arm_sim/tasks/cookie_transfer.py", "src/a3_dual_arm_sim/tasks/cookie_2x10.py")]
    result = {str(path.relative_to(job)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result["versions"] = {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "lerobot", "mujoco", "datasets", "av")}
    return result


def contact_evidence(env, obs):
    from segment_resync import read_release_evidence
    return read_release_evidence(env, obs)


def features(policy, obs, task):
    """Frozen vision/language/state prefix, with no privileged state channels."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from pi05_adapter import observation_batch
    batch = policy.pre(observation_batch(obs, task))
    images, image_masks = policy.policy._preprocess_images(batch)
    core = policy.policy.model
    core._a3_state = batch["observation.state"]
    with torch.no_grad():
        prefix, mask, attention = core.embed_prefix(images, image_masks,
                                            batch["observation.language.tokens"],
                                            batch["observation.language.attention_mask"])
        attention_4d = core._prepare_attention_masks_4d(make_att_2d_masks(mask, attention))
        position = torch.cumsum(mask, dim=1)-1
        core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        (context, _), _ = core.paligemma_with_expert.forward(attention_mask=attention_4d,
            position_ids=position, inputs_embeds=[prefix, None], use_cache=False)
        pooled = (context.float()*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
    return pooled[0].cpu().numpy()
