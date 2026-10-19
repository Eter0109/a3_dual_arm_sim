"""Export verified short clips into an independent left8 LeRobot v3 dataset."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from .markers import CAMERAS

DATASET_CAMERAS = {"front":"base_0_rgb", "left_wrist":"left_wrist_0_rgb", "right_wrist":"right_wrist_0_rgb"}


def action_chunk_indices(length, start, horizon):
    """Clip-local indexing: padding repeats the final action and is masked out."""
    if length < 1 or not 0 <= start < length or horizon < 1:
        raise ValueError("Invalid action chunk boundary")
    indices = np.arange(start, start+horizon)
    return np.minimum(indices, length-1), indices >= length


def rgb_frames(path):
    import av
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def export_clips(inputs, output, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    clips = []
    for root in inputs:
        for path in sorted(Path(root).glob("batch_*/metadata.json")):
            meta = json.loads(path.read_text())
            if meta["clip_success"]:
                clips.append((path.parent, meta))
    if not clips:
        raise ValueError("No verified successful clips; failures are not imitation labels")
    if len({p.resolve() for p, _ in clips}) != len(clips):
        raise ValueError("Duplicate source clips")
    signatures = {(m["fps"],m["goal_conditioning"],m["state_dim"],m["action_dim"]) for _,m in clips}
    if len(signatures) != 1 or next(iter(signatures))[2:] != (8,8):
        raise ValueError("Mixed conditioning, frame rate, or non-left8 clips")
    fps, conditioning, _, _ = next(iter(signatures))
    first = next(rgb_frames(clips[0][0]/"front.mkv"))
    h, w = first.shape[:2]
    features = {"observation.state":dict(dtype="float32", shape=(8,), names=[f"left_{i}" for i in range(8)]),
                "action":dict(dtype="float32", shape=(8,), names=[f"left_{i}" for i in range(8)])}
    for key in DATASET_CAMERAS.values():
        features["observation.images."+key] = dict(dtype="video", shape=(h,w,3), names=["height","width","channels"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite an existing dataset")
    kwargs = dict(repo_id=repo_id, root=output, fps=int(fps), robot_type="A3_left8_batch_skill",
                  features=features, use_videos=True, image_writer_threads=3)
    if "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["vcodec"] = "h264"
    dataset = LeRobotDataset.create(**kwargs)
    provenance = []
    try:
        for path, meta in clips:
            with np.load(path/"trajectory.npz", allow_pickle=False) as trace:
                state, action = trace["state"], trace["action"]
                steps = trace["global_step"]
                n = meta["frames"]
                if state.shape != (n,8) or action.shape != (n,8) or not np.all(np.diff(steps)==1):
                    raise ValueError("Clip state/action/time alignment failed")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError("Non-finite demonstration values")
                videos = {name:rgb_frames(path/(name+".mkv")) for name in CAMERAS}
                for i in range(n):
                    frame = {"observation.state":state[i], "action":action[i], "task":meta["task"]}
                    for name, frames in videos.items():
                        image = next(frames)
                        if image.shape != (h,w,3):
                            raise ValueError("Inconsistent camera resolution")
                        frame["observation.images."+DATASET_CAMERAS[name]] = image
                    dataset.add_frame(frame)
                if any(next(frames, None) is not None for frames in videos.values()):
                    raise ValueError("RGB and applied action counts differ")
                dataset.save_episode(parallel_encoding=True)
                provenance.append(dict(source_clip=str(path.resolve()), **meta))
    finally:
        dataset.finalize()
    (output/"a3_skill_metadata.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    (output/"goal_conditioning.json").write_text(json.dumps(dict(goal_conditioning=conditioning,
        state_dim=8,action_dim=8,localization="simulator_truth",training_episodes="one verified batch per episode",
        note="Copy this descriptor only alongside weights actually trained on these goal-conditioned images."),indent=2),encoding="utf-8")
    report = dict(episodes=len(provenance),frames=sum(m["frames"] for m in provenance),
                  entry_kinds=sorted({m["entry_kind"] for m in provenance}), goal_conditioning=conditioning,
                  state_dim=8,action_dim=8,failed_clips_exported=0)
    (output/"export_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True, help="Directories containing batch_*/metadata.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-id", default="local/a3_goal_batch_left8")
    a = p.parse_args()
    print(json.dumps(export_clips(a.inputs, a.output, a.repo_id),indent=2))


if __name__ == "__main__":
    main()
