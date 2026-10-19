"""Bounded, lossless RGB clips with pre-action left8 observations and applied actions."""
import json
from pathlib import Path
import numpy as np
from .grounding import group_frame
from .markers import CAMERAS, MARKER_VERSION, MARKED_INSTRUCTION, annotate_goals


class LosslessRGBVideo:
    def __init__(self, path, fps, shape):
        import av
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264rgb", rate=int(fps))
        self.stream.height, self.stream.width = shape[:2]
        self.stream.pix_fmt = "rgb24"
        self.stream.options = {"crf": "0", "preset": "ultrafast"}

    def add(self, image):
        for packet in self.stream.encode(self.av.VideoFrame.from_ndarray(image, format="rgb24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class SkillClipRecorder:
    def __init__(self, root, env, *, seed, marked=True, video_factory=LosslessRGBVideo, provenance=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.env, self.seed, self.marked = env, seed, marked
        self.video_factory = video_factory
        self.provenance = json.loads(json.dumps(provenance or {}))
        self.active = None
        self.pending = None
        self.clips = []

    def prepare(self, observation, goal, scene, global_step, phase):
        if self.pending is not None:
            raise RuntimeError("A prepared observation has no applied action")
        if self.active is None:
            path = self.root/f"batch_{goal.index:02d}"
            path.mkdir(exist_ok=False)
            self.active = dict(path=path, goal=goal, rows=[], videos={}, first_global_step=global_step)
        if self.active["goal"].index != goal.index:
            raise RuntimeError("A batch clip was not finalized at its boundary")
        obs = annotate_goals(observation, self.env, scene, goal) if self.marked else observation
        center, rotation, _ = group_frame(scene, goal.cookie_ids)
        state = np.asarray(observation["observation.state"][:8], dtype=np.float32).copy()
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Invalid left8 state before action")
        self.pending = dict(state=state, phase=phase, global_step=global_step,
            source_frame=np.r_[center, rotation.ravel()].astype(np.float32),
            relative_tool_pose=np.r_[rotation.T @ (scene.tool_position-center),
                                    (rotation.T @ scene.tool_rotation).ravel()].astype(np.float32),
            images={name:np.asarray(obs["observation.images."+name]).copy() for name in CAMERAS})

    def commit(self, applied_left8, *, batch_completed=False, failed=False, reason=None):
        row = self.pending
        if row is None:
            raise RuntimeError("No pre-action observation was prepared")
        action = np.asarray(applied_left8, dtype=np.float32).copy()
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Invalid applied left8 action")
        row["action"] = action
        images = row.pop("images")
        for name, image in images.items():
            if name not in self.active["videos"]:
                self.active["videos"][name] = self.video_factory(self.active["path"]/(name+".mkv"),
                    self.env.config.control_hz, image.shape)
            self.active["videos"][name].add(image)
        self.active["rows"].append(row)
        self.pending = None
        if batch_completed or failed:
            self.finish(batch_completed, reason)

    def finish(self, success, reason=None):
        if self.active is None:
            return
        if self.pending is not None:
            raise RuntimeError("Cannot finalize a clip before applying its pending action")
        active = self.active
        for video in active["videos"].values():
            video.close()
        rows = active["rows"]
        if rows:
            keys = ("state", "action", "phase", "global_step", "source_frame", "relative_tool_pose")
            np.savez_compressed(active["path"]/"trajectory.npz",
                **{key:np.asarray([r[key] for r in rows]) for key in keys})
            goal = active["goal"]
            meta = dict(schema="a3_single_batch_v1", seed=self.seed, batch_index=goal.index,
                entry_kind="initial" if goal.index == 0 else "after_previous_release",
                frames=len(rows), fps=self.env.config.control_hz, clip_success=bool(success),
                failure_reason=reason, cookie_ids=list(goal.cookie_ids),
                target_positions_box=goal.target_positions.tolist(),
                first_global_step=active["first_global_step"],
                last_global_step=int(rows[-1]["global_step"]),
                task=MARKED_INSTRUCTION if self.marked else goal.instruction,
                goal_conditioning=MARKER_VERSION if self.marked else "none",
                localization="simulator_truth", camera_annotations="simulator_truth_projection" if self.marked else "none",
                state_dim=8, action_dim=8, stored_action_mode="joint_position",
                timing="observation_before_step -> applied_joint_target_at_that_step",
                physics_resets_between_batches=0, rgb_codec="lossless_libx264rgb")
            meta["provenance"] = self.provenance
            (active["path"]/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.clips.append(meta)
        self.active = None

    def close(self, task_success=False):
        # An exception before env.step completed must not create an unlabeled
        # frame or replace the original exception with a recorder exception.
        self.pending = None
        self.finish(False, "Task ended before this batch was verified")
        (self.root/"manifest.json").write_text(json.dumps(dict(schema="a3_single_batch_v1",
            task_success=bool(task_success), clips=self.clips), indent=2), encoding="utf-8")
