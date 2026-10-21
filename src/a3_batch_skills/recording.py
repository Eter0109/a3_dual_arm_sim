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
    def __init__(self, root, env, *, seed, marked=True, video_factory=LosslessRGBVideo, provenance=None,
                 visual_sensor=None,visual_spec=None,source_column="auto",record_geometry=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.env, self.seed, self.marked = env, seed, marked
        self.video_factory = video_factory
        self.provenance = json.loads(json.dumps(provenance or {}))
        self.active = None
        self.pending = None
        self.clips = []
        self.visual_sensor,self.visual_spec=visual_sensor,visual_spec
        self.source_column=source_column
        self.visual_completed=[]
        self.record_geometry=record_geometry;self.geometry_context=None
        if record_geometry:
            if visual_sensor is None:raise ValueError('Geometry needs calibrated RGB-D')
            from .geometry_context import GeometryContext
            self.geometry_context=GeometryContext(visual_spec)
            self.geometry_context.scene_provider=lambda:self._measured_scene
        if visual_sensor is not None:
            from .rgbd_localization import RGBDLocalizer
            self.visual_localizer=RGBDLocalizer(visual_spec)
            if not marked:raise ValueError("RGB-D goal recording requires the same marked images used at inference")

    def prepare(self, observation, goal, scene, global_step, phase):
        frame=None
        if self.pending is not None:
            raise RuntimeError("A prepared observation has no applied action")
        if self.active is None:
            path = self.root/f"batch_{goal.index:02d}"
            path.mkdir(exist_ok=False)
            self.active = dict(path=path, goal=goal, rows=[], videos={}, first_global_step=global_step)
            if self.visual_sensor is not None:
                from .visual_goals import make_visual_goal
                frame=self.visual_sensor.capture()
                from .rgbd import save_rgbd_packet
                save_rgbd_packet(path/"rgbd_entry.npz",frame)
                visual_scene=self.visual_localizer.observe(frame)
                visual=make_visual_goal(frame,visual_scene,self.visual_spec,goal.index,column=self.source_column)
                # The teacher may use truth to generate actions. Check its goal
                # agrees AFTER independent visual selection; do not generate
                # neural inputs by copying the teacher's object positions.
                from scipy.optimize import linear_sum_assignment
                actual=np.asarray([scene.cookies[i].position for i in goal.cookie_ids])
                distance=np.linalg.norm(visual.source_positions[:,None]-actual[None],axis=2)
                a,b=linear_sum_assignment(distance)
                if np.max(distance[a,b])>.003:
                    raise RuntimeError("Independent RGB-D goal differs from the teacher's selected five")
                self.active.update(visual_goal=visual,teacher_goal_match_max_m=float(distance[a,b].max()))
                if self.geometry_context:self.geometry_context.start(visual)
        if self.active["goal"].index != goal.index:
            raise RuntimeError("A batch clip was not finalized at its boundary")
        if self.visual_sensor is not None:
            from .visual_goals import annotate_visual_goal,camera_calibrations_from_fk
            obs=annotate_visual_goal(observation,camera_calibrations_from_fk(self.env),self.active["visual_goal"])
        else:
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
        if self.geometry_context:
            from .geometry_context import robot_state_from_env
            from .rgbd_localization import LocalizationFailure
            if frame is None:frame=self.visual_sensor.capture()
            try:self._measured_scene=self.visual_localizer.observe(frame)
            except LocalizationFailure:self._measured_scene=None
            self.geometry_context.update(frame,robot_state_from_env(self.env))
            self.pending['geometry']=self.geometry_context.build()

    def commit(self, applied_left8, *, batch_completed=False, failed=False, reason=None,native_count=None):
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
        if native_count is not None:self.active['native_count']=int(native_count)
        self.pending = None
        if batch_completed or failed:
            self.finish(batch_completed, reason)

    def finish(self, success, reason=None):
        if self.active is None:
            return
        if self.pending is not None:
            raise RuntimeError("Cannot finalize a clip before applying its pending action")
        active = self.active
        if self.record_geometry and success and active.get('native_count',0)!=5*(active['goal'].index+1):
            success=False;reason='Native upright/contact count did not retain this and previous batches'
        visual_verified=False
        if self.visual_sensor is not None and success:
            from .visual_goals import visual_region_complete
            from .rgbd_localization import LocalizationFailure
            try:
                frame=self.visual_sensor.capture()
                from .rgbd import save_rgbd_packet
                save_rgbd_packet(active["path"]/"rgbd_boundary.npz",frame)
                scene=self.visual_localizer.observe(frame)
                visual_verified=all(visual_region_complete(scene,g)
                    for g in [*self.visual_completed,active["visual_goal"]])
                active["visual_target_count"]=len(scene.target_cookies)
            except LocalizationFailure as exc:
                reason="RGB-D boundary not observable: "+str(exc)
            success=bool(visual_verified)
            if success:self.visual_completed.append(active["visual_goal"])
            elif reason is None:reason="RGB-D could not verify the placed batch and previous progress"
        for video in active["videos"].values():
            video.close()
        rows = active["rows"]
        if rows:
            keys = ("state", "action", "phase", "global_step", "source_frame", "relative_tool_pose")
            if self.record_geometry:keys+=('geometry',)
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
            if self.visual_sensor is not None:
                from .visual_goals import VISUAL_MARKER_VERSION
                from .rgbd_localization import LOCALIZER_VERSION
                from dataclasses import asdict
                from .rgbd_audit import json_value
                meta.update(goal_conditioning=VISUAL_MARKER_VERSION,localization=LOCALIZER_VERSION,
                    camera_annotations="calibrated_projection_of_latched_rgbd_goal",
                    visual_boundary_verified=visual_verified,
                    visual_target_count=active.get("visual_target_count"),
                    teacher="privileged_contact_expert_actions_only",
                    teacher_goal_match_max_m=active["teacher_goal_match_max_m"],
                    visual_goal=json.loads(json.dumps(asdict(active["visual_goal"]),default=json_value)),
                    rgbd_sensor=self.visual_sensor.protocol,geometry_spec=self.visual_spec,
                    auxiliary_pose_fields="teacher_diagnostics_not_policy_inputs")
                if self.record_geometry:
                    from .geometry_context import GEOMETRY_VERSION,GEOMETRY_DIM
                    meta.update(geometry_conditioning=GEOMETRY_VERSION,geometry_dim=GEOMETRY_DIM,
                        per_frame_geometry=True,geometry_source='calibrated_RGBD_and_robot_FK_only',
                        clip_role='teacher_bootstrap',native_count_at_accept=active.get('native_count'))
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
