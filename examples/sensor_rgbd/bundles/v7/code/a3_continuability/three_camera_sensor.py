"""Simulation driver for the three declared physical RGB-D camera positions.

Rendering necessarily touches simulation data. Calibration and robot FK use a
separate nominal model and ONLY robot encoder observations, never object qpos.
Construct BEFORE randomized reset, so mounting perturbations are not secretly
corrected using the current simulator camera pose.
"""
import copy
import numpy as np
from .sensor_boundary import CAMERAS, ExecutionPacket, SensorInputError


def freeze_camera_mounts_at_nominal_home(env):
    """Bake dataset look-at cameras into physical rigid mounts before reset.

    Uses the declared deployment pose and model dimensions, not sampled object
    or camera poses. Existing benchmark/training processes are unaffected.
    """
    import mujoco
    from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
    from a3_dual_arm_sim.sim.randomization import AppearanceRandomizer
    model = copy.copy(env.model)
    data = mujoco.MjData(model)
    addresses = [env._qpos_ids[n] for n in (*LEFT_JOINTS, *RIGHT_JOINTS)]
    data.qpos[addresses] = np.r_[env.DEPLOYMENT_HOME[:7], env.DEPLOYMENT_HOME[8:15]]
    for side in ("L", "R"):
        for jid in env._finger_joints[side]:
            data.qpos[int(model.jnt_qposadr[jid])] = 0.
    for cid in env._camera_ids.values():
        if model.cam_mode[cid] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            continue
        target = int(model.cam_targetbodyid[cid])
        if model.cam_mode[cid] != mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY or target < 0:
            raise ValueError("Only declared fixed or robot-mounted calibration targets allowed")
        while target > 0:
            for jid in range(int(model.body_jntadr[target]),
                             int(model.body_jntadr[target]+model.body_jntnum[target])):
                if int(model.jnt_qposadr[jid]) not in addresses:
                    raise ValueError("Camera aims at a non-robot dynamic object")
            target = int(model.body_parentid[target])
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    for cid in env._camera_ids.values():
        body = int(model.cam_bodyid[cid])
        rotation = data.xmat[body].reshape(3, 3).T @ data.cam_xmat[cid].reshape(3, 3)
        mujoco.mju_mat2Quat(env.model.cam_quat[cid], rotation.ravel())
        env.model.cam_mode[cid] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    # Otherwise reset would restore the former non-rigid look-at convention.
    env._appearance_randomizer = AppearanceRandomizer(env.model)


class NominalRobotCalibration:
    def __init__(self, env, *, camera_baseline=None):
        import mujoco
        from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
        self.model = copy.copy(env.model)
        if camera_baseline is not None:
            for key in ('cam_pos','cam_quat','cam_mode','cam_fovy'):
                getattr(self.model,key)[:]=camera_baseline[key]
        self.data = mujoco.MjData(self.model)
        self.arm_addresses = [env._qpos_ids[n] for n in (*LEFT_JOINTS, *RIGHT_JOINTS)]
        self.finger_addresses = [[int(self.model.jnt_qposadr[j])
            for j in env._finger_joints[side]] for side in ("L", "R")]
        self.camera_ids = dict(env._camera_ids)
        self.pending_rigid=[cid for cid in self.camera_ids.values()
            if self.model.cam_mode[cid]!=mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            and env.model.cam_mode[cid]==mujoco.mjtCamLight.mjCAMLIGHT_FIXED]
        for cid in self.camera_ids.values():
            if self.model.cam_mode[cid] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED:continue
            target=int(self.model.cam_targetbodyid[cid])
            if self.model.cam_mode[cid] != mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY or target<0:
                raise ValueError('Camera calibration must be rigid or a static world look-at')
            while target>0:
                if self.model.body_jntnum[target]:
                    raise ValueError('Camera calibration cannot track a moving target')
                target=int(self.model.body_parentid[target])
        self.pad_ids = tuple(env._left_finger_geoms)
        self.pad_half_thickness = env.config.cookie_transfer.left_finger_pad_half_thickness_m
        self.gripper_range = env.GRIPPER_RANGE_M

    def update(self, measured_state):
        import mujoco
        state = np.asarray(measured_state)
        # Right encoders are needed for the right wrist camera, never model loss.
        if state.shape != (16,) or not np.isfinite(state).all():
            raise SensorInputError("Camera FK requires both arms' measured encoders")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_addresses] = np.r_[state[:7], state[8:15]]
        for addresses, grip in zip(self.finger_addresses, (state[7], state[15])):
            self.data.qpos[addresses] = (1-float(grip))*self.gripper_range
        # Kinematics only: no dynamics, contacts, simulation object measurements.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        if self.pending_rigid:
            # main's rotated-camera protocol freezes a nominal look-at mount at
            # reset. Reconstruct its ZERO-error rigid mounting from encoders.
            # Do not copy sampled camera positions, quaternions or FOV values.
            for cid in self.pending_rigid:
                body=int(self.model.cam_bodyid[cid])
                relative=self.data.xmat[body].reshape(3,3).T@self.data.cam_xmat[cid].reshape(3,3)
                mujoco.mju_mat2Quat(self.model.cam_quat[cid],relative.ravel())
                self.model.cam_mode[cid]=mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            self.pending_rigid=[];mujoco.mj_camlight(self.model,self.data)
        cameras = {name: (self.data.cam_xpos[cid].copy(),
                         self.data.cam_xmat[cid].reshape(3, 3).copy(),
                         float(self.model.cam_fovy[cid]))
                   for name, cid in self.camera_ids.items()}
        pads = self.data.geom_xpos[list(self.pad_ids)]
        gap = float(np.linalg.norm(pads[1]-pads[0])-2*self.pad_half_thickness)
        return cameras, pads.mean(0).copy(), max(0., gap)


class ThreeCameraRGBD:
    def __init__(self, env, *, resolution=512, preserve_native=False,table_height_m=None):
        import mujoco
        if int(resolution) != resolution or resolution < 64:
            raise ValueError("Invalid RGB-D resolution")
        if not preserve_native and (env._step_count != 0 or getattr(env, "_np_random", None) is not None):
            raise ValueError("Create nominal calibration before randomized reset")
        self.env = env
        self.table_height_m=table_height_m;self.calibration_diagnostics={}
        self.refresh_calibration=True;self.mount_corrections={}
        self.handeye_ready=False;self.focal_scales={};self.nominal_frames=[]
        if preserve_native:
            # main binds external policies AFTER env.reset(). Its randomizer
            # retains the original calibrated mount arrays. Use those, never
            # today's exact perturbed camera pose, and do not rewrite main.
            baseline=env._appearance_randomizer.base
            self.calibration=NominalRobotCalibration(env,camera_baseline=baseline)
        else:
            freeze_camera_mounts_at_nominal_home(env)
            self.calibration = NominalRobotCalibration(env)
        # Additional RGB-D uses its own rasterizer; do not change main's RGB
        # renderer or model buffers. Avoid mixing foreground/background depths
        # across antialiased boundary pixels in this metric sensor.
        render_model=copy.copy(env.model) if preserve_native else env.model
        if preserve_native:render_model.vis.quality.offsamples=0
        render_model.vis.global_.offwidth = max(render_model.vis.global_.offwidth, resolution)
        render_model.vis.global_.offheight = max(render_model.vis.global_.offheight, resolution)
        self.renderer = mujoco.Renderer(render_model, height=resolution, width=resolution)
        self.protocol = dict(schema="a3_three_native_rgbd_v1", cameras=list(CAMERAS),
            resolution=resolution, additional_virtual_cameras=0,
            camera_mount='unchanged_main' if preserve_native else "rigid_at_nominal_deployment_pose",
            legacy_dataset_lookat_semantics_changed=not preserve_native,
            calibration="nominal_pre_randomization_mounts_plus_robot_encoders",
            calibration_errors_automatically_corrected=table_height_m is not None,
            gripper_signal="simulated_measured_opening_real_hardware_unconfirmed",
            contact_force_required=False, object_truth_in_control_packet=False,
            real_hardware_validated=False)
        self.protocol['rgbd_rasterizer']='independent_single_sample' if preserve_native else 'scene_renderer'
        self.protocol.update(observed_table_plane_calibration=table_height_m is not None,
            reference_table_height_m=table_height_m,
            table_reference='declared_static_calibration_measurement_not_sampled_object_pose',
            correction_observability='table normal/height plus relative camera XY/yaw; absolute reference XY/yaw remain nominal')

    def request_recalibration(self):
        self.refresh_calibration=True

    def capture(self, raw):
        from a3_batch_skills.rgbd import RGBDFrame, RGBDViews
        now = float(raw["time"])
        if abs(now-float(self.env.data.time)) > 1e-6:
            raise SensorInputError("Robot observation does not match rendered instant")
        calibrations, pad, gap = self.calibration.update(raw["observation.state"])
        frames = []
        for name in CAMERAS:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.env.data, camera=self.env._camera_ids[name])
            rgb = self.renderer.render().copy()
            flags = self.renderer.scene.flags.copy()
            try:
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering()
                self.renderer.scene.flags[:] = flags
            depth[(depth < .05) | (depth > 2.) | ~np.isfinite(depth)] = np.nan
            position, rotation, fovy = calibrations[name]
            transform = np.eye(4)
            transform[:3, :3] = rotation @ np.diag([1., -1., -1.])
            transform[:3, 3] = position
            h, w = depth.shape
            focal = h/(2*np.tan(np.deg2rad(fovy)/2))
            intrinsic = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
            frame=RGBDFrame(rgb, depth, intrinsic, transform, now,
                "nominal_mount_calibration_plus_robot_encoder_FK")
            frames.append(frame)
        nominal_frames=frames;self.nominal_frames=list(frames)
        if self.handeye_ready:
            frames=[]
            for name,frame in zip(CAMERAS,nominal_frames):
                intrinsic=frame.intrinsic.copy();intrinsic[:2,:2]*=self.focal_scales.get(name,1.)
                supported=name in self.mount_corrections
                corrected=RGBDFrame(frame.rgb,frame.depth_m if supported else np.full_like(frame.depth_m,np.nan),
                    intrinsic,frame.world_from_optical@self.mount_corrections.get(name,np.eye(4)),now,
                    'encoder_FK_and_observed_handeye_calibration' if supported else 'uncalibrated_view_masked')
                if supported and self.refresh_calibration:
                    from .plane_calibration import rectify_frame
                    try:
                        corrected,plane=rectify_frame(corrected,table_height_m=self.table_height_m)
                        self.mount_corrections[name]=np.linalg.inv(frame.world_from_optical)@corrected.world_from_optical
                        self.calibration_diagnostics[name+'_boundary_plane']=dict(residual_m=plane.residual_m,
                            tilt_degrees=plane.tilt_degrees,height_shift_m=plane.height_shift_m)
                    except ValueError:pass  # retain the last validated rigid mounting
                frames.append(corrected)
            self.refresh_calibration=False
        elif self.table_height_m is not None and self.refresh_calibration:
            frames=[]
            for name,frame in zip(CAMERAS,nominal_frames):
                from .plane_calibration import rectify_frame
                try:
                    frame,correction=rectify_frame(frame,table_height_m=self.table_height_m)
                    self.calibration_diagnostics[name]=dict(valid=True,
                        residual_m=correction.residual_m,tilt_degrees=correction.tilt_degrees,
                        height_shift_m=correction.height_shift_m,inlier_fraction=correction.inlier_fraction)
                except ValueError as exc:
                    self.calibration_diagnostics[name]=dict(valid=False,reason=str(exc))
                    if name in self.mount_corrections:
                        frame=RGBDFrame(frame.rgb,frame.depth_m,frame.intrinsic,
                            frame.world_from_optical@self.mount_corrections[name],now,'previous_validated_mount_estimate')
                    else:
                        frame=RGBDFrame(frame.rgb,np.full_like(frame.depth_m,np.nan),frame.intrinsic,
                            frame.world_from_optical,now,'uncertain_calibration_depth_masked_unknown')
                frames.append(frame)
            from .view_registration import register_views
            from a3_batch_skills.rgbd import geometry_spec_from_env
            # geometry_spec is static public dimensions, not the sampled layout.
            spec=geometry_spec_from_env(self.env)
            registered,diagnostic=register_views(RGBDViews(tuple(frames)),spec,table_z=self.table_height_m)
            frames=list(registered.frames);self.calibration_diagnostics['_registration']=diagnostic
            for name,original,corrected in zip(CAMERAS,nominal_frames,frames):
                if np.isfinite(corrected.depth_m).mean()>.1:
                    self.mount_corrections[name]=np.linalg.inv(original.world_from_optical)@corrected.world_from_optical
            self.refresh_calibration=False
        elif self.table_height_m is not None:
            frames=[]
            for name,frame in zip(CAMERAS,nominal_frames):
                if name in self.mount_corrections:
                    frames.append(RGBDFrame(frame.rgb,frame.depth_m,frame.intrinsic,
                        frame.world_from_optical@self.mount_corrections[name],now,'encoder_FK_plus_observed_mount_correction'))
                else:frames.append(RGBDFrame(frame.rgb,np.full_like(frame.depth_m,np.nan),
                    frame.intrinsic,frame.world_from_optical,now,'unknown_camera_calibration'))
        for name,frame in zip(CAMERAS,frames):
            t=frame.world_from_optical;h=frame.depth_m.shape[0]
            fy=float(np.rad2deg(2*np.arctan(h/(2*frame.intrinsic[1,1]))))
            calibrations[name]=(t[:3,3].copy(),t[:3,:3]@np.diag([1.,-1.,-1.]),fy)
        return ExecutionPacket.create(raw, RGBDViews(tuple(frames)), calibrations,
            timestamp=now, gripper_position_measured=True,
            pad_midpoint_world=pad, jaw_gap_m=gap)

    def install_handeye(self,scan):
        """Anchor the external camera using the visually measured static rim."""
        from .handeye_calibration import observe_rim,FOCAL_SCALES
        correction,scale,details=scan.solve(self.table_height_m)
        robot,frame=scan.selected_samples[-1]
        rim,_=observe_rim(frame,scan.spec,table_height=self.table_height_m,focal_scale=scale)
        anchor=frame.world_from_optical@correction@rim
        self.mount_corrections={'left_wrist':correction};self.focal_scales={'left_wrist':scale}
        external=[]
        for name,raw in zip(CAMERAS,self.nominal_frames):
            if name=='left_wrist':continue
            matches=[]
            for focal in FOCAL_SCALES:
                try:
                    measured,error=observe_rim(raw,scan.spec,table_height=self.table_height_m,focal_scale=focal)
                    corrected=anchor@np.linalg.inv(measured)
                    delta=np.linalg.inv(raw.world_from_optical)@corrected
                    from scipy.spatial.transform import Rotation
                    if np.linalg.norm(delta[:3,3])>.04 or np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())>np.deg2rad(6):continue
                    matches.append((error,focal,delta))
                except (ValueError,RuntimeError):continue
            if matches:
                error,focal,delta=min(matches,key=lambda t:t[0])
                self.mount_corrections[name]=delta;self.focal_scales[name]=float(focal)
                external.append(dict(camera=name,extent_error_m=error))
        self.handeye_ready=True
        self.calibration_diagnostics={'handeye':details,'focal_scales':self.focal_scales,'anchored_views':external}
        self.protocol.update(calibration='observed_multiview_encoder_handeye_and_known_metric_rim',
            correction_observability='metric rim with diverse robot rotations; absolute base anchoring from robot encoders',
            boundary_table_normal_height_recheck=True)
        return self.calibration_diagnostics

    def close(self):
        self.renderer.close()
