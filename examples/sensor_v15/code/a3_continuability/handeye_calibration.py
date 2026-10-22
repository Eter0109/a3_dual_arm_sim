"""Eye-in-hand calibration from encoder motions and observed metric bin rims.

No sampled camera/object pose is accepted. A static, known-size rectangular
source bin is the calibration target. Its position in the robot base is unknown.
"""
import numpy as np

FOCAL_SCALES=np.linspace(.94,1.06,25)


def observation_ready(index,age,goal_stable,encoder_stable,pad_z,rim_z):
    # Viewpoints are measured, not manipulation targets. An actually settled
    # high pose is a valid calibration sample even if its requested IK endpoint
    # is slightly unattainable. The initial safety lift must still reach its goal.
    return goal_stable>=8 or (index>0 and age>=30 and encoder_stable>=8 and pad_z>=rim_z+.065)


def transform(rotation, position):
    out=np.eye(4);out[:3,:3]=rotation;out[:3,3]=position
    return out


def solve_handeye(robot_poses, camera_targets, nominal_mount):
    import cv2
    from scipy.spatial.transform import Rotation
    if len(robot_poses)<5 or len(robot_poses)!=len(camera_targets):
        raise ValueError('At least five paired calibration views are required')
    axes=[Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec()
          for a,b in zip(robot_poses,robot_poses[1:])]
    singular=np.linalg.svd(np.asarray(axes),compute_uv=False)
    if singular[1]<.12:raise ValueError('Insufficient rotation-axis diversity')
    # Leave one view out: test a pose that did not participate in this solve.
    def fit(indices):
        a=[robot_poses[i] for i in indices];b=[camera_targets[i] for i in indices]
        r,t=cv2.calibrateHandEye([v[:3,:3] for v in a],[v[:3,3] for v in a],
            [v[:3,:3] for v in b],[v[:3,3] for v in b],method=cv2.CALIB_HAND_EYE_PARK)
        return transform(r,np.asarray(t).ravel())
    def residual(x):
        targets=[a@x@b for a,b in zip(robot_poses,camera_targets)]
        position=np.asarray([t[:3,3] for t in targets]);center=np.median(position,axis=0)
        return float(np.max(np.linalg.norm(position-center,axis=1)))
    holdout=fit(range(len(robot_poses)-1))
    x=fit(range(len(robot_poses)))
    if not np.isfinite(x).all():raise ValueError('Nonfinite hand-eye solution')
    delta=np.linalg.inv(nominal_mount)@x
    shift=float(np.linalg.norm(delta[:3,3]))
    angle=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec()))
    error=residual(x);heldout_error=residual(holdout);before=residual(nominal_mount)
    if shift>.025 or angle>np.deg2rad(5) or error>.002 or heldout_error>.003:
        raise ValueError(f'Uncertain hand-eye fit: shift={shift:.4f}, residual={error:.4f}, holdout={heldout_error:.4f}')
    if before<error*.9:raise ValueError('Calibration degrades the static-target residual')
    return x,dict(position_residual_m=error,holdout_residual_m=heldout_error,
        nominal_residual_m=before,mount_shift_m=shift,mount_rotation_deg=np.rad2deg(angle),
        axis_singular_values=singular.tolist(),views=len(robot_poses))


def observe_rim(frame,spec,*,table_height=.75,focal_scale=1.):
    from a3_batch_skills.rgbd import unproject
    from a3_batch_skills.rgbd_localization import RGBDLocalizer
    from .plane_calibration import rectify_frame
    corrected,plane=rectify_frame(frame,table_height_m=table_height,focal_scale=focal_scale)
    clone=dict(spec,target_inner_half_size=spec['source_inner_half_size'],
               wall_height_m=spec['source_wall_top_offset_m'])
    box,_=RGBDLocalizer(clone)._boxes(corrected,unproject(corrected),table_height)
    if box.supported_sides!=4:raise ValueError('Calibration requires four observed rim sides')
    world_rim=transform(box.rotation,np.r_[box.position[:2],box.rim_z])
    return np.linalg.inv(corrected.world_from_optical)@world_rim,box.extent_error_m


class HandEyeScan:
    """Bounded physical high-clearance left-wrist observation sequence."""
    def __init__(self,motion,spec,*,emit=None):
        self.motion=motion;self.spec=spec;self.emit=emit or (lambda **x:None)
        self.goals=None;self.index=0;self.age=0;self.stable=0;self.steps=0
        self.done=False;self.samples=[];self.failure=None
        self.calibration_trials=[]
        self.previous_encoder=None;self.encoder_stable=0

    def plan(self,packet):
        from scipy.spatial.transform import Rotation
        from .sensor_batch_loop import SensorBatchLoop
        from .three_view_localizer import ThreeViewLocalizer
        loop=SensorBatchLoop(self.motion,self.spec,source_column=1)
        try:loop.scene=ThreeViewLocalizer(self.spec).observe(packet.rgbd)
        except RuntimeError:
            # Calibration needs only the static source rim, not a grasp-ready
            # scene. Bootstrap from its independently observed metric dimensions.
            from types import SimpleNamespace
            from a3_batch_skills.rgbd import unproject
            from a3_batch_skills.rgbd_localization import RGBDLocalizer
            from .plane_calibration import rectify_frame
            clone=dict(self.spec,target_inner_half_size=self.spec['source_inner_half_size'],
                wall_height_m=self.spec['source_wall_top_offset_m'])
            candidates=[]
            for image in packet.rgbd.frames:
                for scale in (.96,.98,1.,1.02,1.04):
                    try:
                        image_fixed,_=rectify_frame(image,focal_scale=scale)
                        box,_=RGBDLocalizer(clone)._boxes(image_fixed,unproject(image_fixed),.75)
                        candidates.append(box)
                    except (ValueError,RuntimeError):pass
            if not candidates:raise ValueError('No observed source rim for calibration bootstrap')
            box=min(candidates,key=lambda b:b.extent_error_m+.003*(4-b.supported_sides))
            loop.scene=SimpleNamespace(source_box=box,target_box=box,source_cookies=())
        p,r=loop.observe_goal(packet)
        state=packet.observation['observation.state']
        cp,cr,_=packet.calibrations['left_wrist'];pos,rot=self.motion.pose(state)
        offset=rot.T@(cp-pos);mount=rot.T@cr
        rim_center=np.r_[loop.scene.source_box.position[:2],loop.scene.source_box.rim_z]
        goals=[]
        # Independent pitch/roll rotations make mount translation observable.
        for axis,angle,dx,dy in ((0,0,0,0),(0,.23,0,0),(1,.23,0,0),
                                  (0,-.23,0,0),(1,-.23,0,0),(0,.14,.018,-.018),
                                  (1,.14,-.018,.018)):
            v=np.zeros(3);v[axis]=angle
            camera_r=Rotation.from_rotvec(v).as_matrix()@(r@mount)
            rotation=camera_r@mount.T
            # Keep the entire static rim centered while changing optical axis.
            # The narrow entry inspection pose clips the far rim and is not a
            # valid calibration image. All translations are robot FK targets.
            camera_position=rim_center+camera_r[:,2]*.34+np.array([dx,dy,0])
            position=camera_position-rotation@offset
            pe,re=self.motion.pose_error(state,position,rotation)
            if pe<.0025 and re<.065:goals.append((position,rotation))
        if len(goals)<5:raise ValueError('Fewer than five reachable calibration views')
        # First move up with the current wrist angle before rotating above bins.
        pad_z=float(packet.pad_midpoint_world[2]);rim=max(loop.scene.source_box.rim_z,loop.scene.target_box.rim_z)
        self.rim_z=float(rim)
        lift=pos+np.array([0,0,max(0.,rim+.09-pad_z)])
        pe,re=self.motion.pose_error(state,lift,rot)
        if pe>.0025 or re>.065:raise ValueError('Calibration clearance lift unreachable')
        self.goals=[(lift,rot),*goals]
        self.emit(event='handeye_plan',views=len(goals))

    def act(self,packet,nominal_frame):
        self.steps+=1;self.age+=1
        try:
            if self.goals is None:self.plan(packet)
            if self.steps>400 or self.age>130:raise ValueError('Calibration motion budget exceeded')
            state=packet.observation['observation.state']
            stationary=self.previous_encoder is not None and np.max(abs(state[:7]-self.previous_encoder))<.0008
            self.encoder_stable=self.encoder_stable+1 if stationary else 0
            self.previous_encoder=state[:7].copy()
            action,reached=self.motion.command(state,*self.goals[self.index],1.,speed=.004)
            self.stable=self.stable+1 if reached else 0
            if observation_ready(self.index,self.age,self.stable,self.encoder_stable,
                                 float(packet.pad_midpoint_world[2]),self.rim_z):
                if self.index>0:
                    p,r=self.motion.pose(state)
                    self.samples.append((transform(r,p),nominal_frame))
                    self.emit(event='handeye_view',view=self.index,requested_pose_reached=bool(reached))
                self.index+=1;self.age=0;self.stable=0;self.encoder_stable=0
                if self.index>=len(self.goals):self.done=True
            return action
        except (ValueError,RuntimeError) as exc:
            self.failure=str(exc);self.done=True
            self.emit(event='handeye_failed',reason=self.failure)
            return self.motion.hold(1.)

    def solve(self,table_height):
        # Estimate intrinsics from repeated known-size rims, shared across poses.
        first_robot,first_frame=self.samples[0]
        nominal=np.linalg.inv(first_robot)@first_frame.world_from_optical
        trials=[]
        scales=list(FOCAL_SCALES)
        for trial_index in range(len(FOCAL_SCALES)+9):
            if trial_index==len(FOCAL_SCALES):
                if not trials:break
                best=min(trials,key=lambda t:t[0])[1]
                scales.extend(best+np.linspace(-.004,.004,9))
            scale=scales[trial_index]
            measurements=[];errors=[];robots=[];accepted=[]
            try:
                for robot,frame in self.samples:
                    try:target,error=observe_rim(frame,self.spec,table_height=table_height,focal_scale=scale)
                    except (ValueError,RuntimeError):continue
                    measurements.append(target);errors.append(error)
                    robots.append(robot);accepted.append((robot,frame))
                x,details=solve_handeye(robots,measurements,nominal)
                reconstructed=[r@x@t for r,t in zip(robots,measurements)]
                tilt=float(np.median([np.linalg.norm(t[:2,2]) for t in reconstructed]))
                details['horizontal_rim_normal_error_rad']=tilt
                # Known horizontal rim normal provides the independent constraint
                # that removes the focal-length/mount-rotation tradeoff.
                trials.append((np.mean(errors)+details['holdout_residual_m']+.34*tilt,scale,x,details,accepted))
            except (ValueError,RuntimeError) as exc:
                self.calibration_trials.append(dict(scale=float(scale),rejected=str(exc)))
                continue
        if not trials:raise ValueError('No intrinsics/hand-eye candidate passed held-out validation')
        _,scale,x,details,self.selected_samples=min(trials,key=lambda t:t[0])
        return np.linalg.inv(nominal)@x,float(scale),details
