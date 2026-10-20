"""Optional hybrid placement: VLA transport, measured visual servo, VLA restart.

This is explicitly NOT a pure-VLA result. Only sensor geometry and robot FK
drive the correction. Missing/stale geometry cannot authorize release.
"""
from dataclasses import dataclass,replace
from collections import deque
import numpy as np
from .rgbd_held import HeldBatchLocalizer
from .rgbd_localization import LocalizationFailure

PLACEMENT_VERSION="rgbd_pre_release_servo_v3_filtered_grasp"


class GraspPoseFilter:
    """Short robust filter in tool coordinates, updated only with fresh RGB-D."""
    def __init__(self):self.rows=deque(maxlen=5);self.timestamp=None

    def update(self,pose,robot):
        if self.timestamp is not None and pose.timestamp<=self.timestamp+1e-6:
            raise ValueError('Repeated pose cannot update the grasp filter')
        if self.timestamp is not None and pose.timestamp-self.timestamp>.15:self.rows.clear()
        self.timestamp=pose.timestamp
        offset=(pose.center-robot.tool_position)@robot.tool_rotation
        relative=robot.tool_rotation.T@pose.rotation
        self.rows.append((offset,relative))
        offsets=np.array([x[0] for x in self.rows]);rotations=np.array([x[1] for x in self.rows])
        distance=np.array([[np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)) for b in rotations] for a in rotations])
        costs=distance.sum(1)
        index=np.flatnonzero(costs<=costs.min()+1e-9)[-1]
        rotation=robot.tool_rotation@rotations[index]
        center=robot.tool_position+robot.tool_rotation@np.median(offsets,axis=0)
        local=(pose.positions-pose.center)@pose.rotation
        return replace(pose,center=center,rotation=rotation,positions=local@rotation.T+center)


@dataclass(frozen=True)
class RobotPlacementState:
    tool_position: np.ndarray
    tool_rotation: np.ndarray
    pad_midpoint: np.ndarray
    jaw_gap_m: float
    opening: float
    timestamp: float


def relative_goal_features(target,rotation,held,robot):
    """18 separate geometry features; never appended to left8 proprioception.

    Translations in tool frame / 0.1 m (6), first two columns of target and
    observed object rotations in tool frame (12). Not fed to current weights.
    """
    return np.r_[(target-robot.tool_position)@robot.tool_rotation/.1,
                 (held.center-robot.tool_position)@robot.tool_rotation/.1,
                 (robot.tool_rotation.T@rotation)[:,:2].T.ravel(),
                 (robot.tool_rotation.T@held.rotation)[:,:2].T.ravel()]


class MeasuredPlacementServo:
    def __init__(self,servo,half_size=(.025,.0031666667,.0125)):
        self.servo=servo;self.half_size=np.asarray(half_size,dtype=float)

    def start(self,held,robot,target,rotation,remaining_steps=600):
        self.stage="LIFT";self.steps=self.stage_steps=self.stable=self.missing=0
        self.failure=None;self.opening=robot.opening;self._last_timestamp=None
        self.clearance_z=max(held.center[2],target[2]+.060)
        self.lift_xy=held.center[:2].copy();self.lift_rotation=held.rotation.copy()
        self.events=[];self.realignments=0;self._descent_history=[]
        self.drift_observations=0;self.diagnostics={}
        self.pose_filter=GraspPoseFilter()
        travel=np.linalg.norm(held.center[:2]-target[:2])/.0008+(self.clearance_z-target[2])/.0003
        self.step_budget=max(1,min(int(remaining_steps),600,int(np.ceil(travel+180))))

    def transition(self,stage):
        self.events.append(dict(step=self.steps,from_stage=self.stage,to_stage=stage))
        self.stage=stage;self.stage_steps=self.stable=0;self._descent_history=[]
        self.drift_observations=0

    def act(self,held,robot,target,rotation):
        self.steps+=1;self.stage_steps+=1
        self.diagnostics=dict(stage=self.stage,decision="hold")
        if self.steps>self.step_budget:self.failure=f"Visual placement exceeded its {self.step_budget}-step geometry budget"
        if self.failure:return self.servo.hold(self.opening)
        if self.stage=="OPEN":
            self.opening=min(self.release_opening,self.opening+.010)
            if self.opening>=self.release_opening and self.stage_steps>=15:
                self.retract_position=robot.tool_position+np.array([0.,0.,.160])
                self.retract_rotation=robot.tool_rotation.copy();self.transition("RETRACT")
            return self.servo.hold(self.opening)
        if self.stage=="RETRACT":
            command,reached=self.servo.move(self.retract_position,self.retract_rotation,self.opening,speed=.003)
            if reached:self.transition("WAIT_BOUNDARY")
            return command
        if self.stage=="WAIT_BOUNDARY":return self.servo.hold(self.opening)
        fresh=(held is not None and 0<=robot.timestamp-held.timestamp<=.075
               and (self._last_timestamp is None or held.timestamp>self._last_timestamp+1e-6))
        if not fresh:
            self.missing+=1;self.stable=0
            self.drift_observations=0;self._descent_history=[]
            self.diagnostics['decision']='hold_missing_or_repeated_pose'
            if self.missing>=30:self.failure="No fresh supported held-object pose; release blocked"
            return self.servo.hold(self.opening)
        self.missing=0;self._last_timestamp=held.timestamp
        raw=held
        raw_xy=float(np.linalg.norm(raw.center[:2]-target[:2]))
        raw_angle=float(np.arccos(np.clip((np.trace(rotation.T@raw.rotation)-1)/2,-1,1)))
        held=self.pose_filter.update(raw,robot)
        xy=float(np.linalg.norm(held.center[:2]-target[:2]))
        angle=float(np.arccos(np.clip((np.trace(rotation.T@held.rotation)-1)/2,-1,1)))
        self.diagnostics.update(xy_error_m=xy,angle_error_deg=float(np.rad2deg(angle)),
                                orientation_support_deg=raw.orientation_support_deg,
                                raw_xy_error_m=raw_xy,raw_angle_error_deg=float(np.rad2deg(raw_angle)),
                                geometry_step_budget=self.step_budget,
                                height_above_goal_m=float(held.center[2]-target[2]))
        if self.stage=="DESCEND" and (max(xy,raw_xy)>.004 or max(angle,raw_angle)>np.deg2rad(3)):
            # A single noisy pose may stop descent but must not command a full
            # retreat. Keep the gripper closed while confirming persistent drift.
            self.drift_observations+=1;self.stable=0;self._descent_history=[]
            self.diagnostics.update(decision='hold_recheck_drift',drift_observations=self.drift_observations)
            if self.drift_observations<3:return self.servo.hold(self.opening)
            self.realignments+=1
            if self.realignments>2:
                self.failure="Repeated drift during descent; release blocked";return self.servo.hold(self.opening)
            self.lift_xy=held.center[:2].copy();self.lift_rotation=held.rotation.copy();self.transition("LIFT")
        elif self.stage=="DESCEND":self.drift_observations=0
        if self.stage=="LIFT":
            desired=np.r_[self.lift_xy,self.clearance_z];desired_rotation=self.lift_rotation
            if abs(held.center[2]-self.clearance_z)<.0025:
                self.stable+=1
                if self.stable>=2:self.transition("ALIGN")
        elif self.stage=="ALIGN":
            desired=np.r_[target[:2],self.clearance_z];desired_rotation=rotation
            aligned=max(xy,raw_xy)<.0025 and max(angle,raw_angle)<np.deg2rad(2) and abs(held.center[2]-self.clearance_z)<.003
            self.stable=self.stable+1 if aligned else 0
            if self.stable>=3:self.transition("DESCEND")
        else:
            descent_step=.0012 if held.center[2]-target[2]>.020 else .0006
            desired=target.copy();desired[2]=max(target[2]+.0005,held.center[2]-descent_step)
            desired_rotation=rotation
            self._descent_history.append(float(held.center[2]))
            if len(self._descent_history)>20:self._descent_history.pop(0)
            if (len(self._descent_history)==20 and held.center[2]-target[2]>.004
                    and self._descent_history[0]-self._descent_history[-1]<.001):
                self.failure="Measured descent stalled; release blocked";return self.servo.hold(self.opening)
            placed=(max(xy,raw_xy)<.0025 and max(angle,raw_angle)<np.deg2rad(2)
                    and max(abs(held.center[2]-target[2]),abs(raw.center[2]-target[2]))<.0012)
            self.stable=self.stable+1 if placed else 0
            if self.stable>=3:
                jaw=robot.tool_rotation[:,0]
                offsets=held.positions@jaw
                radius=np.abs(jaw@held.rotation)@self.half_size
                width=float(offsets.max()-offsets.min()+2*radius)
                self.release_opening=max(self.opening,self.servo.opening_for_width(width,margin=.006))
                self.transition("OPEN")
        # Re-estimate the measured object-to-tool transform each update. A
        # grasp transform latched once is insufficient when cookies slip.
        relative=robot.tool_rotation.T@held.rotation
        tool_rotation=desired_rotation@relative.T
        offset=robot.tool_rotation.T@(held.center-robot.tool_position)
        tool_position=desired-tool_rotation@offset
        self.diagnostics.update(decision='move',command_stage=self.stage)
        return self.servo.move(tool_position,tool_rotation,self.opening,
                               speed=(.0012 if held.center[2]-target[2]>.020 else .0006)
                               if self.stage=="DESCEND" else .002)[0]


class RGBDPlacementPolicy:
    """Wrap VisualGoalPolicy without changing its learned weights or inputs."""
    def __init__(self,policy,servo,spec):
        self.policy,self.servo,self.spec=policy,servo,spec
        self.localizer=HeldBatchLocalizer(spec)
        self.scene_provider=lambda:None
        self.budget_provider=lambda:600
        self.goal=self.robot=self.frame=self.placement=None
        self.phase="INACTIVE";self.failure=None;self.wait=0
        self.counts=dict(vla=0,visual_servo=0,hold_for_measurement=0)
        self.events=[];self.trace=[];self.last_error=None;self.last_pose=None

    def update(self,frame,robot):self.frame,self.robot=frame,robot

    @property
    def boundary_ready(self):
        # Finish the measured retreat before the supervisor pauses all motion.
        # A momentarily visible box at 120 mm must not truncate the 160 mm lift.
        return self.placement is not None and self.placement.stage=='WAIT_BOUNDARY'

    def start(self,goal,context=None):
        self.goal=goal;self.context=context;self.failure=None;self.wait=0;self.placement=None
        self.phase="VLA";self.policy.start(goal,context)

    def pause(self,context=None):
        self.policy.pause(context);self.phase="BOUNDARY_PAUSED"

    def act(self,observation,calibrations):
        robot=self.robot;scene=self.scene_provider()
        target=self.goal.target_position+self.goal.target_rotation@self.goal.target_positions_box.mean(0)
        rotation=self.goal.target_rotation
        current=(scene is not None and 0<=robot.timestamp-scene.timestamp<=.075)
        if current:
            target=scene.target_box.position+scene.target_box.rotation@self.goal.target_positions_box.mean(0)
            rotation=scene.target_box.rotation
        near=np.linalg.norm(robot.pad_midpoint[:2]-target[:2])<.075
        width=10*self.spec['cookie_half_size'][1]
        closed=width-.0035<robot.jaw_gap_m<width+.002
        if self.phase=="VLA":
            if near and closed and robot.pad_midpoint[2]>target[2]+.065:
                self.policy.pause(self.context);self.phase="WAIT_VIEW";self.wait=0
                self.events.append(dict(timestamp=robot.timestamp,event="pause_vla_before_descent",batch=self.goal.index))
            else:
                action=self.policy.act(observation,calibrations)
                if near and closed and np.asarray(action)[7]>robot.opening+.020:
                    self.policy.pause(self.context);self.phase="WAIT_VIEW";self.wait=0
                    self.events.append(dict(timestamp=robot.timestamp,event="block_unverified_release",batch=self.goal.index))
                else:self.counts['vla']+=1;return action
        pose=None
        if self.frame is not None and current:
            try:
                pose=self.localizer.observe(self.frame,robot.pad_midpoint,rotation)
                self.last_error=None;self.last_pose=pose
            except LocalizationFailure as exc:self.last_error=str(exc)
        if self.phase=="WAIT_VIEW":
            self.wait+=1
            if pose is None:
                self.counts['hold_for_measurement']+=1
                if self.wait>=40:self.failure="Cannot measure the held batch before placement: "+str(self.last_error)
                return self.servo.hold(robot.opening)
            self.placement=MeasuredPlacementServo(self.servo,self.spec['cookie_half_size'])
            self.placement.start(pose,robot,target,rotation,remaining_steps=self.budget_provider());self.phase="VISUAL_SERVO"
            self.events.append(dict(timestamp=robot.timestamp,event="measured_visual_servo_started",batch=self.goal.index,
                                    visible_cookies=pose.visible_count,formation_error_m=pose.formation_error_m))
        if not current and self.placement.stage not in ("OPEN","RETRACT","WAIT_BOUNDARY"):
            pose=None
        action=self.placement.act(pose,robot,target,rotation)
        self.failure=self.placement.failure;self.counts['visual_servo']+=1
        self.trace.append(dict(timestamp=robot.timestamp,batch=self.goal.index,stage=self.placement.stage,
            measured_pose=pose is not None,perception_error=self.last_error,
            servo_diagnostics=dict(self.placement.diagnostics),
            relative_geometry_18d=relative_goal_features(target,rotation,pose,robot).tolist() if pose is not None else None,
            state_left8=np.asarray(observation['observation.state'])[:8].tolist(),
            requested_left8=np.asarray(action)[:8].tolist(),target_world=target.tolist()))
        return action
