"""Sensor-only geometry, identical before-action representation in train/deploy.

Three poses in the current tool frame: latched source, measured box goal, and
visible five-object group near the fingers. The last is not proof of a grasp.
Each pose has translation / 0.1 m and two rotation columns, followed by three
validity flags. No teacher phase, object ID, contact or simulator pose enters.
"""
import numpy as np
from .rgbd_localization import LocalizationFailure
from .rgbd_held import HeldBatchLocalizer

GEOMETRY_VERSION='a3_relative_geometry30_v1'
GEOMETRY_KEY='observation.geometry'
GEOMETRY_DIM=30
TRANSLATION_SCALE_M=.1
GEOMETRY_MAX_AGE_S=.025


def robot_state_from_env(env):
    """Robot joints/FK and clock only; no movable-object body lookups."""
    from .rgbd_placement import RobotPlacementState
    site=env._eef_sites[0]
    pads=env.data.geom_xpos[list(env._left_finger_geoms)].copy()
    gap=np.linalg.norm(pads[1]-pads[0])-2*env.config.cookie_transfer.left_finger_pad_half_thickness_m
    return RobotPlacementState(env.data.site_xpos[site].copy(),env.data.site_xmat[site].reshape(3,3).copy(),
        pads.mean(0),float(gap),float(env.last_applied_action[7]),float(env.data.time))


def pose_features(position,rotation,robot):
    relative=robot.tool_rotation.T@np.asarray(rotation)
    return np.r_[(np.asarray(position)-robot.tool_position)@robot.tool_rotation/TRANSLATION_SCALE_M,
                 relative[:,:2].T.ravel()]


def geometry_features(goal,robot,scene=None,observed=None):
    value=np.zeros(GEOMETRY_DIM,dtype=np.float32)
    if goal is None:return value
    value[:9]=pose_features(goal.source_positions.mean(0),goal.source_rotation,robot)
    value[27]=1.
    if scene is not None and 0<=robot.timestamp-scene.timestamp<=GEOMETRY_MAX_AGE_S:
        box=scene.target_box
        target=box.position+box.rotation@goal.target_positions_box.mean(0)
        value[9:18]=pose_features(target,box.rotation,robot);value[28]=1.
    if observed is not None and 0<=robot.timestamp-observed.timestamp<=GEOMETRY_MAX_AGE_S:
        value[18:27]=pose_features(observed.center,observed.rotation,robot);value[29]=1.
    if not np.isfinite(value).all():raise ValueError('Nonfinite measured geometry')
    return value


class GeometryContext:
    def __init__(self,spec):
        self.localizer=HeldBatchLocalizer(spec)
        self.frame=self.robot=self.goal=None
        self.scene_provider=lambda:None
        self.observed_provider=lambda:None
        self.last_observed=None;self.last_error=None;self.last_frame_timestamp=None

    def update(self,frame,robot):
        self.robot=robot
        if frame is not None:self.frame=frame

    def start(self,goal):
        self.goal=goal;self.last_observed=None;self.last_frame_timestamp=None

    def build(self):
        if self.robot is None:raise RuntimeError('Missing current robot FK')
        scene=self.scene_provider()
        if (self.frame is not None and self.goal is not None
                and 0<=self.robot.timestamp-self.frame.timestamp<=GEOMETRY_MAX_AGE_S
                and self.frame.timestamp!=self.last_frame_timestamp):
            self.last_frame_timestamp=self.frame.timestamp;self.last_observed=None
            reference=scene.target_box.rotation if scene is not None else self.goal.target_rotation
            try:
                external=self.observed_provider()
                self.last_observed=(external if external is not None and abs(external.timestamp-self.frame.timestamp)<1e-6
                    else self.localizer.observe(self.frame,self.robot.pad_midpoint,reference))
                self.last_error=None
            except LocalizationFailure as exc:self.last_error=str(exc)
        return geometry_features(self.goal,self.robot,scene,self.last_observed)
