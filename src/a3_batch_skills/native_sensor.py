"""Additional depth rasterization of the three existing training cameras.

The policy RGB renderer and legacy target-body mounting are never changed.
Calibration is cloned before randomized reset; FK consumes robot encoders only.
"""
import copy
import numpy as np
from .markers import CAMERAS
from .rgbd import RGBDFrame,RGBDViews


class NativeThreeRGBD:
    def __init__(self,env,resolution=512):
        import mujoco
        from a3_dual_arm_sim.contracts import LEFT_JOINTS,RIGHT_JOINTS
        self.env=env;self.model=copy.copy(env.model);self.data=mujoco.MjData(self.model)
        self.addresses=[env._qpos_ids[n] for n in (*LEFT_JOINTS,*RIGHT_JOINTS)]
        self.fingers=[[int(self.model.jnt_qposadr[j]) for j in env._finger_joints[s]] for s in ('L','R')]
        self.ids={n:int(env.model.camera(n).id) for n in CAMERAS}
        self.renderer=None;self.resolution=resolution
        self.protocol=dict(sensor='native_three_legacy_training_mounts',view_count=3,
            cameras=list(CAMERAS),resolution=resolution,additional_virtual_cameras=0,
            policy_rgb_changed=False,rigid_mount_migration=False,real_hardware_validated=False,
            calibration='pre_reset_nominal_camera_parameters_plus_robot_encoders',
            object_truth_access=False,segmentation_ids_used=False)

    def capture(self,observation):
        import mujoco
        if self.renderer is None:
            render_model=copy.copy(self.env.model)
            render_model.vis.global_.offwidth=max(render_model.vis.global_.offwidth,self.resolution)
            render_model.vis.global_.offheight=max(render_model.vis.global_.offheight,self.resolution)
            render_model.vis.quality.offsamples=0
            self.renderer=mujoco.Renderer(render_model,height=self.resolution,width=self.resolution)
        state=np.asarray(observation['observation.state'])
        if state.shape!=(16,) or not np.isfinite(state).all():raise ValueError('Invalid robot encoders')
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.addresses]=np.r_[state[:7],state[8:15]]
        for ids,opening in zip(self.fingers,(state[7],state[15])):
            self.data.qpos[ids]=(1-float(opening))*self.env.GRIPPER_RANGE_M
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_camlight(self.model,self.data)
        frames=[]
        for name,cid in self.ids.items():
            self.renderer.disable_depth_rendering();self.renderer.update_scene(self.env.data,camera=cid)
            rgb=self.renderer.render().copy();flags=self.renderer.scene.flags.copy()
            try:
                self.renderer.enable_depth_rendering();depth=self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering();self.renderer.scene.flags[:]=flags
            depth[(depth<.05)|(depth>2)|~np.isfinite(depth)]=np.nan
            h,w=depth.shape;focal=h/(2*np.tan(np.deg2rad(self.model.cam_fovy[cid])/2))
            intrinsic=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1.]])
            transform=np.eye(4);transform[:3,3]=self.data.cam_xpos[cid]
            transform[:3,:3]=self.data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.])
            frames.append(RGBDFrame(rgb,depth,intrinsic,transform,float(self.env.data.time),
                'legacy_nominal_robot_FK_'+name))
        return RGBDViews(tuple(frames))

    def close(self):
        if self.renderer:self.renderer.close()
