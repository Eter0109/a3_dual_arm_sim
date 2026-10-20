"""Calibrated RGB-D packets. This module never reads simulated object poses/IDs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    world_from_optical: np.ndarray  # optical axes: right, down, forward
    timestamp: float
    calibration_source: str = "calibrated_camera"

    def __post_init__(self):
        if (self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3
                or self.depth_m.shape != self.rgb.shape[:2]
                or not np.issubdtype(self.depth_m.dtype,np.floating)):
            raise ValueError("Expected aligned uint8 RGB and floating-point metric depth")
        if self.intrinsic.shape != (3,3) or self.world_from_optical.shape != (4,4):
            raise ValueError("Missing RGB-D calibration")
        if (not np.isfinite(self.intrinsic).all() or not np.isfinite(self.world_from_optical).all()
                or min(self.intrinsic[0,0],self.intrinsic[1,1])<=0
                or np.linalg.det(self.intrinsic)<=0
                or not np.allclose(self.intrinsic[2],[0,0,1])
                or not np.allclose(self.world_from_optical[3],[0,0,0,1])):
            raise ValueError("Invalid camera intrinsics or homogeneous transform")
        rotation = self.world_from_optical[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.999:
            raise ValueError("Camera extrinsics are not a proper rotation")
        for name in ("rgb","depth_m","intrinsic","world_from_optical"):
            value=np.asarray(getattr(self,name)).copy()
            if name=="depth_m":value[value<=0]=np.nan
            value.setflags(write=False)
            object.__setattr__(self,name,value)
        if not np.isfinite(self.timestamp):raise ValueError("Nonfinite RGB-D timestamp")
        object.__setattr__(self,"timestamp",float(self.timestamp))


@dataclass(frozen=True)
class RGBDViews:
    frames: tuple[RGBDFrame,...]

    def __post_init__(self):
        object.__setattr__(self,"frames",tuple(self.frames))
        if not self.frames or max(f.timestamp for f in self.frames)-min(f.timestamp for f in self.frames)>1e-6:
            raise ValueError("Multi-view RGB-D must describe the same physical instant")

    @property
    def timestamp(self):return self.frames[0].timestamp


def save_rgbd_packet(path,packet):
    frames=packet.frames if isinstance(packet,RGBDViews) else (packet,)
    fields={"view_count":len(frames)}
    for i,frame in enumerate(frames):
        prefix="" if i==0 else f"view_{i}_"
        for key in ("rgb","depth_m","intrinsic","world_from_optical","timestamp"):
            fields[prefix+key]=getattr(frame,key)
    np.savez_compressed(path,**fields)


def load_rgbd_packet(path):
    with np.load(path,allow_pickle=False) as data:
        count=int(data["view_count"]) if "view_count" in data else 1
        frames=[]
        for i in range(count):
            prefix="" if i==0 else f"view_{i}_"
            frames.append(RGBDFrame(**{key:data[prefix+key] for key in
                ("rgb","depth_m","intrinsic","world_from_optical","timestamp")}))
    return RGBDViews(tuple(frames)) if count>1 else frames[0]


def geometry_spec_from_env(env):
    """Static, public task dimensions only; no sampled poses, IDs or lattice."""
    cfg=env.config.cookie_transfer
    return dict(schema="a3_rgbd_geometry_v2",cookie_half_size=list(env.COOKIE_HALF_SIZE),
        source_inner_half_size=np.asarray(env.SOURCE_INNER_HALF_SIZE).tolist(),
        source_wall_center_half_size=list(cfg.source_bin_half_size_m),
        target_inner_half_size=list(env.TARGET_INNER_HALF_SIZE),wall_thickness_m=cfg.bin_wall_thickness_m,
        source_wall_top_offset_m=cfg.source_wall_base_z_m+cfg.source_bin_wall_height_m,
        source_floor_top_offset_m=cfg.source_floor_z_m+cfg.bin_wall_thickness_m/2,
        floor_offset_m=cfg.target_floor_z_m+cfg.bin_wall_thickness_m/2,
        wall_height_m=env.TARGET_WALL_HEIGHT,required_cookies=env.task_config.required_cookies,
        target_slots_local=np.asarray(env.TARGET_SLOTS_LOCAL).tolist())


def unproject(frame):
    h,w = frame.depth_m.shape
    v,u = np.mgrid[:h,:w]
    z = frame.depth_m.astype(np.float64)
    pixels=np.stack((u+.5,v+.5,np.ones_like(u)),axis=-1)
    optical=(pixels@np.linalg.inv(frame.intrinsic).T)*z[:,:,None]
    return optical@frame.world_from_optical[:3,:3].T+frame.world_from_optical[:3,3]


class MujocoRGBDSensor:
    """An extra calibrated overhead sensor; original policy cameras stay in place.

    Only rendering, camera calibration and simulation time cross this adapter.
    No segmentation pass or object-body lookup is exposed to the localizer.
    """
    def __init__(self, env, *, resolution=768, lookat=(.16,.18,.76), distance=.68,
                 elevation=-90., azimuth=90., depth_noise_m=0., seed=0):
        import mujoco
        self.env,self.depth_noise_m = env,float(depth_noise_m)
        self.rng = np.random.default_rng(seed)
        env.model.vis.global_.offwidth=max(env.model.vis.global_.offwidth,resolution)
        env.model.vis.global_.offheight=max(env.model.vis.global_.offheight,resolution)
        self.renderer=mujoco.Renderer(env.model,height=resolution,width=resolution)
        self.camera=mujoco.MjvCamera()
        self.camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:]=lookat
        self.camera.distance=distance
        self.camera.elevation=elevation
        self.camera.azimuth=azimuth
        self.protocol=dict(resolution=resolution,lookat=list(lookat),distance=distance,
            elevation=elevation,azimuth=azimuth,depth_noise_m=depth_noise_m,
            sensor="additional_overhead_rgbd",object_truth_access=False,segmentation_ids_used=False)

    def capture(self):
        import mujoco
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.env.data,camera=self.camera)
        rgb=self.renderer.render().copy()
        cam=mujoco.mjv_averageCamera(*self.renderer.scene.camera)
        forward=np.asarray(cam.forward,dtype=float).copy()
        up=np.asarray(cam.up,dtype=float).copy()
        forward/=np.linalg.norm(forward)
        right=np.cross(forward,up);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        transform=np.eye(4)
        transform[:3,:3]=np.column_stack((right,down,forward))
        transform[:3,3]=cam.pos
        h,w=rgb.shape[:2]
        focal=h*cam.frustum_near/(cam.frustum_top-cam.frustum_bottom)
        intrinsic=np.array([[focal,0,w/2],[0,focal,h*cam.frustum_top/(cam.frustum_top-cam.frustum_bottom)],[0,0,1]])
        flags=self.renderer.scene.flags.copy()
        try:
            self.renderer.enable_depth_rendering()
            depth=self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
            self.renderer.scene.flags[:]=flags
        if self.depth_noise_m:
            depth+=self.rng.normal(0,self.depth_noise_m,depth.shape).astype(np.float32)
        depth[(depth<.05)|(depth>2.)|~np.isfinite(depth)]=np.nan
        return RGBDFrame(rgb,depth,intrinsic,transform,float(self.env.data.time),
                         "simulated_calibrated_rgbd_camera")

    def close(self):
        self.renderer.close()


class MujocoRGBDViews:
    """Calibrated virtual sensors at three explicit, fixed viewpoints.

    They share a renderer for memory efficiency; no physics step occurs between
    captures. A real setup would need calibrated synchronous physical cameras.
    """
    def __init__(self,env,*,views=((-65.,180.),(-90.,90.),(-65.,90.)),**kwargs):
        self.sensor=MujocoRGBDSensor(env,**kwargs)
        self.views=tuple(views)
        self.protocol=dict(self.sensor.protocol,sensor="three_fixed_rgbd_views",
                           views=[dict(elevation=e,azimuth=a) for e,a in self.views])

    def capture(self):
        frames=[]
        for elevation,azimuth in self.views:
            self.sensor.camera.elevation=elevation;self.sensor.camera.azimuth=azimuth
            frames.append(self.sensor.capture())
        return RGBDViews(tuple(frames))

    def close(self):self.sensor.close()
