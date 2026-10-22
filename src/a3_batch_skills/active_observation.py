"""Empty-hand observation moves only; no pick/place/adjustment controller.

Camera poses for optimization are nominal robot FK, including the frozen
legacy target-body semantics. Every visited joint state is reached by env.step.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from .rgbd import unproject
from .rgbd_localization import LocalizationFailure,select_visible_five


def held_surface_suspected(packet,pad,rim):
    frames=packet.frames if hasattr(packet,'frames') else (packet,)
    for f in frames:
        q=unproject(f);r,g,b=f.rgb.astype(float).transpose(2,0,1)
        orange=(r>70)&(r>g*1.13)&(g>b*1.25)&np.isfinite(q).all(2)
        near=(np.linalg.norm(q-pad,axis=2)<.07)&(q[:,:,2]>rim+.025)
        if np.sum(orange&near)>=20:return True
    return False


def observation_entry_ready(pad_z,rim_z,jaw_gap,required_width,held_suspected):
    # Use encoder-derived jaw spacing, not a second arbitrary normalized
    # opening threshold. Clearance includes the long finger geometry.
    return bool(pad_z>=rim_z+.065 and jaw_gap>required_width and not held_suspected)


class LeftWristObservation:
    def __init__(self,sensor,env,*,max_steps=200):
        import mujoco
        self.model=copy.copy(sensor.model);self.data=mujoco.MjData(self.model)
        self.addresses=sensor.addresses;self.fingers=sensor.fingers
        self.cid=sensor.ids['left_wrist'];self.site=env._eef_sites[0];self.right_site=env._eef_sites[1]
        self.pad_ids=list(env._left_finger_geoms);self.grip_range=env.GRIPPER_RANGE_M
        self.pad_half=env.config.cookie_transfer.left_finger_pad_half_thickness_m
        self.open_width=10*env.COOKIE_HALF_SIZE[1]+.003
        self.cookie_pitch=2*env.COOKIE_HALF_SIZE[0]
        joint_ids=[int(np.flatnonzero(self.model.jnt_qposadr==i)[0]) for i in self.addresses[:7]]
        self.lower=self.model.jnt_range[joint_ids,0]+.002;self.upper=self.model.jnt_range[joint_ids,1]-.002
        self.max_steps=max_steps;self.active=False;self.done=False;self.error=None;self.used_steps=0
        self.events=[];self._phase='IDLE';self._return=[]

    def fk(self,q):
        import mujoco
        self.data.qpos[self.addresses]=np.r_[q,self.state[8:15]]
        for ids,g in zip(self.fingers,(self.state[7],self.state[15])):self.data.qpos[ids]=(1-g)*self.grip_range
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_camlight(self.model,self.data)
        return (self.data.cam_xpos[self.cid].copy(),self.data.cam_xmat[self.cid].reshape(3,3).copy(),
                self.data.geom_xpos[self.pad_ids].mean(0).copy(),self.data.site_xpos[self.site].copy(),
                self.data.site_xpos[self.right_site].copy())

    def _safe(self,q):
        _,_,pad,tool,right=self.fk(q)
        return (pad[2]>self.rim+.065 and tool[2]>self.rim+.055 and np.linalg.norm(tool-right)>.12
            and -.15<tool[0]<.65 and -.25<tool[1]<.65)

    def begin(self,state,scene,packet,reason,attempt=0,remaining=200,*,verify=False):
        self.state=np.asarray(state,dtype=float).copy();self.home=self.state[:7].copy()
        self.rim=max(scene.source_box.rim_z,scene.target_box.rim_z)
        _,_,pad,_,_=self.fk(self.home)
        gap=np.linalg.norm(self.data.geom_xpos[self.pad_ids[1]]-self.data.geom_xpos[self.pad_ids[0]])-2*self.pad_half
        held=held_surface_suspected(packet,pad,self.rim)
        if not observation_entry_ready(pad[2],self.rim,gap,self.open_width,held):
            raise LocalizationFailure('Observation requires a clear, open, visually empty hand: '+
                f'pad_clearance={pad[2]-self.rim:.4f}, jaw_gap={gap:.4f}, held_surface={held}')
        self.limit=min(self.max_steps,int(remaining));self.used_steps=0;self.done=False;self.error=None
        if self.limit<20:raise LocalizationFailure('Observation budget exhausted')
        target_focus=verify or 'Target' in reason or 'target' in reason
        box=scene.target_box if target_focus else scene.source_box
        center=box.position.copy();center[2]=self.rim-.015
        if not target_focus:
            try:
                five=select_visible_five(scene)
                p=np.asarray([c.position for c in five]);local=(p-box.position)@box.rotation
                xy=np.array([local[:,0].mean(),(-box.inner_half_size[1]+local[:,1].min())/2])
                center=box.position+box.rotation@np.r_[xy,0.];center[2]=self.rim-.015
            except LocalizationFailure:
                # A released hand is often looking at the target, so a single
                # visible source cookie is not a five-object goal. Use public
                # bin dimensions only to propose an outer-end OBSERVATION ROI;
                # real RGB-D must still discover and certify all five later.
                columns=sorted({c.column for c in scene.source_cookies if c.column is not None})
                col=columns[min(attempt,len(columns)-1)] if columns else min(attempt+1,4)
                x=(col-2.5)*self.cookie_pitch
                y=-box.inner_half_size[1]+.025
                center=box.position+box.rotation@np.array([x,y,0.]);center[2]=self.rim-.015
        center[:2]+=box.rotation[:2,0]*(.015 if attempt else -.015)
        def residual(q):
            cp,cr,pad,tool,right=self.fk(q);optical=(center-cp)@(cr@np.diag([1.,-1.,-1.]))
            return np.r_[optical[:2]/.035,(optical[2]-.25)/.15,
                max(0,self.rim+.08-pad[2])/.01,max(0,.14-np.linalg.norm(tool-right))/.02,
                .10*(q-self.home)]
        candidates=[]
        for offset in (0.,.12,-.12):
            initial=np.clip(self.home+np.array([offset,0,0,-offset,0,0,0]),self.lower,self.upper)
            fit=least_squares(residual,initial,bounds=(self.lower,self.upper),max_nfev=45)
            q=fit.x;cp,cr,_,_,_=self.fk(q);optical=(center-cp)@(cr@np.diag([1.,-1.,-1.]))
            fov=np.tan(np.deg2rad(self.model.cam_fovy[self.cid])/2)
            path=[self.home+(q-self.home)*f for f in np.linspace(0,1,30)]
            if optical[2]>.1 and np.max(abs(optical[:2]))<optical[2]*fov*.8 and all(self._safe(v) for v in path):
                travel=int(np.ceil(np.max(abs(q-self.home))/.025))
                if 2*travel+12<=self.limit:candidates.append((np.linalg.norm(residual(q)),q))
        if not candidates:raise LocalizationFailure('No reachable safe left-wrist observation position')
        self.target=min(candidates,key=lambda v:v[0])[1];self.active=True;self._phase='OUTBOUND'
        self._stable=0;self._captured=0;self._last_capture=None;self._path=[self.home.copy()]
        self.events.append(dict(event='observation_started',focus='target' if target_focus else 'source',
            home=self.home.tolist(),goal=self.target.tolist(),attempt=attempt+1))

    def act(self,state,timestamp):
        measured=np.asarray(state)[:7];self.used_steps+=1
        if self._phase=='OUTBOUND':
            if not self._safe(measured):
                self.error='Observation path lost measured clearance';self._start_return()
            elif self.used_steps>=self.limit//2:
                self.error='Observation outward budget exhausted';self._start_return()
            else:
                self._path.append(measured.copy())
                self._stable=self._stable+1 if np.max(abs(measured-self.target))<=.005 else 0
                if self._stable>=3:self._phase='CAPTURE'
        if self._phase=='CAPTURE':
            if self._last_capture is None or timestamp>self._last_capture+1e-6:
                self._captured+=1;self._last_capture=timestamp
            if self._captured>=6:self._start_return()
        if self._phase=='RETURN':
            if self._return and np.max(abs(measured-self._return[0]))<=.005:self._return.pop(0)
            target=self._return[0] if self._return else self.home
            at_home=np.max(abs(measured-self.home))<=.005
            self._stable=self._stable+1 if at_home and not self._return else 0
            if self._stable>=3:
                self.done=True;self.active=False;self._phase='DONE'
                self.events.append(dict(event='observation_handoff_returned',steps=self.used_steps,
                    maximum_joint_error_rad=float(np.max(abs(measured-self.home))),error=self.error))
        else:target=self.target
        if self.used_steps>=self.limit and not self.done:
            self.error='Observation return budget exhausted';self.active=False;self.done=True
        command=measured+np.clip(target-measured,-.025,.025)
        if not self._safe(command):
            self.error='Observation command clearance rejected';self.active=False;self.done=True;command=measured
        return np.r_[command,self.state[7]]

    def _start_return(self):
        self._phase='RETURN';self._stable=0
        self._return=[p.copy() for p in reversed(self._path[::2])]
        if not self._return or np.max(abs(self._return[-1]-self.home))>.001:self._return.append(self.home.copy())
