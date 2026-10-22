"""Optional slow right-arm camera scan; robot FK only, no scene-state queries.

The nominal tabletop workspace is a declared calibration/search volume, not a
sampled box pose. Every candidate is screened for IK and inter-arm clearance.
"""
import numpy as np
from types import SimpleNamespace


def joint_step(current,target,max_step=.018):
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    if current.shape!=(7,) or target.shape!=(7,) or not np.isfinite(np.r_[current,target]).all():
        raise ValueError('Right observation requires seven finite joint encoders')
    delta=target-current
    return current+delta*min(1.,max_step/max(float(np.max(abs(delta))),1e-9))


def scan_encoder_state(left_state,right_state):
    left=np.asarray(left_state,dtype=float);right=np.asarray(right_state,dtype=float)
    if left.shape!=(8,) or right.shape!=(8,) or not np.isfinite(np.r_[left,right]).all():
        raise ValueError('Right observation requires separate measured left and right 8D states')
    return np.r_[left,right]


def scan_due(completed_batches,phase,scanned_batches,*,each_batch=False):
    if completed_batches in scanned_batches:return False
    return phase=='INITIALIZING' or (each_batch and phase=='LOCATE' and completed_batches>0)


class RightObservationScan:
    def __init__(self,calibration,config,state,spec,*,emit=None,column=1):
        import mujoco
        from a3_dual_arm_sim.sim.actions import CartesianDeltaAdapter
        self.model=calibration.model;self.data=mujoco.MjData(self.model)
        self.calibration=calibration
        scales=SimpleNamespace(cartesian_translation_scale_m=config.cartesian_translation_scale_m,
            cartesian_rotation_scale_rad=config.cartesian_rotation_scale_rad)
        self.ik=CartesianDeltaAdapter(self.model,scales)
        self.state=np.asarray(state).copy();self.spec=spec;self.column=column
        self.emit=emit or (lambda **x:None)
        self.command=self.state[8:15].copy();self.initial=self.command.copy()
        self.left=self.state[:8].copy();self.grip=float(self.state[15])
        self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'R_eef')
        self.left_geoms,self.right_geoms=self._arm_geoms()
        self._fk(self.state)
        self.rotation=self.data.site_xmat[self.site].reshape(3,3).copy()
        self.candidates=None;self.index=0;self.age=0;self.settled=0
        self.done=False;self.best_quality=-1;self.best=self.initial.copy();self.returning=False
        self.steps=0;self.history=[]
        self.rejections={'workspace':0,'inter_arm':0,'ik':0}

    def _arm_geoms(self):
        import mujoco
        roots=[int(self.model.jnt_bodyid[self.ik._joint_id(arm[0])]) for arm in self.ik._arms]
        groups=[[],[]]
        for geom,body in enumerate(self.model.geom_bodyid):
            chain=[];body=int(body)
            while body>0:chain.append(body);body=int(self.model.body_parentid[body])
            for i,root in enumerate(roots):
                if root in chain:groups[i].append(geom)
        return groups

    def _fk(self,state):
        import mujoco
        self.data.qpos[self.calibration.arm_addresses]=np.r_[state[:7],state[8:15]]
        for addresses,grip in zip(self.calibration.finger_addresses,(state[7],state[15])):
            self.data.qpos[addresses]=(1-float(grip))*self.calibration.gripper_range
        mujoco.mj_kinematics(self.model,self.data)

    def _clear_path(self,start,finish):
        import mujoco
        for alpha in np.linspace(0,1,18):
            state=self.state.copy();state[8:15]=start*(1-alpha)+finish*alpha;self._fk(state)
            for a in self.right_geoms:
                p=self.data.geom_xpos[a];radius=self.model.geom_rbound[a]
                # Conservative keep-out around the tabletop task volume.
                if (.015-radius<p[0]<.30+radius and -.015-radius<p[1]<.46+radius
                        and p[2]-radius<.865):
                    self.rejections['workspace']+=1;return False
                for b in self.left_geoms:
                    distance=np.linalg.norm(p-self.data.geom_xpos[b])
                    if distance>radius+self.model.geom_rbound[b]+.035:continue
                    if mujoco.mj_geomDistance(self.model,self.data,a,b,.10,None)<.035:
                        self.rejections['inter_arm']+=1;return False
        return True

    def _plan(self,state):
        import mujoco
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,self.rotation.ravel())
        choices=[]
        # Pose the CAMERA rather than preserving the old EEF orientation.
        # A near-vertical view can see the narrow approach strip behind a rim.
        self._fk(state)
        eef=self.data.site_xpos[self.site].copy()
        er=self.data.site_xmat[self.site].reshape(3,3).copy()
        cameras,_,_=self.calibration.update(state);cp,cr,_=cameras['right_wrist']
        offset=er.T@(cp-eef);relative=er.T@cr
        for center in ((.10,.21,1.10),(.18,.20,1.10),(.26,.16,1.12)):
            for yaw in (0.,np.pi/2,-np.pi/2,np.pi):
                c,s=np.cos(yaw),np.sin(yaw)
                cam_r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                rotation=cam_r@relative.T;position=np.asarray(center)-rotation@offset
                orientation=np.empty(4);mujoco.mju_mat2Quat(orientation,rotation.ravel())
                q=self.ik.solve(position,orientation,state[8:15],arm_index=1)
                test=state.copy();test[8:15]=q;self._fk(test)
                if (np.linalg.norm(self.data.site_xpos[self.site]-position)<.0025
                        and np.linalg.norm(self.data.site_xmat[self.site].reshape(3,3)-rotation)<.065):
                    choices.append(q)
                else:self.rejections['ik']+=1
        # Observation EEF stays to the right of the left-arm work area and high
        # above the bins. These are search waypoints, never grasp coordinates.
        for x,y,z in ((.36,.16,1.04),(.34,.28,1.04),(.40,.20,1.10),(.30,.12,1.04),(.38,.08,.98)):
            position=np.array([x,y,z]);q=self.ik.solve(position,quat,state[8:15],arm_index=1)
            test=state.copy();test[8:15]=q;self._fk(test)
            error=np.linalg.norm(self.data.site_xpos[self.site]-position)
            rotation_error=np.linalg.norm(self.data.site_xmat[self.site].reshape(3,3)-self.rotation)
            if error>.0025 or rotation_error>.065:
                self.rejections['ik']+=1;continue
            choices.append(q)
        # Nearby joint-space candidates do not impose an unreachable wrist
        # orientation. Both their encoder FK and complete paths are screened.
        limits=self.ik._ranges[1]
        for j,amount in ((0,.30),(0,-.30),(1,.25),(1,-.25),(2,.30),(2,-.30),(3,.25),(3,-.25)):
            q=state[8:15].copy();q[j]+=amount
            if np.all(q>limits[:,0]+.01) and np.all(q<limits[:,1]-.01):choices.append(q)
        safe=[]
        for q in choices:
            if not self._clear_path(state[8:15],q):continue
            cameras,_,_=self.calibration.update(np.r_[self.left,q,self.grip])
            cp,cr,fovy=cameras['right_wrist']
            xx,yy=np.meshgrid(np.linspace(.04,.30,5),np.linspace(.03,.45,7))
            world=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,.76)]
            optical=(world-cp)@(cr@np.diag([1.,-1.,-1.]))
            radius=np.tan(np.deg2rad(fovy)/2)*optical[:,2]
            visible=(optical[:,2]>.05)&(np.max(abs(optical[:,:2]),axis=1)<radius)
            score=float(visible.mean())+.2*float(np.mean(np.maximum(0.,(cp[2]-.76)/np.linalg.norm(world-cp,axis=1))))
            safe.append((score,q,cp))
        safe.sort(key=lambda value:-value[0]);self.candidates=[];positions=[]
        for _,q,cp in safe:
            if any(np.linalg.norm(cp-p)<.025 for p in positions):continue
            if not all(self._clear_path(previous,q) for previous in self.candidates):continue
            self.candidates.append(q);positions.append(cp)
            if len(self.candidates)==3:break
        self.emit(event='right_scan_plan',candidates=len(self.candidates),
            camera_mode='unchanged_main_nominal_FK',object_truth=False,rejections=self.rejections)

    def quality(self,packet):
        from .three_view_localizer import ThreeViewLocalizer
        from a3_batch_skills.rgbd_localization import LocalizationFailure
        from a3_batch_skills.visual_goals import make_visual_goal
        from a3_batch_skills.planning import target_regions
        try:
            scene=ThreeViewLocalizer(self.spec).observe(packet.rgbd)
            quality=len(scene.source_cookies)+len(scene.target_cookies)
            for index in range(len(target_regions(self.spec['target_slots_local'],self.spec['floor_offset_m']+self.spec['cookie_half_size'][2]))):
                try:
                    make_visual_goal(packet.rgbd,scene,self.spec,index,column=self.column)
                    return quality+1000,'visible_five_and_empty_target'
                except LocalizationFailure:pass
            return quality,'anchor_only_no_certified_goal'
        except LocalizationFailure as exc:return 0,str(exc)

    def act(self,packet,right_state):
        state=scan_encoder_state(packet.observation['observation.state'],right_state)
        self.state=state.copy()
        self.steps+=1;self.age+=1
        if self.candidates is None:
            score,reason=self.quality(packet);self.best_quality=score
            self.history.append(dict(step=self.steps,quality=score,reason=reason))
            if score>=1000:self.done=True;return np.r_[self.left,self.command,self.grip]
            self._plan(state)
        if self.steps>=240 and not self.returning:
            self.returning=True;self.age=0
        if self.returning:target=self.best
        elif self.index>=len(self.candidates):
            self.returning=True;self.age=0;target=self.best
        else:target=self.candidates[self.index]
        if not hasattr(self,'checked_target') or not np.array_equal(target,self.checked_target):
            if not self._clear_path(state[8:15],target):
                self.command=state[8:15].copy();self.done=True
                self.emit(event='right_scan_path_blocked_hold_measured')
                return np.r_[self.left,self.command,self.grip]
            self.checked_target=target.copy()
        self.command=joint_step(self.command,target)
        self.command=np.clip(self.command,state[8:15]-.06,state[8:15]+.06)
        settled=np.max(abs(state[8:15]-target))<.025
        self.settled=self.settled+1 if settled else 0
        if self.settled>=8 or self.age>110:
            if self.returning:
                # Never declare a non-arrived moving pose a safe hold target.
                if self.settled>=8:self.done=True
            elif self.settled>=8:
                score,reason=self.quality(packet)
                pose=packet.calibrations['right_wrist'][0].tolist()
                self.history.append(dict(step=self.steps,quality=score,reason=reason,
                    right_encoder=state[8:15].tolist(),camera_position_nominal_fk=pose))
                self.emit(event='right_scan_view',index=self.index,quality=score,reason=reason,
                    camera_position_nominal_fk=pose)
                if score>self.best_quality:self.best_quality=score;self.best=state[8:15].copy()
                if score>=1000 and settled:self.done=True;self.command=state[8:15].copy()
                else:self.index+=1;self.age=0;self.settled=0
            else:
                self.emit(event='right_scan_pose_timeout',index=self.index)
                self.index+=1;self.age=0;self.settled=0
        if self.steps>350:
            # Bounded scan: stop commanding motion, keep the measured pose.
            self.command=state[8:15].copy();self.done=True
            self.emit(event='right_scan_budget_exhausted')
        return np.r_[self.left,self.command,self.grip]
