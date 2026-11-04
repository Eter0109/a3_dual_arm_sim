"""Continuous RGB-D five-cookie loop. No environment/score/contact-graph handle.

Classic calibrated visual control, not a neural-policy weight improvement.
Robot model is used only for FK/IK. Observations enter via ExecutionPacket.
"""
import numpy as np
from types import SimpleNamespace
from .batch_vision import RGBDFrame as FitFrame, LocalizationError
from .held_batch_vision import CANONICAL, localize_held_batch, placement_goal
from .three_view_localizer import ThreeViewLocalizer
from .observation_clearance import observation_lift
from .task_completion import EncoderSettling,FinalVisualSettling,observed_roi,view_completion_ready,insertion_release_ready
from a3_batch_skills.rgbd_localization import LocalizationFailure,select_visible_five
from a3_batch_skills.visual_goals import make_visual_goal, visual_region_complete
from a3_batch_skills.planning import target_regions


class EncoderMotion:
    def __init__(self, model, config, initial_state):
        import mujoco
        from a3_dual_arm_sim.sim.actions import CartesianDeltaAdapter
        from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
        # Retain robot servo scales only, rather than the environment's scene
        # configuration (which also contains nominal object layout fields).
        self.model = model
        self.config = SimpleNamespace(cartesian_translation_scale_m=config.cartesian_translation_scale_m,
            cartesian_rotation_scale_rad=config.cartesian_rotation_scale_rad)
        self.data = mujoco.MjData(model)
        self.ik = CartesianDeltaAdapter(model, self.config)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'L_eef')
        self.addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model,
            mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in (*LEFT_JOINTS,*RIGHT_JOINTS)]
        self.right = np.asarray(initial_state[8:]).copy()
        self.last = np.asarray(initial_state[:8], dtype=float).copy()
        self.pad_offset = None
        self.precision=False
        self.reference_q=None

    def pose(self, state):
        import mujoco
        self.data.qpos[self.addresses] = np.r_[state[:7],self.right[:7]]
        mujoco.mj_kinematics(self.model,self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3,3).copy()

    def hold(self, opening=None):
        if opening is not None:self.last[7]=np.clip(opening,0,1)
        return self.last.copy()

    def pad_height_bounds(self,state):
        """Known pad geometry transformed by measured robot joints only."""
        import mujoco
        from itertools import product
        if not hasattr(self,'_release_geometry_data'):
            self._release_geometry_data=mujoco.MjData(self.model)
            self._release_pad_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,
                'L_finger_'+name+'_geom') for name in ('inner','outer')]
            self._release_finger_joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,
                'L_finger_'+name+'_joint') for name in ('inner','outer')]
            if min(self._release_pad_ids+self._release_finger_joints)<0:
                raise LocalizationError('Known finger geometry is unavailable')
        data=self._release_geometry_data
        state=np.asarray(state,dtype=float)
        if state.shape not in ((8,),(16,)) or not np.isfinite(state).all():
            raise LocalizationError('Finite measured robot encoders required for pad clearance')
        data.qpos[self.addresses]=np.r_[state[:7],self.right[:7]]
        for joint in self._release_finger_joints:
            lower,upper=self.model.jnt_range[joint]
            data.qpos[int(self.model.jnt_qposadr[joint])]=lower+(1-state[7])*(upper-lower)
        mujoco.mj_kinematics(self.model,data)
        corners=[]
        signs=np.asarray(list(product((-1.,1.),repeat=3)))
        for geom in self._release_pad_ids:
            if self.model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_BOX:
                raise LocalizationError('Pad clearance requires the declared box geometry')
            local=signs*self.model.geom_size[geom]
            corners.extend(local@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
        z=np.asarray(corners)[:,2]
        return float(z.min()),float(z.max())

    def resync(self,state,opening=None):
        self.last=np.asarray(state[:8],dtype=float).copy()
        return self.hold(opening)

    def errors(self,state,position,rotation):
        pos,rot=self.pose(state)
        angle=np.arccos(np.clip((np.trace(rot.T@rotation)-1)/2,-1.,1.))
        return float(np.linalg.norm(pos-position)),float(angle)

    def pose_error(self,state,position,rotation):
        import mujoco
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.asarray(rotation).ravel())
        q=self.ik.solve(position,quat,np.asarray(state[:7]))
        test=np.asarray(state).copy();test[:7]=q
        actual,actual_r=self.pose(test)
        error=(float(np.linalg.norm(actual-position)),float(np.linalg.norm(actual_r-rotation)))
        self.pose(state)
        return error

    def command(self, state, position, rotation, opening, speed=.004):
        import mujoco
        measured_pos,measured_rot=self.pose(state)
        commanded=np.r_[self.last,self.right]
        self.ik._work.qpos[:]=self.data.qpos
        self.ik._work.qpos[self.addresses[:7]]=self.last[:7]
        mujoco.mj_kinematics(self.model,self.ik._work)
        current=self.ik._work.site_xpos[self.site].copy()
        current_r=self.ik._work.site_xmat[self.site].reshape(3,3).copy()
        delta=np.asarray(position)-current
        distance=np.linalg.norm(delta)
        delta*=min(1.,min(speed,max(.0008,distance*.25))/max(distance,1e-9))
        target_q,current_q,measured_q=np.empty(4),np.empty(4),np.empty(4)
        mujoco.mju_mat2Quat(target_q,np.asarray(rotation).ravel())
        mujoco.mju_mat2Quat(current_q,current_r.ravel())
        mujoco.mju_mat2Quat(measured_q,measured_rot.ravel())
        error,measured_error=np.empty(3),np.empty(3)
        mujoco.mju_subQuat(error,target_q,current_q)
        mujoco.mju_subQuat(measured_error,target_q,measured_q)
        error*=min(1.,.04/max(np.linalg.norm(error),1e-9))
        action=np.zeros(14)
        action[:3]=delta/self.config.cartesian_translation_scale_m
        action[3:6]=error/self.config.cartesian_rotation_scale_rad
        action[6],action[13]=2*opening-1,2*self.right[7]-1
        if self.precision:
            from .precision_servo import precision_joint_target
            bounded=np.clip(action,-1,1)
            target_quat=current_q.copy()
            mujoco.mju_quatIntegrate(target_quat,bounded[3:6]*self.config.cartesian_rotation_scale_rad,1.)
            q=precision_joint_target(self.model,self.ik._work,self.addresses[:7],self.site,
                self.ik._ranges[0],self.last[:7],current+bounded[:3]*self.config.cartesian_translation_scale_m,target_quat,
                reference_q=self.reference_q)
            result=np.r_[q,opening]
        else:result=self.ik.convert(self.data,np.clip(action,-1,1),commanded)[:8]
        self.last=result.copy()
        reached=np.linalg.norm(measured_pos-position)<.0025 and np.linalg.norm(measured_error)<.045
        return result,bool(reached)

    def placement_ik(self,position,rotation,seed):
        """Robot-only continuation IK; never changes the physical command."""
        import mujoco
        from .precision_servo import precision_joint_target
        work=self.ik._work
        work.qpos[:]=self.data.qpos
        work.qpos[self.addresses[:7]]=seed
        quaternion=np.empty(4);mujoco.mju_mat2Quat(quaternion,np.asarray(rotation).ravel())
        q=precision_joint_target(self.model,work,self.addresses[:7],self.site,
            self.ik._ranges[0],seed,position,quaternion,max_joint_step=.20)
        work.qpos[self.addresses[:7]]=q;mujoco.mj_kinematics(self.model,work)
        return q,work.site_xpos[self.site].copy(),work.site_xmat[self.site].reshape(3,3).copy()


class SensorBatchLoop:
    def __init__(self,motion,spec,*,event_sink=None,max_batches=None,source_column='auto',refine=False,visual_memory=None,settle_observe=False,placement_planning=False,guarded_release=False):
        self.motion,self.spec=motion,dict(spec)
        self.localizer=ThreeViewLocalizer(spec)
        self.events=[];self.sink=event_sink
        self.phase='LOCATE';self.steps=0;self.age=0;self.stable=0
        self.completed=[];self.failure_reason=None;self.scene=None
        self.max_batches=max_batches or spec['required_cookies']//5
        self.view_attempt=0;self.goal=None;self.visual_stable=0
        self.last_visual_timestamp=None;self.last_positions=None
        self.opening=1.;self.grip=.4;self.pinch_stable=0
        self.visual_error=None;self.held_attempt=0
        self.source_column=source_column
        self.view_focus='source_entry';self.blocked_region=None
        self.refine=bool(refine);self.encoder_settling=EncoderSettling()
        self.motion.precision=self.refine
        self.insert_visual_frames=0;self.insert_visual_positions=None
        self.visual_memory=visual_memory
        self.memory_settling=EncoderSettling();self.memory_goal_accepts=0
        self.scene_state_memory=getattr(visual_memory,'is_scene_state_memory',False) is True
        self.settle_observe=bool(settle_observe or self.scene_state_memory);self.observation_hold=None
        self.observation_hold_completed=0;self.observation_hold_steps=0
        # These fields are inert in V15/V4 mode.  Preparation holds and physical
        # view requests are bounded independently of manipulation/verification.
        self.scene_observation_batch=None;self.scene_batch_hold_count=0
        self.scene_preparation_start=0;self.scene_view_attempts={}
        self.scene_observations_by_focus={'source_entry':0,'target_empty':0}
        self.scene_observation_requests=0;self.scene_max_observation_requests=6
        self.scene_max_preparation_steps=600
        self.placement_planning=bool(placement_planning)
        self.placement_plans=0;self.placement_symmetry_changes=0
        self.placement_tilt_changes=0
        self.placement_guidance_activations=0;self.placement_reference_by_phase={}
        self.placement_guided_phase=None
        self.final_settling=None
        self.guarded_release=bool(guarded_release)
        self.release_plan=None;self.release_gap_frames=0
        self.release_clearance_frames=0;self.guarded_releases=0
        self.held_release_shapes=None

    def release_sweep_lowest(self,state):
        # With fixed arm joints the pads move linearly as the jaw opens.
        # Checking both endpoints includes the vertical sweep of a tilted jaw.
        opened=np.asarray(state,dtype=float).copy();opened[7]=.49
        return min(self.motion.pad_height_bounds(state)[0],
                   self.motion.pad_height_bounds(opened)[0])

    def start_release(self,packet):
        if self.guarded_release:
            from .release_planning import plan_narrow_release,required_detach_height
            state=packet.observation['observation.state']
            self.release_plan=plan_narrow_release(float(state[7]),packet.jaw_gap_m)
            # These are latched measured geometry bounds, not live object poses.
            # Add the known box rim so full opening also occurs above the wall.
            half=np.asarray(self.spec['cookie_half_size'])
            tops=[float(self.scene.target_box.rim_z)]
            tops.extend(float(c.position[2]+np.abs(c.rotation[2])@half)
                for c in self.scene.target_cookies)
            if self.held_release_shapes is None:
                raise LocalizationError('Measured held geometry required before guarded release')
            pos,rot=self.motion.pose(state)
            tops.extend(float((pos+rot@offset)[2]+np.abs((rot@rotation)[2])@half)
                for offset,rotation in self.held_release_shapes)
            self.release_protected_top=max(tops)
            low=self.release_sweep_lowest(state)
            needed=required_detach_height(low,self.release_protected_top,max_detach_m=.127)
            self.release_lift_m=max(.030,needed+.003)
            self.release_gap_frames=0;self.release_clearance_frames=0
            self.guarded_releases+=1
            self.emit('guarded_release_planned',measured_opening=float(state[7]),
                measured_gap_m=packet.jaw_gap_m,target_opening=self.release_plan.opening,
                target_gap_m=self.release_plan.target_gap_m,clearance_increment_m=.002,
                protected_top_z_m=self.release_protected_top,initial_pad_lowest_z_m=low,
                detach_height_m=self.release_lift_m,object_truth_used=False)
        self.opening=self.grip
        self.transition('OPEN',self.place)

    @property
    def finished(self):return self.phase in ('DONE','FAILED')

    def emit(self,event,**values):
        row=dict(step=self.steps,phase=self.phase,event=event,**values)
        self.events.append(row)
        if self.sink:self.sink(row)

    def transition(self,phase,goal=None):
        if getattr(self,'placement_planning',False):
            self.motion.reference_q=None;self.placement_guided_phase=None
        if self.visual_memory is not None and phase in ('APPROACH','DONE','FAILED'):
            self.visual_memory.clear('manipulation_started' if phase=='APPROACH' else phase.lower())
        self.phase,self.age,self.stable=phase,0,0
        self.encoder_settling=EncoderSettling()
        if phase=='INSERT':self.insert_visual_frames=0;self.insert_visual_positions=None
        if phase=='FINAL_SETTLE':self.final_settling=FinalVisualSettling()
        if goal is not None:self.move_goal=goal
        self.emit('stage')

    def fail(self,reason):
        self.failure_reason=str(reason);self.transition('FAILED');self.emit('failure',reason=str(reason))

    def scene_from(self,packet):
        try:
            scene=self.localizer.observe(packet.rgbd)
            self.scene=scene;self.visual_error=None
            return scene
        except LocalizationFailure as exc:
            self.visual_error=str(exc);return None

    def _ensure_scene_observation_batch(self):
        batch=len(self.completed)
        if self.scene_observation_batch==batch:return
        self.scene_observation_batch=batch;self.scene_batch_hold_count=0
        self.scene_preparation_start=self.steps;self.scene_view_attempts={}
        self.scene_observation_requests=0
        self.visual_memory.begin_batch(batch)
        # A new batch cannot inherit empty-space evidence from before contact.
        self.visual_memory.clear('new_scene_observation_batch')
        self.emit('scene_observation_batch_started',batch=batch,
            max_observation_requests=self.scene_max_observation_requests,
            max_preparation_control_steps=self.scene_max_preparation_steps)

    def _scene_observation_key(self,focus=None,index=None):
        focus=self.view_focus if focus is None else focus
        index=self.blocked_region if index is None else index
        return (focus,index if focus=='target_empty' else None)

    def _scene_observation_roi(self,scene,*,focus=None,index=None):
        """The actual metric gate footprint, not five points near its centre."""
        from a3_batch_skills.visual_goals import rectangle_samples
        focus=self.view_focus if focus is None else focus
        half=np.asarray(self.spec['cookie_half_size'])
        if focus=='target_empty':
            regions=target_regions(self.spec['target_slots_local'],self.spec['floor_offset_m']+half[2])
            index=self.blocked_region if index is None else index
            if index is None:index=next(i for i in range(len(regions)) if i not in {g.index for g in self.completed})
            target=regions[index];box=scene.target_box;offset=self.spec['floor_offset_m']
            xy=rectangle_samples(target[:,:2].min(0)-half[:2]+.001,
                target[:,:2].max(0)+half[:2]-.001)
        else:
            box=scene.source_box;offset=self.spec['source_floor_top_offset_m']
            try:
                five=select_visible_five(scene,column=self.source_column)
                local=(np.asarray([cookie.position for cookie in five])-box.position)@box.rotation
                low=np.array([local[:,0].mean()-.018,-box.inner_half_size[1]+.001])
                high=np.array([local[:,0].mean()+.018,local[:,1].min()-half[1]-.0007])
                if high[1]<=low[1]:raise LocalizationFailure('No measurable source entry corridor')
                xy=rectangle_samples(low,high)
            except LocalizationFailure:
                # No guessed cookie positions: observe the public bin footprint
                # to recover a CURRENT visible five-object group.
                xy=rectangle_samples(-np.asarray(box.inner_half_size)+.006,
                    np.asarray(box.inner_half_size)-.006)
        return np.c_[xy,np.full(len(xy),offset)]@box.rotation.T+box.position

    def _request_scene_observation(self,packet,requirement):
        focus=getattr(requirement,'missing_kind',None)
        if focus not in ('source_entry','target_empty'):
            raise requirement
        index=getattr(requirement,'region_index',None)
        key=self._scene_observation_key(focus,index)
        attempt=self.scene_view_attempts.get(key,0)
        if attempt>=3 or self.scene_observation_requests>=self.scene_max_observation_requests:
            self.fail('Scene observation exhausted for '+focus+': '+str(requirement));return
        self.view_focus=focus;self.blocked_region=index if focus=='target_empty' else None
        self.view_attempt=attempt
        self.emit('scene_observation_required',reason=str(requirement),missing_kind=focus,
            target_region_index=index,focus_attempt=attempt,
            source_fraction=getattr(requirement,'source_fraction',None),
            target_fraction=getattr(requirement,'target_fraction',None))
        self.start_observation(packet)
        self.scene_view_attempts[key]=attempt+1
        self.scene_observation_requests+=1
        self.scene_observations_by_focus[focus]+=1

    def _scene_requirement(self,kind,index,reason):
        # Keep the loop independent of the memory implementation's constructor.
        exc=LocalizationFailure(reason);exc.missing_kind=kind;exc.region_index=index
        return exc

    def observe_goal(self,packet,*,verify=False):
        """Plan a physical wrist pose, never render a teleported camera."""
        import mujoco
        state=packet.observation['observation.state']
        pos,rot=self.motion.pose(state)
        cp,cr,fovy=packet.calibrations['left_wrist']
        offset=rot.T@(cp-pos);relative=rot.T@cr
        source,target=self.scene.source_box,self.scene.target_box
        if not verify and self.view_focus=='target_empty':
            regions=target_regions(self.spec['target_slots_local'],
                self.spec['floor_offset_m']+self.spec['cookie_half_size'][2])
            index=self.blocked_region
            if index is None:index=next(i for i in range(len(regions)) if i not in {g.index for g in self.completed})
            center=(target.position+target.rotation@regions[index].mean(0))[:2]
            centers=[center,center+target.rotation[:2,0]*.025,center-target.rotation[:2,0]*.025]
            self.emit('active_view_target',region='target_empty',target_region_index=index,position_xy=center.tolist())
        elif verify:
            centers=[target.position[:2]+target.rotation[:2,0]*dx for dx in (0.,.025,-.025)]
        else:
            try:
                five=select_visible_five(self.scene,column=self.source_column)
                local=(np.asarray([cookie.position for cookie in five])-source.position)@source.rotation
                front=-source.inner_half_size[1]+.001
                end=local[:,1].min()-self.spec['cookie_half_size'][1]-.0007
                entry=np.array([local[:,0].mean(),(front+end)/2])
                center=source.position[:2]+source.rotation[:2,:2]@entry
                centers=[center,center-source.rotation[:2,1]*.010,
                         center+source.rotation[:2,0]*.010]
                self.emit('active_view_target',region='source_entry',position_xy=center.tolist())
            except LocalizationFailure:
                centers=[(source.position[:2]+target.position[:2])/2,
                         source.position[:2]+source.rotation[:2,:2]@np.array([-.060,-.055]),
                         source.position[:2]+source.rotation[:2,1]*-.040]
        attempt=(self.scene_view_attempts.get(self._scene_observation_key(),0)
            if self.scene_state_memory and not verify else self.view_attempt)
        center=centers[min(attempt,len(centers)-1)]
        # Small measured region of interest. Positive occupancy/completion is
        # still required by LOCATE/VERIFY after the camera has settled.
        if verify:
            half=np.asarray(target.inner_half_size)
            roi=np.array([[x,y,self.spec['floor_offset_m']+.02]
                for x in (-half[0]+.006,0.,half[0]-.006)
                for y in (-half[1]+.006,0.,half[1]-.006)])
            self.view_roi=roi@target.rotation.T+target.position
        elif self.scene_state_memory:
            self.view_roi=self._scene_observation_roi(self.scene)
        else:
            self.view_roi=np.array([[center[0]+dx,center[1]+dy,source.rim_z-.015]
                for dx,dy in ((0,0),(-.020,-.004),(-.020,.004),(.020,-.004),(.020,.004))])
        camera_pos=np.r_[center,(max(source.rim_z,target.rim_z) if self.scene_state_memory else source.rim_z)+.30]
        candidates=[]
        for yaw in (0.,np.pi/2,-np.pi/2,np.pi):
            c,s=np.cos(yaw),np.sin(yaw)
            cam_r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            r=cam_r@relative.T;p=camera_pos-r@offset
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,r.ravel())
            q=self.motion.ik.solve(p,quat,state[:7])
            test=state.copy();test[:7]=q
            actual,actual_r=self.motion.pose(test)
            position_error=np.linalg.norm(actual-p);rotation_error=np.linalg.norm(actual_r-r)
            error=position_error+.1*rotation_error
            # Observation waypoints need a valid physical view, not an exact
            # grasp pose. If joint limits prevent the requested pose, retain the
            # actually reachable FK pose only when it still sees the desired ROI
            # from safely above both bins. Manipulation tolerances stay intact.
            actual_cp=actual+actual_r@offset;actual_cr=actual_r@relative
            point=np.r_[center,source.rim_z-.015]
            optical=(point-actual_cp)@(actual_cr@np.diag([1.,-1.,-1.]))
            limit=np.tan(np.deg2rad(fovy)/2)*optical[2]*.65
            pad_offset=self.motion.pad_offset
            if pad_offset is None:pad_offset=rot.T@(packet.pad_midpoint_world-pos)
            actual_pad=actual+actual_r@pad_offset
            visible=(optical[2]>.15 and np.max(abs(optical[:2]))<limit and
                actual_cp[2]>max(source.rim_z,target.rim_z)+.18 and
                actual_pad[2]>max(source.rim_z,target.rim_z)+.065)
            if self.scene_state_memory and not verify:
                # Covers the whole requested physical footprint.  This is a
                # view feasibility test, not a claim that occluded floor is free.
                roi_optical=(self.view_roi-actual_cp)@(actual_cr@np.diag([1.,-1.,-1.]))
                visible=bool(visible and np.all(roi_optical[:,2]>.05) and
                    np.all(np.max(abs(roi_optical[:,:2]),axis=1)<
                        np.tan(np.deg2rad(fovy)/2)*roi_optical[:,2]*.90))
            if not visible:continue
            adjusted=bool(position_error>.0025 or rotation_error>.06)
            candidates.append((error+.0002*np.linalg.norm(q-state[:7]),
                actual.copy() if adjusted else p,actual_r.copy() if adjusted else r,error,adjusted))
        if not candidates:raise LocalizationFailure('No safe calibrated observation view covers requested region')
        best=min(candidates,key=lambda v:v[0])
        self.emit('physical_observation_goal',position=best[1].tolist(),ik_error=best[3],
            observation_waypoint_adjusted_to_reachable_FK=best[4])
        return best[1],best[2]

    def start_observation(self,packet,*,verify=False):
        self.view_goal=self.observe_goal(packet,verify=verify)
        state=packet.observation['observation.state']
        pos,rot=self.motion.pose(state)
        self.after_view='VERIFY' if verify else 'LOCATE'
        rim=max(self.scene.source_box.rim_z,self.scene.target_box.rim_z)
        lift,delta=observation_lift(pos,packet.pad_midpoint_world,rim)
        self.emit('observation_clearance',pad_z=float(packet.pad_midpoint_world[2]),
            rim_z=float(rim),additional_lift_m=delta)
        if delta<.001:
            # RETREAT already moved the released gripper above both rims.
            # Do not demand another high vertical pose with the old wrist angle.
            self.transition('VIEW',self.view_goal)
        else:
            pe,re=self.motion.pose_error(state,lift,rot)
            if pe>.0025 or re>.065:
                raise LocalizationFailure('No reachable safe observation lift')
            self.transition('VIEW_LIFT',(lift,rot))

    def _remember_view(self,packet,scene=None):
        if getattr(self,'scene_state_memory',False):return
        if self.visual_memory is None or self.memory_settling.count<3:return
        if self.phase!='LOCATE' and not (self.phase=='VIEW' and self.after_view=='LOCATE'):return
        if scene is None:
            # Collect extra geometry without changing V15's planning scene or
            # its error flags merely because memory is enabled.
            try:scene=self.localizer.observe(packet.rgbd)
            except LocalizationFailure:return
        if scene is not None:
            self.visual_memory.observe(packet,scene,batch=len(self.completed),phase=self.phase,stationary=True)

    def _remember_verified_boundary(self,packet,scene):
        # Called only after live release and previous-progress verification.
        # These are observations of the NEXT batch's pre-manipulation state.
        if self.visual_memory is None or getattr(self,'scene_state_memory',False):return
        ready=(self.phase=='VERIFY' and self.visual_stable>=3 and len(self.completed)>0
            and self.memory_settling.count>=3 and packet.jaw_gap_m is not None
            and packet.jaw_gap_m>.0347)
        if not ready:
            self.emit('memory_boundary_capture_rejected',stationary_frames=self.memory_settling.count,
                visual_frames=self.visual_stable)
            return
        self.visual_memory.observe(packet,scene,batch=len(self.completed),phase='BATCH_READY',stationary=True)

    def _make_goal(self,packet,scene,index,column):
        if getattr(self,'scene_state_memory',False):
            return self.visual_memory.make_goal(packet,scene,self.spec,index,
                column=column,batch=len(self.completed))
        # Preserve the complete V15 path whenever current evidence suffices.
        try:return make_visual_goal(packet.rgbd,scene,self.spec,index,column=column)
        except LocalizationFailure:
            if self.visual_memory is None:raise
            goal=self.visual_memory.make_goal(packet,scene,self.spec,index,
                column=column,batch=len(self.completed))
            self.memory_goal_accepts+=1
            self.emit('visual_memory_goal_accepted',target_region_index=index,source_column=column)
            return goal

    def _locate_after_observation(self,reason):
        if not self.settle_observe:
            self.transition('LOCATE');return
        if self.scene_state_memory:self._ensure_scene_observation_batch()
        from .observation_hold import ObservationHold
        self.observation_hold=ObservationHold(max_steps=40,required_stable=3)
        self.transition('OBSERVE_HOLD')
        self.emit('observation_hold_started',reason=reason,max_control_steps=40,
            required_independent_stationary_updates=3)

    def _act_observation_hold(self,packet,*,fresh):
        hold=self.observation_hold
        row=hold.update(packet.observation['observation.state'],packet.timestamp,
            packet.jaw_gap_m,fresh=fresh,opening=float(self.motion.last[7]))
        self.observation_hold_steps+=1
        if hold.target is not None:
            self.motion.last=hold.target.copy()
        action=self.motion.hold()
        if row['latched']:
            self.emit('observation_hold_latched',target=hold.target.tolist(),**row)
        elif row['independent'] or row['timed_out']:
            self.emit('observation_hold_status',**row)
        if row['timed_out']:
            self.fail('OBSERVE_HOLD timed out: '+row['reason'])
        elif row['ready']:
            # Fit current sensor evidence equally in both arms. This is a
            # read-only capture fit: do not replace self.scene/visual_error or
            # use its result as the next planning goal. LOCATE still fits live.
            try:scene=self.localizer.observe(packet.rgbd)
            except LocalizationFailure as exc:
                scene=None
                self.emit('observation_hold_localization_unavailable',reason=str(exc))
            if scene is not None and self.visual_memory is not None:
                self.visual_memory.observe(packet,scene,batch=len(self.completed),
                    phase='OBSERVE_HOLD',stationary=True)
            if self.scene_state_memory:
                self.scene_batch_hold_count+=1
                self.emit('scene_observation_recorded',batch=len(self.completed),
                    localized=scene is not None,focus=self.view_focus,
                    batch_hold_count=self.scene_batch_hold_count,
                    state_memory=self.visual_memory.snapshot_summary())
            self.observation_hold_completed+=1
            self.emit('observation_hold_completed',localized=scene is not None,
                memory_enabled=self.visual_memory is not None,**row)
            self.transition('LOCATE')
        return action

    def choose_pick(self,packet,scene):
        if getattr(self,'scene_state_memory',False):return self._choose_scene_state_pick(packet,scene)
        # Fill regions farthest from the fixed front camera first. Filling the
        # near half first can occlude the remaining far floor in the 2x10 box.
        # Every candidate still needs CURRENT positive depth evidence of free
        # floor; no cached empty mask and no truth-based occupancy fallback.
        regions=target_regions(self.spec['target_slots_local'],
            self.spec['floor_offset_m']+self.spec['cookie_half_size'][2])
        camera_position=packet.calibrations['front'][0]
        remaining=[i for i in range(len(regions)) if i not in {g.index for g in self.completed}]
        remaining.sort(key=lambda i:-np.linalg.norm(
            scene.target_box.position+scene.target_box.rotation@regions[i].mean(0)-camera_position))
        errors=[];source_blocked=False
        self.view_focus='source_entry';self.blocked_region=None
        self.goal=None
        for index in remaining:
            columns=sorted({c.column for c in scene.source_cookies})
            if self.source_column!='auto':columns=[self.source_column]
            for column in columns:
                try:
                    candidate=self._make_goal(packet,scene,index,column)
                    if not self.reachable_pick(candidate,packet.observation['observation.state']):
                        errors.append('Visible source group has no reachable canonical grasp')
                        continue
                    self.goal=candidate;break
                except LocalizationFailure as exc:
                    errors.append(str(exc))
                    if 'Target region' in str(exc):
                        self.view_focus='target_empty';self.blocked_region=index
                    elif 'five-cookie' in str(exc):source_blocked=True
            if self.goal is not None:break
        if self.goal is None:
            if source_blocked:self.view_focus='source_entry';self.blocked_region=None
            try:
                from a3_batch_skills.visual_goals import floor_visibility,rectangle_samples
                five=select_visible_five(scene,column=self.source_column)
                local=(np.asarray([c.position for c in five])-scene.source_box.position)@scene.source_box.rotation
                low=np.array([local[:,0].mean()-.018,-scene.source_box.inner_half_size[1]+.001])
                high=np.array([local[:,0].mean()+.018,local[:,1].min()-self.spec['cookie_half_size'][1]-.0007])
                xy=rectangle_samples(low,high)
                self.emit('entry_visibility',fractions=[floor_visibility(f,scene.source_box,xy,self.spec['source_floor_top_offset_m']) for f in packet.rgbd.frames],
                    visible_source_cookies=len(scene.source_cookies))
            except LocalizationFailure:pass
            raise LocalizationFailure('; '.join(dict.fromkeys(errors)))
        rotation=self.goal.source_rotation@CANONICAL
        center=(self.goal.source_positions[0]+self.goal.source_positions[-1])/2
        self.pick=(center+self.goal.source_rotation[:,2]*.010-rotation@self.motion.pad_offset,rotation)
        half=np.asarray(self.spec['cookie_half_size'])
        axis=rotation[:,0]
        positions=self.goal.source_positions@axis
        width=np.ptp(positions)+2*np.abs(self.goal.source_rotation.T@axis)@half
        self.insert=float(np.clip((width-.0038)/.085,0,1))
        self.opening=1.
        self.emit('visual_pick_goal',source_column=self.goal.source_column,
            target_region_index=self.goal.index,
            positions=self.goal.source_positions.tolist(),insert_opening=self.insert,
            approach_clear=self.goal.source_clear_fraction,target_empty=self.goal.target_empty_fraction)
        self.transition('APPROACH',(self.pick[0]+[0,0,.082],rotation))

    def _choose_scene_state_pick(self,packet,scene):
        """Observe/record first, then combine independently certified regions."""
        if self.scene_batch_hold_count<1:
            raise self._scene_requirement('source_entry',None,'No stationary scene observation has been recorded')
        regions=target_regions(self.spec['target_slots_local'],
            self.spec['floor_offset_m']+self.spec['cookie_half_size'][2])
        camera_position=packet.calibrations['front'][0]
        remaining=[i for i in range(len(regions)) if i not in {g.index for g in self.completed}]
        remaining.sort(key=lambda i:-np.linalg.norm(
            scene.target_box.position+scene.target_box.rotation@regions[i].mean(0)-camera_position))
        summary=self.visual_memory.snapshot_summary()
        # Record existence never substitutes for a positive floor gate: the
        # memory's make_goal still independently validates both query regions.
        if not self.visual_memory.has_record('source',len(self.completed)):
            raise self._scene_requirement('source_entry',remaining[0], 'Source state has not been recorded from a stationary calibrated observation')
        if not self.visual_memory.has_record('target',len(self.completed)):
            raise self._scene_requirement('target_empty',remaining[0], 'Target state has not been recorded from a stationary calibrated observation')
        requirements=[];self.goal=None
        for index in remaining:
            columns=sorted({cookie.column for cookie in scene.source_cookies})
            if self.source_column!='auto':columns=[self.source_column]
            if not columns:
                requirements.append(self._scene_requirement('source_entry',index,'No currently visible source group'));continue
            for column in columns:
                try:
                    candidate=self._make_goal(packet,scene,index,column)
                    if not self.reachable_pick(candidate,packet.observation['observation.state']):
                        requirements.append(self._scene_requirement('source_entry',index,'Current source group has no reachable canonical grasp'));continue
                    self.goal=candidate;break
                except LocalizationFailure as exc:
                    if getattr(exc,'missing_kind',None) is None:
                        exc=self._scene_requirement('source_entry',index,str(exc))
                    requirements.append(exc)
            if self.goal is not None:break
        if self.goal is None:
            if requirements:raise requirements[0]
            raise LocalizationFailure('No remaining scene-state placement region')
        # Retain the original V15 grasp, held correction, insertion and release.
        rotation=self.goal.source_rotation@CANONICAL
        center=(self.goal.source_positions[0]+self.goal.source_positions[-1])/2
        self.pick=(center+self.goal.source_rotation[:,2]*.010-rotation@self.motion.pad_offset,rotation)
        half=np.asarray(self.spec['cookie_half_size']);axis=rotation[:,0]
        positions=self.goal.source_positions@axis
        width=np.ptp(positions)+2*np.abs(self.goal.source_rotation.T@axis)@half
        self.insert=float(np.clip((width-.0038)/.085,0,1));self.opening=1.
        self.emit('scene_state_goal_accepted',batch=len(self.completed),
            source_column=self.goal.source_column,target_region_index=self.goal.index,
            observations_by_focus=dict(self.scene_observations_by_focus),
            preparation_control_steps=self.steps-self.scene_preparation_start,
            current_source_five_required=True,state_memory=summary)
        self.emit('visual_pick_goal',source_column=self.goal.source_column,
            target_region_index=self.goal.index,positions=self.goal.source_positions.tolist(),
            insert_opening=self.insert,approach_clear=self.goal.source_clear_fraction,
            target_empty=self.goal.target_empty_fraction)
        self.transition('APPROACH',(self.pick[0]+[0,0,.082],rotation))

    def reachable_pick(self,goal,state):
        import mujoco
        rotation=goal.source_rotation@CANONICAL
        center=(goal.source_positions[0]+goal.source_positions[-1])/2
        pick=center+goal.source_rotation[:,2]*.010-rotation@self.motion.pad_offset
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,rotation.ravel())
        seed=state[:7].copy()
        for p in (pick+[0,0,.082],pick):
            q=self.motion.ik.solve(p,quat,seed)
            test=state.copy();test[:7]=q
            actual,actual_r=self.motion.pose(test)
            if np.linalg.norm(actual-p)>.0025 or np.linalg.norm(actual_r-rotation)>.065:
                self.emit('unreachable_visual_group',source_column=goal.source_column,
                    position_error_m=float(np.linalg.norm(actual-p)))
                return False
            seed=q
        return True

    def fit_held(self,packet):
        frames=[]
        for name,f in zip(('front','left_wrist','right_wrist'),packet.rgbd.frames):
            p,r,fovy=packet.calibrations[name]
            frames.append(FitFrame(f.rgb,f.depth_m,p,r,fovy,name))
        pos,rot=self.motion.pose(packet.observation['observation.state'])
        estimates=localize_held_batch(frames,pos,rot,self.motion.pad_offset,
            self.spec['cookie_half_size'],source_floor_z=self.scene.source_box.position[2]+self.spec['source_floor_top_offset_m'])
        if self.guarded_release:
            self.held_release_shapes=[(rot.T@(e.position-pos),rot.T@e.rotation) for e in estimates]
        center=self.goal.target_position+self.goal.target_rotation@self.goal.target_positions_box.mean(0)
        self.place=placement_goal(estimates,pos,rot,self.goal.target_rotation,center+[0,0,.002])
        if self.placement_planning:
            from .placement_planning import choose_placement
            selected,diagnostics=choose_placement(self.motion,packet.observation['observation.state'],
                estimates,pos,rot,self.goal.target_rotation,center+[0,0,.002],
                emit=lambda **row:self.emit(**row),target_box=self.scene.target_box,
                half_size=self.spec['cookie_half_size'],floor_offset=self.spec['floor_offset_m'])
            self.place=(selected.position,selected.rotation)
            self.placement_plans+=1
            self.placement_symmetry_changes+=int(selected.yaw_radians!=0.)
            self.placement_tilt_changes+=int(selected.tilt_radians!=0.)
            self.placement_diagnostics=diagnostics
            self.placement_reference_by_phase={phase:np.asarray(q,dtype=float)
                for phase,q in diagnostics[-1]['endpoint_joints'].items()}
        self.emit('visual_held_five',positions=[e.position.tolist() for e in estimates],
                  fit_errors=[e.fit_error_m for e in estimates])
        self.transition('TRANSPORT',(self.place[0]+[0,0,.10],self.place[1]))

    def _refined_view(self,packet):
        frame=packet.rgbd.frames[1]
        rim=max(self.scene.source_box.rim_z,self.scene.target_box.rim_z)
        visible=(frame.world_from_optical[2,3]>=rim+.18 and observed_roi(frame,self.view_roi))
        return view_completion_ready(age=self.age,stationary_frames=self.encoder_settling.count,
            pad_z=float(packet.pad_midpoint_world[2]),rim_z=rim,jaw_gap=packet.jaw_gap_m,roi_visible=visible)

    def _refined_release(self,packet):
        scene=self.scene_from(packet)
        if scene is None:self.insert_visual_frames=0;return False
        current=visual_region_complete(scene,self.goal,xy_tolerance=.007,z_tolerance=.004)
        previous=all(visual_region_complete(scene,g,xy_tolerance=.007,z_tolerance=.004) for g in self.completed)
        positions=np.asarray([c.position for c in scene.target_cookies])
        stable=False
        if current and previous:
            stable=True
            if self.insert_visual_positions is not None:
                if len(positions)!=len(self.insert_visual_positions):stable=False
                elif len(positions):
                    from scipy.optimize import linear_sum_assignment
                    d=np.linalg.norm(positions[:,None]-self.insert_visual_positions[None],axis=2)
                    a,b=linear_sum_assignment(d);stable=bool(np.max(d[a,b])<.0015)
        self.insert_visual_frames=self.insert_visual_frames+1 if stable else 0
        self.insert_visual_positions=positions
        pe,re=self.motion.errors(packet.observation['observation.state'],*self.move_goal)
        return insertion_release_ready(age=self.age,stationary_frames=self.encoder_settling.count,
            visual_frames=self.insert_visual_frames,position_error=pe,rotation_error=re,
            current_complete=current,previous_complete=previous)

    def _maybe_guide_placement(self,state,*,measured_new):
        if not (self.placement_planning and self.refine and measured_new and self.age>=60
                and self.encoder_settling.count>=8 and self.phase in self.placement_reference_by_phase
                and self.placement_guided_phase!=self.phase):return
        pe,re=self.motion.errors(state,*self.move_goal)
        tracking=float(np.max(abs(state[:7]-self.motion.last[:7])))
        if (pe>.0025 or re>.045) and tracking<.01:
            self.motion.reference_q=self.placement_reference_by_phase[self.phase].copy()
            self.placement_guided_phase=self.phase;self.placement_guidance_activations+=1
            self.emit('placement_guidance_activated',age=self.age,position_error_m=pe,
                rotation_error_rad=re,joint_tracking_error_rad=tracking,
                reference_q=self.motion.reference_q.tolist(),maximum_joint_command_step_rad=.025)

    def act(self,packet,*,fresh=True):
        self.steps+=1;self.age+=1
        state=packet.observation['observation.state']
        if self.finished:return self.motion.hold()
        measured_new=self.encoder_settling.update(state,packet.timestamp) if fresh else False
        # Independent acquisition monitor survives VIEW -> VERIFY -> LOCATE.
        # Planning-stage transitions still reset the original V15 monitor.
        if fresh and self.visual_memory is not None:self.memory_settling.update(state,packet.timestamp)
        if self.scene_state_memory:
            preparing=(self.phase in ('LOCATE','OBSERVE_HOLD') or
                (self.phase in ('VIEW','VIEW_LIFT') and getattr(self,'after_view',None)=='LOCATE'))
            if preparing:
                self._ensure_scene_observation_batch()
                if self.steps-self.scene_preparation_start>self.scene_max_preparation_steps:
                    self.fail('Scene observation preparation budget exhausted');return self.motion.hold()
        if self.phase=='OBSERVE_HOLD':
            return self._act_observation_hold(packet,fresh=fresh)
        if self.age>260:
            self.fail(self.phase+' timed out');return self.motion.hold()
        # Full scene fitting is needed at batch boundaries. During the robot's
        # continuous motion, encoder feedback closes the servo; held fitting
        # separately uses the current RGB-D at its explicit validation stage.
        scene=self.scene_from(packet) if fresh and self.phase in ('LOCATE','VERIFY','FINAL_SETTLE') else None
        if fresh and (self.phase=='LOCATE' or self.age%8==0):self._remember_view(packet,scene)
        try:
            if self.refine and measured_new and self.phase=='VIEW' and self._refined_view(packet):
                pe,re=self.motion.errors(state,*self.move_goal)
                self.emit('view_completed_from_measured_coverage',position_error_m=pe,
                    rotation_error_rad=re,stationary_frames=self.encoder_settling.count)
                action=self.motion.resync(state,self.opening if self.completed else 1.)
                if fresh:self._remember_view(packet,scene)
                if self.after_view=='LOCATE':self._locate_after_observation('view_completed')
                else:self.transition(self.after_view)
                return action
            if self.refine and measured_new and self.phase=='INSERT' and self.age>=40 and self.encoder_settling.count>=6:
                if self._refined_release(packet):
                    pe,re=self.motion.errors(state,*self.move_goal)
                    self.emit('visual_insert_release_ready',position_error_m=pe,rotation_error_rad=re,
                        stable_visual_frames=self.insert_visual_frames)
                    pos,rot=self.motion.pose(state);self.place=(pos.copy(),rot.copy())
                    action=self.motion.resync(state,self.grip)
                    self.start_release(packet)
                    return action
            if self.phase in ('VIEW','INSERT') and (self.age==1 or self.age%50==0):
                pe,re=self.motion.errors(state,*self.move_goal)
                self.emit('measured_motion_status',age=self.age,position_error_m=pe,rotation_error_rad=re,
                    stationary_frames=self.encoder_settling.count,
                    joint_command_error=float(np.max(abs(state[:7]-self.motion.last[:7]))))
            if self.phase=='LOCATE':
                if scene is None:
                    if self.age>30:self.fail(self.visual_error or 'No visual scene')
                    return self.motion.hold(1.)
                if self.motion.pad_offset is None:
                    pos,rot=self.motion.pose(state)
                    self.motion.pad_offset=rot.T@(packet.pad_midpoint_world-pos)
                if self.scene_state_memory and self.scene_batch_hold_count<1:
                    self._locate_after_observation('initial_scene_observation')
                    return self.motion.hold(1.)
                try:self.choose_pick(packet,scene)
                except LocalizationFailure as exc:
                    if self.scene_state_memory:self._request_scene_observation(packet,exc)
                    elif self.view_attempt>=3:self.fail(str(exc))
                    else:
                        self.emit('reobserve_needed',reason=str(exc));self.start_observation(packet)
                        self.view_attempt+=1
                return self.motion.hold(1.)
            if self.phase=='PINCH':
                # Encoder resistance is a candidate contact signal, confirmed by
                # airborne RGB-D next. It is never treated as object identity.
                if self.pinch_stable<3:self.grip=max(.24,self.grip-.004)
                resistance=state[7]-self.grip>.025
                self.pinch_stable=self.pinch_stable+1 if resistance else 0
                action,_=self.motion.command(state,*self.pick,self.grip,speed=.0008)
                if self.pinch_stable>=3:
                    self.emit('encoder_grasp_candidate',command=self.grip,measured=float(state[7]))
                    pos,rot=self.motion.pose(state)
                    self.transition('LIFT',(pos+[0,0,.14],rot))
                return action
            if self.phase=='HELD':
                if fresh:
                    try:self.fit_held(packet)
                    except LocalizationError as exc:
                        self.visual_error=str(exc)
                        if self.age>30:self.fail(str(exc))
                return self.motion.hold(self.grip)
            if self.phase=='FINAL_SETTLE':
                if fresh:
                    ready=(scene is not None and
                        all(visual_region_complete(scene,g,xy_tolerance=.007,z_tolerance=.004)
                            for g in self.completed))
                    positions=(np.asarray([c.position for c in scene.target_cookies])
                        if scene is not None else None)
                    settled=self.final_settling.update(state,packet.timestamp,positions,
                        complete=ready,jaw_gap=packet.jaw_gap_m)
                    if settled:
                        self.emit('final_visual_settled',independent_frames=self.final_settling.frames,
                            continuous_seconds=self.final_settling.elapsed_s,
                            required_seconds=self.final_settling.min_duration_s,
                            required_frames=self.final_settling.min_frames,
                            native_success_input_used=False)
                        self.transition('DONE')
                    elif self.age==1 or self.age%20==0:
                        self.emit('final_visual_settling',independent_frames=self.final_settling.frames,
                            continuous_seconds=self.final_settling.elapsed_s,
                            reset_reason=self.final_settling.reset_reason)
                return self.motion.hold(self.opening)
            if self.phase=='VERIFY':
                if scene is not None:
                    ready=(packet.jaw_gap_m is not None and packet.jaw_gap_m>.0347 and
                        all(visual_region_complete(scene,g,xy_tolerance=.007,z_tolerance=.004)
                            for g in [*self.completed,self.goal]))
                    positions=np.asarray([c.position for c in scene.target_cookies])
                    if ready and (self.last_visual_timestamp is None or packet.timestamp>self.last_visual_timestamp):
                        stable=True
                        if self.last_positions is not None and len(positions)==len(self.last_positions):
                            from scipy.optimize import linear_sum_assignment
                            d=np.linalg.norm(positions[:,None]-self.last_positions[None],axis=2)
                            a,b=linear_sum_assignment(d);stable=bool(np.max(d[a,b])<.0015)
                        self.visual_stable=self.visual_stable+1 if stable else 1
                        self.last_positions=positions;self.last_visual_timestamp=packet.timestamp
                    else:self.visual_stable=0
                    if self.visual_stable>=3:
                        self.completed.append(self.goal);self.emit('visual_batch_verified',batches=len(self.completed))
                        if len(self.completed)>=self.max_batches:self.transition('FINAL_SETTLE')
                        else:
                            self._remember_verified_boundary(packet,scene)
                            self.goal=None;self.view_attempt=0;self.visual_stable=0
                            self.view_focus='source_entry';self.blocked_region=None
                            self.last_positions=None;self.last_visual_timestamp=None
                            self._locate_after_observation('verified_batch_boundary')
                    elif self.age>50:
                        self.emit('verification_incomplete',target_visible=len(scene.target_cookies))
                        if self.view_attempt>=3:self.fail('Visual release/previous-progress verification failed')
                        else:self.start_observation(packet,verify=True);self.view_attempt+=1
                elif self.age>60:
                    if self.view_attempt>=3:self.fail(self.visual_error or 'No target view')
                    else:self.start_observation(packet,verify=True);self.view_attempt+=1
                return self.motion.hold(self.opening)
            opening=1.;speed=.004
            if self.phase in ('LIFT','TRANSPORT','PREINSERT','INSERT'):opening=self.grip
            if self.phase=='PRE_CLOSE':opening=self.insert
            if self.phase=='DESCEND':opening=self.insert;speed=.0015
            if self.phase=='LIFT':speed=.002
            if self.phase=='INSERT':speed=.001
            if self.phase in ('RETREAT','RELEASE_LIFT') or (self.phase in ('VIEW_LIFT','VIEW') and self.completed):
                opening=self.opening
            if self.phase=='OPEN':
                limit=self.release_plan.opening if self.guarded_release else .49
                self.opening=min(limit,self.opening+.01);opening=self.opening;speed=.0008
            if self.phase=='RELEASE_LIFT':speed=.0015
            if self.phase=='RELEASE_OPEN':
                # The complete jaw sweep is already above the protected bound.
                # Finish the original upward retreat while opening, without an
                # unnecessary second stationary dwell at the clearance height.
                self.opening=min(.49,self.opening+.01);opening=self.opening;speed=.004
            self._maybe_guide_placement(state,measured_new=measured_new)
            action,reached=self.motion.command(state,*self.move_goal,opening,speed=speed)
            self.stable=self.stable+1 if reached else 0
            if self.phase=='OPEN':
                if self.guarded_release:
                    gap_ready=(measured_new and packet.jaw_gap_m is not None and
                        np.isfinite(packet.jaw_gap_m) and
                        packet.jaw_gap_m>=self.release_plan.target_gap_m-.0005 and
                        self.encoder_settling.count>=3)
                    if measured_new:self.release_gap_frames=self.release_gap_frames+1 if gap_ready else 0
                    if self.age>=25 and self.release_gap_frames>=3:
                        pos,rot=self.motion.pose(state)
                        self.release_retreat_goal=(pos+[0,0,.13],rot.copy())
                        self.emit('guarded_release_gap_confirmed',measured_gap_m=packet.jaw_gap_m,
                            target_gap_m=self.release_plan.target_gap_m,independent_frames=self.release_gap_frames)
                        self.transition('RELEASE_LIFT',(pos+[0,0,self.release_lift_m],rot))
                elif self.opening>=.49 and self.age>=25:
                    pos,rot=self.motion.pose(state);self.transition('RETREAT',(pos+[0,0,.13],rot))
                return action
            if self.phase=='RELEASE_LIFT':
                low=self.release_sweep_lowest(state)
                clear=low>=self.release_protected_top+.002
                if measured_new:self.release_clearance_frames=self.release_clearance_frames+1 if clear else 0
                if self.stable>=3 and self.release_clearance_frames>=3:
                    pos,rot=self.motion.pose(state)
                    self.emit('guarded_release_detached',measured_pad_lowest_z_m=low,
                        protected_top_z_m=self.release_protected_top,
                        measured_clearance_m=low-self.release_protected_top,
                        independent_frames=self.release_clearance_frames)
                    self.transition('RELEASE_OPEN',self.release_retreat_goal)
                return action
            if self.phase=='RELEASE_OPEN':
                if self.opening>=.49 and self.age>=25 and self.stable>=3:
                    self.transition('RETREAT',self.release_retreat_goal)
                return action
            if self.stable<3:return action
            if self.phase=='VIEW_LIFT':self.transition('VIEW',self.view_goal)
            elif self.phase=='VIEW':
                if fresh:self._remember_view(packet,scene)
                if self.after_view=='LOCATE':self._locate_after_observation('view_completed')
                else:self.transition(self.after_view)
            elif self.phase=='APPROACH':self.transition('PRE_CLOSE',self.move_goal)
            elif self.phase=='PRE_CLOSE':
                if abs(state[7]-self.insert)<.008:self.transition('DESCEND',self.pick)
            elif self.phase=='DESCEND':self.grip=self.insert;self.pinch_stable=0;self.transition('PINCH',self.pick)
            elif self.phase=='LIFT':self.transition('HELD')
            elif self.phase=='TRANSPORT':self.transition('PREINSERT',(self.place[0]+[0,0,.035],self.place[1]))
            elif self.phase=='PREINSERT':self.transition('INSERT',self.place)
            elif self.phase=='INSERT':self.start_release(packet)
            elif self.phase=='RETREAT':self.view_attempt=0;self.start_observation(packet,verify=True)
            return action
        except (LocalizationFailure,LocalizationError) as exc:
            self.fail(str(exc));return self.motion.hold()
