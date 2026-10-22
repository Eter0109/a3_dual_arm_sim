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
from .task_completion import EncoderSettling,observed_roi,view_completion_ready,insertion_release_ready
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

    def pose(self, state):
        import mujoco
        self.data.qpos[self.addresses] = np.r_[state[:7],self.right[:7]]
        mujoco.mj_kinematics(self.model,self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3,3).copy()

    def hold(self, opening=None):
        if opening is not None:self.last[7]=np.clip(opening,0,1)
        return self.last.copy()

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
                self.ik._ranges[0],self.last[:7],current+bounded[:3]*self.config.cartesian_translation_scale_m,target_quat)
            result=np.r_[q,opening]
        else:result=self.ik.convert(self.data,np.clip(action,-1,1),commanded)[:8]
        self.last=result.copy()
        reached=np.linalg.norm(measured_pos-position)<.0025 and np.linalg.norm(measured_error)<.045
        return result,bool(reached)


class SensorBatchLoop:
    def __init__(self,motion,spec,*,event_sink=None,max_batches=None,source_column='auto',refine=False):
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

    @property
    def finished(self):return self.phase in ('DONE','FAILED')

    def emit(self,event,**values):
        row=dict(step=self.steps,phase=self.phase,event=event,**values)
        self.events.append(row)
        if self.sink:self.sink(row)

    def transition(self,phase,goal=None):
        self.phase,self.age,self.stable=phase,0,0
        self.encoder_settling=EncoderSettling()
        if phase=='INSERT':self.insert_visual_frames=0;self.insert_visual_positions=None
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
        center=centers[min(self.view_attempt,len(centers)-1)]
        # Small measured region of interest. Positive occupancy/completion is
        # still required by LOCATE/VERIFY after the camera has settled.
        if verify:
            half=np.asarray(target.inner_half_size)
            roi=np.array([[x,y,self.spec['floor_offset_m']+.02]
                for x in (-half[0]+.006,0.,half[0]-.006)
                for y in (-half[1]+.006,0.,half[1]-.006)])
            self.view_roi=roi@target.rotation.T+target.position
        else:
            self.view_roi=np.array([[center[0]+dx,center[1]+dy,source.rim_z-.015]
                for dx,dy in ((0,0),(-.020,-.004),(-.020,.004),(.020,-.004),(.020,.004))])
        camera_pos=np.r_[center,source.rim_z+.30]
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

    def choose_pick(self,packet,scene):
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
                    candidate=make_visual_goal(packet.rgbd,scene,self.spec,index,column=column)
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
        center=self.goal.target_position+self.goal.target_rotation@self.goal.target_positions_box.mean(0)
        self.place=placement_goal(estimates,pos,rot,self.goal.target_rotation,center+[0,0,.002])
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

    def act(self,packet,*,fresh=True):
        self.steps+=1;self.age+=1
        state=packet.observation['observation.state']
        if self.finished:return self.motion.hold()
        measured_new=self.encoder_settling.update(state,packet.timestamp) if fresh else False
        if self.age>260:
            self.fail(self.phase+' timed out');return self.motion.hold()
        # Full scene fitting is needed at batch boundaries. During the robot's
        # continuous motion, encoder feedback closes the servo; held fitting
        # separately uses the current RGB-D at its explicit validation stage.
        scene=self.scene_from(packet) if fresh and self.phase in ('LOCATE','VERIFY') else None
        try:
            if self.refine and measured_new and self.phase=='VIEW' and self._refined_view(packet):
                pe,re=self.motion.errors(state,*self.move_goal)
                self.emit('view_completed_from_measured_coverage',position_error_m=pe,
                    rotation_error_rad=re,stationary_frames=self.encoder_settling.count)
                action=self.motion.resync(state,self.opening if self.completed else 1.)
                self.transition(self.after_view)
                return action
            if self.refine and measured_new and self.phase=='INSERT' and self.age>=40 and self.encoder_settling.count>=6:
                if self._refined_release(packet):
                    pe,re=self.motion.errors(state,*self.move_goal)
                    self.emit('visual_insert_release_ready',position_error_m=pe,rotation_error_rad=re,
                        stable_visual_frames=self.insert_visual_frames)
                    pos,rot=self.motion.pose(state);self.place=(pos.copy(),rot.copy())
                    action=self.motion.resync(state,self.grip);self.opening=self.grip
                    self.transition('OPEN',self.place)
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
                try:self.choose_pick(packet,scene)
                except LocalizationFailure as exc:
                    if self.view_attempt>=3:self.fail(str(exc))
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
                        if len(self.completed)>=self.max_batches:self.transition('DONE')
                        else:
                            self.goal=None;self.view_attempt=0;self.visual_stable=0
                            self.view_focus='source_entry';self.blocked_region=None
                            self.last_positions=None;self.last_visual_timestamp=None;self.transition('LOCATE')
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
            if self.phase=='RETREAT' or (self.phase in ('VIEW_LIFT','VIEW') and self.completed):
                opening=self.opening
            if self.phase=='OPEN':
                self.opening=min(.49,self.opening+.01);opening=self.opening;speed=.0008
            action,reached=self.motion.command(state,*self.move_goal,opening,speed=speed)
            self.stable=self.stable+1 if reached else 0
            if self.phase=='OPEN':
                if self.opening>=.49 and self.age>=25:
                    pos,rot=self.motion.pose(state);self.transition('RETREAT',(pos+[0,0,.13],rot))
                return action
            if self.stable<3:return action
            if self.phase=='VIEW_LIFT':self.transition('VIEW',self.view_goal)
            elif self.phase=='VIEW':self.transition(self.after_view)
            elif self.phase=='APPROACH':self.transition('PRE_CLOSE',self.move_goal)
            elif self.phase=='PRE_CLOSE':
                if abs(state[7]-self.insert)<.008:self.transition('DESCEND',self.pick)
            elif self.phase=='DESCEND':self.grip=self.insert;self.pinch_stable=0;self.transition('PINCH',self.pick)
            elif self.phase=='LIFT':self.transition('HELD')
            elif self.phase=='TRANSPORT':self.transition('PREINSERT',(self.place[0]+[0,0,.035],self.place[1]))
            elif self.phase=='PREINSERT':self.transition('INSERT',self.place)
            elif self.phase=='INSERT':self.opening=self.grip;self.transition('OPEN',self.place)
            elif self.phase=='RETREAT':self.view_attempt=0;self.start_observation(packet,verify=True)
            return action
        except (LocalizationFailure,LocalizationError) as exc:
            self.fail(str(exc));return self.motion.hold()
