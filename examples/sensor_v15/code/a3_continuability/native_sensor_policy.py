"""External module:factory policy for the unmodified main benchmark runner.

Binding exposes a rendering/robot sensor adapter; the controller receives only
ExecutionPacket. Native benchmark results are never passed into decisions.
"""
import json
import os
import re
from pathlib import Path
import numpy as np
from .three_camera_sensor import ThreeCameraRGBD
from .sensor_batch_loop import EncoderMotion,SensorBatchLoop
from .sensor_boundary import numeric_json


class NativeSensorPolicy:
    name='three_rgbd_visual_expert'
    action_mode='joint_position'

    def __init__(self,output=None):
        self.output=Path(output) if output else None
        self.sensor=None;self.loop=None;self.video=None;self.events=None
        self.steps=0;self.summaries=[]

    def bind_env(self,env):
        from a3_batch_skills.rgbd import geometry_spec_from_env
        if os.environ.get('A3_SENSOR_REQUIRE_RELEASE')=='1':
            from dataclasses import replace
            # Explicit auxiliary evaluation protocol: keep main's predicate,
            # continue past its early success until visual release completes.
            env.task_config=replace(env.task_config,terminate_on_success=False)
        self._close_episode()
        before={key:getattr(env.model,key).copy() for key in ('cam_pos','cam_quat','cam_mode','cam_fovy')}
        table=os.environ.get('A3_SENSOR_TABLE_HEIGHT_M')
        self.sensor=ThreeCameraRGBD(env,preserve_native=True,resolution=768,
            table_height_m=float(table) if table is not None else None)
        assert all(np.array_equal(value,getattr(env.model,key)) for key,value in before.items())
        self.spec=geometry_spec_from_env(env)
        # Only the controller scales are retained by EncoderMotion.
        self.robot_config=env.config

    def reset(self,context=None):
        self.loop=None;self.context=context;self.steps=0;self.packet=None;self.saved_phase=None
        self.scout=None
        self.refine=os.environ.get('A3_SENSOR_REFINEMENT','0')=='1'
        self.handeye=None;self.handeye_done=os.environ.get('A3_SENSOR_HANDEYE','0')!='1'
        self.active_right=os.environ.get('A3_SENSOR_ACTIVE_RIGHT','0')=='1'
        self.right_each_batch=os.environ.get('A3_SENSOR_RIGHT_EACH_BATCH','0')=='1'
        self.scanned_batches=set();self.scan_records=[]
        if self.output:
            self.directory=self.output/f'seed_{context.seed}'
            self.directory.mkdir(parents=True,exist_ok=False)
            self.events=(self.directory/'events.jsonl').open('w',buffering=1)
            self.controls=(self.directory/'control.jsonl').open('w',buffering=1)
            (self.directory/'sensor.json').write_text(json.dumps(dict(self.sensor.protocol,
                optional_right_observation=self.active_right),indent=2))

    @property
    def phase_name(self):
        if not getattr(self,'handeye_done',True):return 'CALIBRATE'
        if getattr(self,'scout',None) is not None and not self.scout.done:return 'RIGHT_SCAN'
        return self.loop.phase if self.loop else 'INITIALIZING'

    @property
    def finished(self):return bool(self.loop and self.loop.finished)

    def act(self,observation,task=''):
        if not self.handeye_done:
            from .handeye_calibration import HandEyeScan
            if self.handeye is None:
                motion=EncoderMotion(self.sensor.calibration.model,self.robot_config,observation['observation.state'])
                self.handeye=HandEyeScan(motion,self.spec,emit=lambda **row:self._event(dict(step=self.steps,phase='CALIBRATE',**row)))
            if self.handeye.steps==0 or self.handeye.stable>=7:self.sensor.request_recalibration()
            packet=self.sensor.capture(observation)
            action=self.handeye.act(packet,self.sensor.nominal_frames[1]);self.steps+=1
            if self.output:
                (self.directory/'status.json').write_text(json.dumps(dict(step=self.steps,phase='CALIBRATE',completed_batches=0)))
            if self.handeye.done:
                self.handeye_done=True
                if self.output:
                    folder=self.directory/'handeye';folder.mkdir(exist_ok=True)
                    for i,(robot,frame) in enumerate(self.handeye.samples):
                        np.savez_compressed(folder/f'view_{i:02d}.npz',robot=robot,
                            rgb=frame.rgb,depth=frame.depth_m,intrinsic=frame.intrinsic,
                            world_from_optical=frame.world_from_optical)
                try:
                    if self.handeye.failure:raise ValueError(self.handeye.failure)
                    details=self.sensor.install_handeye(self.handeye)
                    self._event(dict(step=self.steps,event='handeye_accepted',**details))
                    self._save_calibration('handeye_complete')
                    if self.output:(self.directory/'sensor.json').write_text(json.dumps(self.sensor.protocol,indent=2))
                except (ValueError,RuntimeError,IndexError) as exc:
                    self.loop=SensorBatchLoop(self.handeye.motion,self.spec,event_sink=self._event,source_column=1,refine=self.refine)
                    self.loop.fail('Calibration rejected: '+str(exc))
                    self.right=np.asarray(observation['observation.state'][8:]).copy()
                if self.output:
                    (self.directory/'handeye_trials.json').write_text(json.dumps(self.handeye.calibration_trials,indent=2))
                self.packet=None;self.saved_phase=None
            return np.r_[action,observation['observation.state'][8:]]
        from .right_observation import scan_due
        completed=len(self.loop.completed) if self.loop else 0
        if self.active_right and scan_due(completed,
                self.loop.phase if self.loop else 'INITIALIZING',self.scanned_batches,
                each_batch=self.right_each_batch):
            from .right_observation import RightObservationScan
            if self.scout is None:
                requested=os.environ.get('A3_SENSOR_SOURCE_COLUMN','1')
                self.scout=RightObservationScan(self.sensor.calibration,self.robot_config,
                    observation['observation.state'],self.spec,column=int(requested),
                    emit=lambda **row:self._event(dict(step=self.steps,phase='RIGHT_SCAN',**row)))
            scan_prefix=f'right_scan_batch_{completed:02d}'
            if self.scout.steps==0 or self.scout.settled>=7:self.sensor.request_recalibration()
            packet=self.sensor.capture(observation)
            if self.scout.steps==0:self._save_calibration(f'batch_{completed:02d}_scan_start')
            packet.validate_age(float(observation['time']),max_age_s=.11)
            if self.output and self.scout.steps==0:
                from a3_batch_skills.rgbd import save_rgbd_packet
                save_rgbd_packet(self.directory/(scan_prefix+'_start.npz'),packet.rgbd)
            command=self.scout.act(packet,observation['observation.state'][8:]);self.steps+=1
            if self.output:
                self.controls.write(json.dumps(dict(step=self.steps,phase='RIGHT_SCAN',
                    action=command[:8].tolist(),right_action=command[8:].tolist()))+'\n')
                if self.steps%10==1 or self.scout.done:
                    (self.directory/'status.json').write_text(json.dumps(dict(step=self.steps,
                        phase='RIGHT_SCAN',completed_batches=completed,failure_reason=None)))
                if self.scout.done:
                    from a3_batch_skills.rgbd import save_rgbd_packet
                    save_rgbd_packet(self.directory/(scan_prefix+'_end.npz'),packet.rgbd)
                    (self.directory/(scan_prefix+'.json')).write_text(json.dumps(dict(
                        steps=self.scout.steps,quality=self.scout.best_quality,
                        candidates=0 if self.scout.candidates is None else len(self.scout.candidates),
                        history=self.scout.history,right_hold=command[8:].tolist(),
                        calibration=self.sensor.calibration_diagnostics,
                        uses_object_truth=False),indent=2))
            if self.scout.done:
                self.scan_records.append(dict(batch=completed,steps=self.scout.steps,
                    quality=self.scout.best_quality))
                self.scanned_batches.add(completed)
                self.right=command[8:].copy()
                if self.loop is not None:
                    self.loop.motion.right=self.right.copy()
                    self.loop.motion.last=command[:8].copy()
                self.packet=None;self.saved_phase=None
                self.scout=None
            return command
        if self.loop is None:
            self.right=np.asarray(observation['observation.state'][8:],dtype=float).copy()
            motion=EncoderMotion(self.sensor.calibration.model,self.robot_config,observation['observation.state'])
            match=re.search(r'source column\s+([1-4])',self.context.task,re.IGNORECASE)
            requested=os.environ.get('A3_SENSOR_SOURCE_COLUMN')
            self.loop=SensorBatchLoop(motion,self.spec,event_sink=self._event,
                source_column=int(requested) if requested else (int(match.group(1)) if match else 'auto'),refine=self.refine)
        from .sensor_boundary import needs_sensor_refresh
        fresh=needs_sensor_refresh(self.steps,self.loop.age,self.loop.phase,self.saved_phase)
        if fresh:
            recalibrate=self.loop.phase in ('LOCATE','VERIFY') and self.loop.age==0
            if recalibrate:self.sensor.request_recalibration()
            self.packet=self.sensor.capture(observation)
            if recalibrate:self._save_calibration(f'batch_{completed:02d}_{self.loop.phase}_{self.steps:04d}')
        self.packet.validate_age(float(observation['time']),max_age_s=.11)
        action=self.loop.act(self.packet,fresh=fresh)
        snapshots=os.environ.get('A3_SENSOR_SNAPSHOTS','stages')
        save_packet=(snapshots=='stages' and self.saved_phase!=self.loop.phase) or (snapshots=='failure' and self.loop.phase=='FAILED')
        if self.output and save_packet:
            from a3_batch_skills.rgbd import save_rgbd_packet
            save_rgbd_packet(self.directory/f'{self.steps:04d}_{self.loop.phase}.npz',self.packet.rgbd)
        self.saved_phase=self.loop.phase;self.steps+=1
        if self.output:
            self.controls.write(json.dumps(dict(step=self.steps,phase=self.phase_name,action=action.tolist()))+'\n')
            if self.steps%25==1 or self.finished:
                (self.directory/'status.json').write_text(json.dumps(dict(step=self.steps,
                    phase=self.phase_name,completed_batches=len(self.loop.completed),
                    failure_reason=self.loop.failure_reason)))
        return np.r_[action,self.right]

    def _event(self,row):
        if self.events:self.events.write(json.dumps(row,default=numeric_json)+'\n')
        print(json.dumps(dict(sensor_policy=True,**row),default=numeric_json),flush=True)

    def _save_calibration(self,tag):
        if not self.output or self.sensor.table_height_m is None:return
        folder=self.directory/'calibration';folder.mkdir(exist_ok=True)
        record=dict(step=self.steps,tag=tag,diagnostics=self.sensor.calibration_diagnostics,
            mount_corrections={name:matrix.tolist() for name,matrix in self.sensor.mount_corrections.items()},
            uses_sampled_object_or_camera_pose=False)
        (folder/(tag+'.json')).write_text(json.dumps(record,indent=2))

    def _close_episode(self):
        if self.loop is not None:
            record=dict(seed=self.context.seed,steps=self.steps,phase=self.phase_name,
                completed_batches=len(self.loop.completed),failure_reason=self.loop.failure_reason,
                right_observation_steps=sum(row['steps'] for row in self.scan_records),
                right_observation_batches=self.scan_records)
            record['handeye_steps']=self.handeye.steps if self.handeye else 0
            record['task_conditioned_completion']=self.refine
            record['handeye_accepted']=self.sensor.handeye_ready if self.sensor else False
            self.summaries.append(record)
            if self.output:(self.directory/'controller.json').write_text(json.dumps(record,indent=2))
            self.loop=None
        self.scout=None
        if self.events:self.events.close();self.events=None
        if getattr(self,'controls',None):self.controls.close();self.controls=None
        if self.sensor:self.sensor.close();self.sensor=None

    def close(self):self._close_episode()

    def __del__(self):
        try:self._close_episode()
        except Exception:pass


def make_policy():
    return NativeSensorPolicy(os.environ.get('A3_SENSOR_NATIVE_OUTPUT'))
