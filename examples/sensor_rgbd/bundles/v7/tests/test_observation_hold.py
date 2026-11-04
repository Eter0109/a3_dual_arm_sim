import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from a3_continuability.observation_hold import ObservationHold
from a3_continuability.sensor_batch_loop import SensorBatchLoop
from a3_batch_skills.rgbd_localization import LocalizationFailure
import test_visual_memory as fixtures_module
from a3_continuability.visual_memory import BatchVisualMemory


class Motion:
    def __init__(self):
        self.last=np.r_[np.full(7,.1),.49]
        self.pad_offset=None
    def hold(self,opening=None):
        if opening is not None:self.last[7]=opening
        return self.last.copy()


class ObservationHoldTests(unittest.TestCase):
    def sample(self,hold,time,state=None,*,fresh=True,jaw=.04,opening=.49):
        return hold.update(np.zeros(8) if state is None else state,time,jaw,
            fresh=fresh,opening=opening)

    def loop(self,memory=None,enabled=True):
        loop=SensorBatchLoop(Motion(),dict(required_cookies=10,schema='a3_rgbd_geometry_v2',
            cookie_half_size=[.02,.003,.02],wall_thickness_m=.005),visual_memory=memory,
            settle_observe=enabled)
        loop._locate_after_observation('test_view_boundary')
        return loop

    def packet(self,time,state=None,jaw=.04):
        return SimpleNamespace(timestamp=time,jaw_gap_m=jaw,rgbd=object(),
            observation={'observation.state':np.zeros(8) if state is None else state})

    def test_stale_frames_and_duplicate_timestamps_cannot_build_settling(self):
        hold=ObservationHold()
        self.sample(hold,0,fresh=False)
        self.assertIsNone(hold.target)
        self.sample(hold,1)
        for _ in range(8):
            self.assertFalse(self.sample(hold,1)['ready'])
        self.assertEqual(hold.fresh_samples,1)
        self.assertEqual(hold.monitor.count,0)
        self.sample(hold,2);self.sample(hold,3)
        self.assertTrue(self.sample(hold,4)['ready'])

    def test_target_latches_once_preserving_commanded_gripper_without_chasing(self):
        hold=ObservationHold()
        state=np.r_[np.arange(7)*.1,.9]
        self.sample(hold,1,state)
        target=hold.target.copy()
        state[:7]+=.001
        row=self.sample(hold,2,state,opening=.8)
        np.testing.assert_array_equal(hold.target,target)
        self.assertEqual(hold.target[7],.49)
        self.assertAlmostEqual(row['tracking_error_rad'],.001)
        self.assertAlmostEqual(row['joint_delta_rad'],.001)
        self.assertEqual(row['stationary_frames'],0)

    def test_readiness_requires_measured_open_jaw_and_independent_current_sample(self):
        hold=ObservationHold()
        for i in range(4):
            self.assertFalse(self.sample(hold,i,jaw=None)['ready'])
        self.assertFalse(self.sample(hold,4,jaw=.02)['ready'])
        self.assertFalse(self.sample(hold,4,jaw=.04)['ready'])
        self.assertTrue(self.sample(hold,5,jaw=.04)['ready'])

    def test_timeout_counts_all_control_steps_and_never_certifies_moving_encoders(self):
        hold=ObservationHold()
        for i in range(40):
            row=self.sample(hold,float(i),np.r_[np.full(7,i*.002),.49])
            self.assertFalse(row['ready'])
            self.assertEqual(row['timed_out'],i==39)
        self.assertEqual(hold.steps,40)
        self.assertGreater(row['max_joint_delta_rad'],.0019)

    def test_loop_hold_behavior_identical_with_and_without_memory(self):
        loops=[self.loop(),self.loop(Mock())]
        for loop in loops:loop.localizer=SimpleNamespace(observe=Mock(return_value=object()))
        actions=[]
        for loop in loops:
            actions.append([loop.act(self.packet(i),fresh=True) for i in range(4)])
            self.assertEqual(loop.phase,'LOCATE')
            self.assertEqual(loop.observation_hold_completed,1)
            self.assertEqual(loop.observation_hold_steps,4)
            self.assertEqual(loop.steps,4)
        np.testing.assert_array_equal(actions[0],actions[1])
        self.assertEqual(actions[0][0][7],.49)

    def test_capture_receives_only_final_fresh_packet_and_cannot_mutate_planning_scene(self):
        memory=Mock();loop=self.loop(memory)
        old_scene=object();current_scene=object()
        loop.scene=old_scene;loop.visual_error='preserve'
        loop.localizer=SimpleNamespace(observe=Mock(return_value=current_scene))
        for i in range(3):loop.act(self.packet(i),fresh=True)
        memory.observe.assert_not_called()
        stale=self.packet(2)
        loop.act(stale,fresh=False)
        loop.localizer.observe.assert_not_called()
        final=self.packet(3)
        loop.act(final,fresh=True)
        memory.observe.assert_called_once_with(final,current_scene,batch=0,
            phase='OBSERVE_HOLD',stationary=True)
        self.assertIs(loop.scene,old_scene)
        self.assertEqual(loop.visual_error,'preserve')

    def test_hold_timeout_is_explicit_failure_and_consumes_budget(self):
        loop=self.loop()
        for i in range(40):
            loop.act(self.packet(float(i),np.r_[np.full(7,.002*i),.49]),fresh=True)
        self.assertEqual(loop.steps,40)
        self.assertEqual(loop.phase,'FAILED')
        self.assertIn('OBSERVE_HOLD timed out',loop.failure_reason)
        self.assertEqual(loop.observation_hold_completed,0)

    def test_hold_disabled_uses_original_locate_transition_without_dwell(self):
        loop=self.loop(enabled=False)
        self.assertEqual(loop.phase,'LOCATE')
        self.assertIsNone(loop.observation_hold)
        self.assertEqual(loop.observation_hold_steps,0)
        np.testing.assert_array_equal(loop.motion.last,np.r_[np.full(7,.1),.49])

    def test_failed_current_fit_returns_to_live_locate_without_inventing_memory(self):
        memory=Mock();loop=self.loop(memory)
        loop.localizer=SimpleNamespace(observe=Mock(side_effect=LocalizationFailure('rim occluded')))
        old=object();loop.scene=old
        for i in range(4):loop.act(self.packet(i),fresh=True)
        self.assertEqual(loop.phase,'LOCATE')
        self.assertIs(loop.scene,old)
        memory.observe.assert_not_called()

    def test_real_memory_retains_original_pose_and_time_until_manipulation(self):
        fixtures=fixtures_module.VisualMemoryTests()
        memory=BatchVisualMemory();loop=self.loop(memory)
        loop.completed=[object()]
        loop.localizer=SimpleNamespace(observe=lambda frame:fixtures.scene(frame.timestamp))
        final=None
        for i in range(4):
            final=fixtures.packet(1.+.1*i)
            final.jaw_gap_m=.04;final.observation={'observation.state':np.zeros(8)}
            loop.act(final,fresh=True)
        self.assertEqual(memory.batch,1)
        self.assertEqual(memory.captures,1)
        self.assertIs(memory.views[0].frame,final.rgbd.frames[0])
        self.assertEqual(memory.views[0].frame.timestamp,1.3)
        loop.transition('APPROACH')
        self.assertEqual(memory.views,[])


if __name__=='__main__':unittest.main()
