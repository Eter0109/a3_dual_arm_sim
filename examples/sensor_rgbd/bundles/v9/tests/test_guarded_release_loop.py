import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from a3_continuability.sensor_batch_loop import SensorBatchLoop


class GuardedReleaseLoopTests(unittest.TestCase):
    def loop(self,enabled=True):
        self.commands=[]
        motion=NS(last=np.zeros(8),precision=False,
            pose=lambda state:(np.array([0.,0.,.030]),np.eye(3)),
            pad_height_bounds=Mock(return_value=(.010,.040)),
            hold=lambda *args:np.zeros(8))
        def command(state,pos,rot,opening,**kwargs):
            self.commands.append(opening)
            return np.r_[np.zeros(7),opening],True
        motion.command=command
        spec=dict(schema='a3_rgbd_geometry_v2',wall_thickness_m=.006,
            cookie_half_size=[.025,.003166666667,.0125],
            target_slots_local=[],required_cookies=10)
        loop=SensorBatchLoop(motion,spec,guarded_release=enabled)
        loop.scene=NS(target_box=NS(rim_z=.040),target_cookies=[])
        loop.goal=NS(target_position=np.zeros(3),target_rotation=np.eye(3),
            target_positions_box=np.array([[0.,i*.006333333,.0125] for i in range(5)]))
        loop.place=(np.array([0.,0.,.030]),np.eye(3));loop.grip=.28
        loop.held_release_shapes=[(p-np.array([0.,0.,.030]),np.eye(3))
            for p in loop.goal.target_positions_box]
        return loop

    def packet(self,t,gap=.032,opening=.313):
        state=np.zeros(8);state[7]=opening
        return NS(timestamp=t,jaw_gap_m=gap,observation={'observation.state':state})

    def test_measured_gap_confirmation_precedes_detach_and_full_open(self):
        loop=self.loop();loop.start_release(self.packet(0.))
        target=loop.release_plan.opening
        for i in range(1,31):loop.act(self.packet(i*.1,gap=.032),fresh=True)
        self.assertEqual(loop.phase,'OPEN')
        self.assertLessEqual(max(self.commands),target)
        for i in range(31,36):loop.act(self.packet(i*.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'RELEASE_LIFT')
        for i in range(36,46):loop.act(self.packet(i*.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'RELEASE_LIFT')
        self.assertLessEqual(max(self.commands),target)
        loop.motion.pad_height_bounds.return_value=(.045,.075)
        for i in range(46,50):loop.act(self.packet(i*.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'RELEASE_OPEN')
        for i in range(50,80):
            loop.act(self.packet(i*.1,gap=.047),fresh=True)
            if loop.phase=='RETREAT':break
        self.assertEqual(loop.phase,'RETREAT')
        self.assertAlmostEqual(max(self.commands),.49)

    def test_cached_timestamps_cannot_confirm_gap_or_detachment(self):
        loop=self.loop();loop.start_release(self.packet(0.));loop.age=24
        for _ in range(20):loop.act(self.packet(.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'OPEN');self.assertEqual(loop.release_gap_frames,0)
        loop.release_retreat_goal=loop.place;loop.transition('RELEASE_LIFT',loop.place)
        loop.motion.pad_height_bounds.return_value=(.045,.075)
        for _ in range(20):loop.act(self.packet(.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'RELEASE_LIFT')
        self.assertLess(loop.release_clearance_frames,3)

    def test_disabled_mode_retains_single_open_then_original_retreat(self):
        loop=self.loop(False);loop.start_release(self.packet(0.))
        for i in range(1,26):loop.act(self.packet(i*.1),fresh=True)
        self.assertEqual(loop.phase,'RETREAT');self.assertAlmostEqual(max(self.commands),.49)
        self.assertFalse(any(e['event']=='guarded_release_planned' for e in loop.events))

    def test_missing_measured_gap_does_not_execute_unverified_release(self):
        loop=self.loop()
        with self.assertRaises(RuntimeError):loop.start_release(self.packet(0.,gap=None))
        self.assertEqual(self.commands,[])

    def test_future_full_open_pad_sweep_must_also_clear_protected_geometry(self):
        loop=self.loop();loop.start_release(self.packet(0.))
        loop.release_retreat_goal=loop.place;loop.transition('RELEASE_LIFT',loop.place)
        loop.motion.pad_height_bounds.side_effect=lambda state:((.041,.070) if state[7]==.49 else (.045,.075))
        for i in range(1,10):loop.act(self.packet(i*.1,gap=.034),fresh=True)
        self.assertEqual(loop.phase,'RELEASE_LIFT')
        self.assertEqual(loop.release_clearance_frames,0)

    def test_new_batch_height_uses_measured_shape_and_actual_tool_pose(self):
        loop=self.loop()
        loop.held_release_shapes=[(np.array([0.,0.,.030]),np.eye(3))]*5
        loop.start_release(self.packet(0.))
        self.assertAlmostEqual(loop.release_protected_top,.0725)
        self.assertGreater(loop.release_lift_m,.06)


if __name__=='__main__':unittest.main()
