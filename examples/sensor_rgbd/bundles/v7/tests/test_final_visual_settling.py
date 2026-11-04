"""Final completion needs independent sensing over time, with no native signal."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np

from a3_continuability.task_completion import FinalVisualSettling
from a3_continuability.sensor_batch_loop import SensorBatchLoop


class FinalVisualSettlingTests(unittest.TestCase):
    def positions(self):return np.array([[0.,i*.01,.02] for i in range(5)])

    def update(self,monitor,t,**values):
        state=values.pop('state',np.zeros(8));positions=values.pop('positions',self.positions())
        options=dict(complete=True,jaw_gap=.04);options.update(values)
        return monitor.update(state,t,positions,**options)

    def test_fast_frames_cannot_replace_half_second(self):
        monitor=FinalVisualSettling()
        for i in range(20):self.assertFalse(self.update(monitor,i*.01))
        self.assertGreaterEqual(monitor.frames,6)
        self.assertLess(monitor.elapsed_s,.5)
        for i in range(20,52):ready=self.update(monitor,i*.01)
        self.assertTrue(ready)

    def test_repeated_or_out_of_order_timestamps_do_not_finish(self):
        monitor=FinalVisualSettling();self.update(monitor,0.)
        self.update(monitor,.1)
        for _ in range(40):
            self.assertFalse(self.update(monitor,.1))
            self.assertFalse(self.update(monitor,.05))
        self.assertEqual(monitor.frames,1)
        self.assertEqual(monitor.elapsed_s,0.)
        for i in range(2,7):ready=self.update(monitor,i*.1)
        self.assertTrue(ready)

    def test_object_identity_order_can_change_without_motion(self):
        monitor=FinalVisualSettling()
        for i in range(7):
            positions=self.positions()[::-1] if i%2 else self.positions()
            ready=self.update(monitor,i*.1,positions=positions)
        self.assertTrue(ready)

    def test_loss_closed_jaw_and_encoder_motion_reset_entire_window(self):
        for invalid in [dict(complete=False),dict(jaw_gap=.02),dict(positions=None),
                        dict(state=np.array([0.,0.,.01,0.,0.,0.,0.,0.]))]:
            monitor=FinalVisualSettling()
            for i in range(5):self.update(monitor,i*.1)
            self.assertFalse(self.update(monitor,.5,**invalid))
            self.assertEqual(monitor.frames,0)
            for i in range(6,10):self.assertFalse(self.update(monitor,i*.1))

    def test_slow_drift_and_observation_gap_cannot_accumulate(self):
        monitor=FinalVisualSettling()
        for i in range(10):
            self.assertFalse(self.update(monitor,i*.1,positions=self.positions()+[i*.0004,0,0]))
        self.assertLess(monitor.frames,6)
        self.assertFalse(self.update(monitor,2.,positions=self.positions()+[.0036,0,0]))
        self.assertEqual(monitor.frames,0)
        self.assertEqual(monitor.reset_reason,'independent_observation_gap')

    def loop(self,max_batches=1):
        motion=NS(last=np.r_[np.zeros(7),.49],precision=False)
        motion.hold=lambda opening=None:motion.last.copy()
        spec=dict(schema='a3_rgbd_geometry_v2',required_cookies=10,
            cookie_half_size=[.025,.003166666666667,.0125],wall_thickness_m=.006,
            source_floor_top_offset_m=.756,floor_offset_m=.006,
            target_slots_local=[[x,y] for x in [-.0252,.0252] for y in np.arange(5)*.0088-.0176])
        loop=SensorBatchLoop(motion,spec,max_batches=max_batches)
        loop.goal=object();loop.phase='VERIFY';loop.visual_stable=2
        scene=NS(target_cookies=[NS(position=p) for p in self.positions()])
        loop.localizer=NS(observe=Mock(return_value=scene))
        return loop

    def packet(self,t):
        return NS(timestamp=t,observation={'observation.state':np.zeros(16)},
            jaw_gap_m=.04,rgbd=object())

    def test_last_verify_enters_counted_settle_and_only_live_frames_finish(self):
        loop=self.loop()
        with patch('a3_continuability.sensor_batch_loop.visual_region_complete',return_value=True):
            loop.act(self.packet(0.),fresh=True)
            self.assertEqual(loop.phase,'FINAL_SETTLE');self.assertFalse(loop.finished)
            for _ in range(10):loop.act(self.packet(0.),fresh=False)
            self.assertEqual(loop.final_settling.frames,0)
            for i in range(1,7):
                loop.act(self.packet(i*.1),fresh=True)
                self.assertEqual(loop.phase,'FINAL_SETTLE')
            loop.act(self.packet(.7),fresh=True)
            self.assertEqual(loop.phase,'DONE');self.assertTrue(loop.finished)
        self.assertEqual(loop.steps,18)
        self.assertEqual(len(loop.completed),1)

    def test_intermediate_batches_keep_original_boundary_flow(self):
        loop=self.loop(max_batches=2)
        with patch('a3_continuability.sensor_batch_loop.visual_region_complete',return_value=True):
            loop.act(self.packet(0.),fresh=True)
        self.assertEqual(loop.phase,'LOCATE');self.assertIsNone(loop.final_settling)
        self.assertEqual(len(loop.completed),1)

    def test_final_settle_retains_existing_260_step_hard_limit(self):
        loop=self.loop();loop.transition('FINAL_SETTLE');loop.age=260
        loop.act(self.packet(0.),fresh=False)
        self.assertEqual(loop.phase,'FAILED')
        self.assertEqual(loop.failure_reason,'FINAL_SETTLE timed out')


if __name__=='__main__':unittest.main()
