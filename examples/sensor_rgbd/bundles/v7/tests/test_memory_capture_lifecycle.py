import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from a3_continuability.sensor_batch_loop import SensorBatchLoop
from a3_continuability.task_completion import EncoderSettling
from a3_continuability.visual_memory import BatchVisualMemory
import test_visual_memory as fixtures_module

class CaptureLifecycleTests(unittest.TestCase):
    def loop(self,memory):
        loop=SensorBatchLoop.__new__(SensorBatchLoop)
        loop.visual_memory=memory;loop.memory_settling=SimpleNamespace(count=3)
        loop.encoder_settling=SimpleNamespace(count=0)
        loop.phase='VERIFY';loop.visual_stable=3;loop.completed=[object()];loop.emit=Mock()
        return loop
    def test_stage_transition_does_not_erase_independent_capture_stability(self):
        memory=Mock();loop=self.loop(memory)
        loop.transition('LOCATE')
        self.assertEqual(loop.encoder_settling.count,0)
        self.assertEqual(loop.memory_settling.count,3)
        packet=object();scene=object();loop._remember_view(packet,scene)
        memory.observe.assert_called_once_with(packet,scene,batch=1,phase='LOCATE',stationary=True)
    def test_boundary_needs_measured_open_gripper_stability_and_live_verification(self):
        memory=Mock();loop=self.loop(memory);scene=object();packet=SimpleNamespace(jaw_gap_m=.04)
        for field,bad in [('phase','INSERT'),('visual_stable',2),('completed',[])]:
            old=getattr(loop,field);setattr(loop,field,bad)
            loop._remember_verified_boundary(packet,scene);setattr(loop,field,old)
        loop.memory_settling.count=2;loop._remember_verified_boundary(packet,scene)
        loop.memory_settling.count=3;loop._remember_verified_boundary(SimpleNamespace(jaw_gap_m=None),scene)
        memory.observe.assert_not_called()
        loop._remember_verified_boundary(packet,scene)
        memory.observe.assert_called_once_with(packet,scene,batch=1,phase='BATCH_READY',stationary=True)
    def test_verified_boundary_records_next_batch_and_complements_later_occlusion(self):
        fixtures=fixtures_module.VisualMemoryTests();memory=BatchVisualMemory();loop=self.loop(memory)
        packet=fixtures.packet(1.);packet.jaw_gap_m=.04
        loop._remember_verified_boundary(packet,fixtures.scene(1.))
        query=fixtures.query(memory,fixtures.packet(2.,depth=.8),batch=1)
        self.assertTrue(query['history_used']);self.assertEqual(query['combined_fraction'],1.)
        self.assertEqual(memory.views[0].frame.timestamp,1.)
        self.assertEqual(memory.captures,1)
    def test_manipulation_still_invalidates_boundary_evidence(self):
        fixtures=fixtures_module.VisualMemoryTests();memory=BatchVisualMemory();loop=self.loop(memory)
        packet=fixtures.packet(1.);packet.jaw_gap_m=.04
        loop._remember_verified_boundary(packet,fixtures.scene(1.))
        loop.transition('APPROACH')
        self.assertEqual(memory.views,[])
        self.assertFalse(fixtures.query(memory,fixtures.packet(2.,depth=.8),batch=1)['history_used'])
    def test_diagnostics_distinguish_expiry_from_geometric_mismatch(self):
        fixtures=fixtures_module.VisualMemoryTests()
        memory=fixtures.recorded(max_age_s=2.)
        expired=fixtures.query(memory,fixtures.packet(4.,depth=.8))
        self.assertEqual(expired['history_rejections']['expired'],3)
        changed=fixtures.query(fixtures.recorded(),fixtures.packet(2.,depth=.8),box=fixtures.box(shift=.004))
        self.assertEqual(changed['history_rejections']['changed_rim'],3)
    def test_explicit_capture_event_keeps_original_time_and_physical_view_names(self):
        fixtures=fixtures_module.VisualMemoryTests();events=[];memory=BatchVisualMemory(emit=lambda **row:events.append(row))
        memory.observe(fixtures.packet(1.),fixtures.scene(1.),batch=1,phase='BATCH_READY',stationary=True)
        self.assertEqual(events[-1]['capture_phase'],'BATCH_READY')
        self.assertEqual(events[-1]['timestamp'],1.)
        self.assertEqual(events[-1]['cameras'],['front','left_wrist','right_wrist'])

if __name__=='__main__':unittest.main()
