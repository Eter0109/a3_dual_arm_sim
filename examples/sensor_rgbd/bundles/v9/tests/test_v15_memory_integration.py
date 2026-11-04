import os,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from a3_batch_skills.rgbd_localization import LocalizationFailure
from a3_continuability.native_sensor_policy import NativeSensorPolicy
from a3_continuability.sensor_batch_loop import SensorBatchLoop

class IntegrationTests(unittest.TestCase):
    def loop(self,memory):
        loop=SensorBatchLoop.__new__(SensorBatchLoop)
        loop.visual_memory=memory;loop.spec={};loop.completed=[];loop.memory_goal_accepts=0;loop.emit=Mock()
        return loop
    def test_current_success_never_needs_or_queries_history(self):
        memory=Mock();loop=self.loop(memory)
        packet=SimpleNamespace(rgbd=object());scene=object();goal=object()
        with patch('a3_continuability.sensor_batch_loop.make_visual_goal',return_value=goal):
            self.assertIs(loop._make_goal(packet,scene,0,1),goal)
        memory.make_goal.assert_not_called()
    def test_missing_current_evidence_only_falls_back_when_enabled(self):
        packet=SimpleNamespace(rgbd=object());scene=object();goal=object()
        with patch('a3_continuability.sensor_batch_loop.make_visual_goal',side_effect=LocalizationFailure('occluded')):
            with self.assertRaises(LocalizationFailure):self.loop(None)._make_goal(packet,scene,0,1)
            memory=Mock();memory.make_goal.return_value=goal
            self.assertIs(self.loop(memory)._make_goal(packet,scene,0,1),goal)
            memory.make_goal.assert_called_once_with(packet,scene,{},0,column=1,batch=0)
    def test_implementation_errors_do_not_become_history_fallback(self):
        memory=Mock()
        with patch('a3_continuability.sensor_batch_loop.make_visual_goal',side_effect=ValueError('invalid input')):
            with self.assertRaises(ValueError):self.loop(memory)._make_goal(SimpleNamespace(rgbd=None),None,0,1)
        memory.make_goal.assert_not_called()
    def test_moving_and_release_views_do_not_capture_memory(self):
        memory=Mock();loop=self.loop(memory);loop.memory_settling=SimpleNamespace(count=6)
        packet=object();scene=object()
        for phase,after in [('INSERT','LOCATE'),('VERIFY','LOCATE'),('VIEW','VERIFY')]:
            loop.phase=phase;loop.after_view=after;loop._remember_view(packet,scene)
        loop.phase='VIEW';loop.after_view='LOCATE';loop.memory_settling.count=2
        loop._remember_view(packet,scene);memory.observe.assert_not_called()
        loop.memory_settling.count=3;loop._remember_view(packet,scene)
        memory.observe.assert_called_once_with(packet,scene,batch=0,phase='VIEW',stationary=True)
    def test_memory_mode_rejects_accidental_right_arm_observation(self):
        policy=NativeSensorPolicy()
        policy.sensor=SimpleNamespace(protocol={})
        with patch.dict(os.environ,{'A3_SENSOR_ACTIVE_RIGHT':'1','A3_SENSOR_VISUAL_MEMORY':'1'}):
            with self.assertRaisesRegex(ValueError,'fixed right arm'):policy.reset()
    def test_passive_view_capture_cannot_mutate_the_v15_planning_scene(self):
        memory=Mock();loop=self.loop(memory);old_scene=object();observed=object()
        loop.phase='VIEW';loop.after_view='LOCATE';loop.memory_settling=SimpleNamespace(count=3)
        loop.scene=old_scene;loop.visual_error='preserved diagnostic'
        loop.localizer=SimpleNamespace(observe=Mock(return_value=observed))
        packet=SimpleNamespace(rgbd=object())
        loop._remember_view(packet)
        self.assertIs(loop.scene,old_scene)
        self.assertEqual(loop.visual_error,'preserved diagnostic')
        memory.observe.assert_called_once_with(packet,observed,batch=0,phase='VIEW',stationary=True)

if __name__=='__main__':unittest.main()
