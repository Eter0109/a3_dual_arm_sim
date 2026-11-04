import unittest
from types import SimpleNamespace as NS
import numpy as np
from a3_continuability.task_completion import EncoderSettling,observed_roi,view_completion_ready,insertion_release_ready


class TaskCompletionTests(unittest.TestCase):
    def test_cached_packet_cannot_build_stability(self):
        s=EncoderSettling();q=np.zeros(8)
        s.update(q,0.)
        for _ in range(20):s.update(q,0.)
        self.assertEqual(s.count,0)
        for i in range(1,7):s.update(q,i*.03)
        self.assertEqual(s.count,6)
        q[2]=.01;s.update(q,.3);self.assertEqual(s.count,0)

    def frame(self):
        t=np.eye(4);t[:3,:3]=np.diag([1,-1,-1]);t[:3,3]=[.1,.2,1.06]
        return NS(world_from_optical=t,intrinsic=np.array([[500.,0,256],[0,500,256],[0,0,1]]),depth_m=np.full((512,512),.3))

    def test_actual_coverage_needs_valid_current_depth_and_inside_image(self):
        f=self.frame();points=np.array([[.1,.2,.76],[.08,.2,.76],[.12,.2,.76]])
        self.assertTrue(observed_roi(f,points))
        f.depth_m[:]=np.nan;self.assertFalse(observed_roi(f,points))
        self.assertFalse(observed_roi(self.frame(),points+[1,0,0]))

    def test_settled_view_cannot_finish_while_low_closed_or_unseen(self):
        values=dict(age=35,stationary_frames=6,pad_z=.94,rim_z=.776,jaw_gap=.04,roi_visible=True)
        self.assertTrue(view_completion_ready(**values))
        for key,value in [('age',10),('stationary_frames',3),('pad_z',.80),('jaw_gap',.02),('roi_visible',False)]:
            self.assertFalse(view_completion_ready(**dict(values,**{key:value})))

    def test_release_requires_current_and_all_previous_progress_plus_stability(self):
        values=dict(age=70,stationary_frames=6,visual_frames=3,position_error=.004,
            rotation_error=.04,current_complete=True,previous_complete=True)
        self.assertTrue(insertion_release_ready(**values))
        for key,value in [('current_complete',False),('previous_complete',False),('visual_frames',2),
                          ('stationary_frames',2),('position_error',.01),('rotation_error',.10)]:
            self.assertFalse(insertion_release_ready(**dict(values,**{key:value})))
