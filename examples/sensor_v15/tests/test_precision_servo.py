import unittest
import numpy as np
from a3_continuability.precision_servo import precision_joint_target


class PrecisionServoTests(unittest.TestCase):
    def robot(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint name="a" axis="0 0 1" range="-90 90"/>
          <geom type="capsule" fromto="0 0 0 .1 0 0" size=".005"/>
          <body pos=".1 0 0"><joint name="b" axis="0 0 1" range="-90 90"/>
          <geom type="capsule" fromto="0 0 0 .1 0 0" size=".005"/>
          <site name="tip" pos=".1 0 0"/></body></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        return model,data

    def test_submillimetre_goal_does_not_stop_at_coarse_gradient_tolerance(self):
        import mujoco
        model,data=self.robot();target=data.site_xpos[0].copy()+[0,.0003,0]
        q=precision_joint_target(model,data,[0,1],0,model.jnt_range,np.zeros(2),target,np.array([1.,0,0,0]))
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        # Keep the joint regularizer; its few-micrometre bias is acceptable,
        # while stopping at the original 300 micrometres is not.
        self.assertLess(np.linalg.norm(data.site_xpos[0]-target),1e-5)
        self.assertGreater(np.max(abs(q)),.001)

    def test_larger_goal_cannot_create_a_joint_jump(self):
        model,data=self.robot();target=data.site_xpos[0].copy()+[0,.08,0]
        q=precision_joint_target(model,data,[0,1],0,model.jnt_range,np.zeros(2),target,np.array([1.,0,0,0]))
        self.assertTrue(np.isfinite(q).all());self.assertLessEqual(np.max(abs(q)),.025+1e-10)
