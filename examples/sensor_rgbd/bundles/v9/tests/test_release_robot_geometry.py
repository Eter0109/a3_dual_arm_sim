"""Exercise actual public robot FK with the deployment left-arm 8D packet."""
import unittest
import numpy as np
from a3_continuability.sensor_batch_loop import EncoderMotion
from a3_continuability.three_camera_sensor import NominalRobotCalibration


class ReleaseRobotGeometryTests(unittest.TestCase):
    def test_left8_and_legacy16_give_same_finite_actual_pad_bounds(self):
        from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
        env=make_cookie_benchmark('2x10',source_column='1',first_grasp=5,max_steps=3000).create_env(render_cameras=False)
        try:
            state=np.asarray(env.DEPLOYMENT_HOME,dtype=float).copy();state[7]=.34
            motion=EncoderMotion(env.model,env.config,state)
            sensor_fk=NominalRobotCalibration(env)
            _,pad,gap=sensor_fk.update(state)
            a=motion.pad_height_bounds(state[:8]);b=motion.pad_height_bounds(state)
            np.testing.assert_allclose(a,b,atol=1e-12)
            self.assertTrue(np.isfinite(a).all())
            self.assertLess(a[0],pad[2]);self.assertGreater(a[1],pad[2])
            self.assertGreater(gap,.03)
            with self.assertRaises(RuntimeError):motion.pad_height_bounds(np.zeros(7))
        finally:env.close()


if __name__=='__main__':unittest.main()
