import unittest
import numpy as np
from a3_continuability.plane_calibration import fit_table_correction


class PlaneCalibrationTests(unittest.TestCase):
    def test_measured_plane_is_levelled_with_outliers_without_xy_pose_claim(self):
        rng=np.random.default_rng(12);xy=rng.uniform([-.08,-.1],[.5,.55],(2400,2))
        z=.752+.02*xy[:,0]-.01*xy[:,1]+rng.normal(0,.00008,len(xy))
        points=np.c_[xy,z];points[:300,2]+=rng.uniform(.012,.04,300)
        cp=np.array([.2,-.2,1.05]);correction=fit_table_correction(points,cp)
        clean=points[300:]@correction.rotation.T+correction.translation
        self.assertLess(np.sqrt(np.mean((clean[:,2]-.75)**2)),.00015)
        shifted=correction.rotation@cp+correction.translation
        np.testing.assert_allclose(shifted[:2],cp[:2],atol=1e-10)
        self.assertGreater(correction.inlier_fraction,.8)

    def test_missing_or_small_plane_cannot_create_calibration(self):
        for points in (np.zeros((2,3)),np.full((500,3),np.nan)):
            with self.assertRaises(ValueError):fit_table_correction(points,[0,0,1])

    def test_visible_cookie_top_is_not_accepted_as_table(self):
        rng=np.random.default_rng(3);points=np.c_[rng.uniform([0,0],[.3,.4],(500,2)),np.full(500,.80)]
        with self.assertRaises(ValueError):fit_table_correction(points,[0,0,1])


if __name__=='__main__':unittest.main()
