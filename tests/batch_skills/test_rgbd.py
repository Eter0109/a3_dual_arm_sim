import unittest
import numpy as np
from a3_batch_skills.rgbd import RGBDFrame,unproject
from a3_batch_skills.rgbd_localization import (RGBDLocalizer,LocalizationFailure,
    BoxEstimate,CookieEstimate,VisualScene,minimum_rectangle,rotation_z,select_visible_five)


class RGBDTests(unittest.TestCase):
    def test_original_and_exported_camera_keys_match_before_replay_collation(self):
        from a3_batch_skills.train_visual import PolicyRows
        from a3_batch_skills.export import DATASET_CAMERAS
        def row(names):
            result={'observation.state':np.zeros(8),'action':np.zeros((50,8)),
                    'action_is_pad':np.zeros(50,dtype=bool),'task':'transfer five'}
            result.update({'observation.images.'+name:np.zeros((3,16,16)) for name in names})
            dataset=object.__new__(PolicyRows);dataset.dataset=[result]
            return dataset[0]
        original=row(DATASET_CAMERAS.keys());exported=row(DATASET_CAMERAS.values())
        self.assertEqual(set(original),set(exported))
        self.assertIn('observation.images.base_0_rgb',original)

    def test_optical_pixel_centres_and_extrinsic_axes(self):
        k=np.array([[2.,0,1],[0,2.,1],[0,0,1]])
        t=np.eye(4);t[:3,:3]=rotation_z(.4);t[:3,3]=[.1,.2,.3]
        f=RGBDFrame(np.zeros((2,2,3),dtype=np.uint8),np.ones((2,2)),k,t,0.)
        expected=np.array([-.25,-.25,1.])@t[:3,:3].T+t[:3,3]
        np.testing.assert_allclose(unproject(f)[0,0],expected)

    def test_invalid_calibration_is_rejected(self):
        t=np.eye(4);t[0,0]=-1
        with self.assertRaises(ValueError):
            RGBDFrame(np.zeros((2,2,3),np.uint8),np.ones((2,2)),np.eye(3),t,0)

    def test_skew_intrinsics_roundtrip_and_missing_depth(self):
        k=np.array([[2.,.4,1.],[0.,3.,1.],[0.,0.,1.]])
        depth=np.ones((2,2));depth[0,0]=0
        frame=RGBDFrame(np.zeros((2,2,3),np.uint8),depth,k,np.eye(4),np.array(0.))
        cloud=unproject(frame)
        self.assertTrue(np.isnan(cloud[0,0]).all())
        projected=k@cloud[1,1]
        np.testing.assert_allclose(projected[:2]/projected[2],[1.5,1.5])
        with self.assertRaises(ValueError):frame.depth_m[1,1]=2

    def test_legacy_misnamed_geometry_is_rejected(self):
        with self.assertRaises(ValueError):RGBDLocalizer({"cookie_half_size":[.025,.003,.0125]})

    def test_rotated_rectangle_recovers_camera_independent_coordinates(self):
        rot=rotation_z(.17)[:2,:2]
        xy=np.array([[-.06,-.03],[-.06,.03],[.06,.03],[.06,-.03]])@rot.T+[.1,.2]
        r,lo,hi=minimum_rectangle(xy)
        np.testing.assert_allclose((lo+hi)/2@r.T,[.1,.2],atol=1e-9)
        np.testing.assert_allclose(hi-lo,[.12,.06],atol=1e-9)

    def scene(self,ys):
        box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.1,.1]),.02,4,0)
        cookies=tuple(CookieEstimate(np.array([0,y,.01]),np.eye(3),2,1.,0.) for y in ys)
        return VisualScene(box,box,cookies,(),0.,0.)

    def test_select_five_does_not_bridge_missing_or_occluded_rows(self):
        with self.assertRaises(LocalizationFailure):
            select_visible_five(self.scene([0,.008,.016,.04,.048]))
        goal=select_visible_five(self.scene(np.arange(6)*.008))
        self.assertEqual(len(goal),5)
        with self.assertRaises(LocalizationFailure):
            select_visible_five(self.scene(np.arange(6)*.008),column=1)


if __name__=="__main__":unittest.main()
