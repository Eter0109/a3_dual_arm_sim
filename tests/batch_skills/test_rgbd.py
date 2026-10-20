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

    def target_detector(self,offset=(0.,0.),angle=0.,length=.050,height=.031):
        # Five physical top ridges, deliberately unaligned with the slot x's.
        x,y=np.meshgrid(np.linspace(-length/2,length/2,85),np.linspace(-.0014,.0014,9))
        base=np.c_[x.ravel(),y.ravel()]
        positions=np.array([[offset[0],offset[1]+i*.006333] for i in range(-2,3)])
        q=np.concatenate([np.c_[base@rotation_z(angle)[:2,:2].T+p,np.full(len(base),height)] for p in positions])
        box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.057,.045]),.026,4,0.)
        spec=dict(schema='a3_rgbd_geometry_v2',cookie_half_size=[.025,.0031665,.0125],
                  wall_thickness_m=.006,floor_offset_m=.006)
        localizer=RGBDLocalizer(spec)
        # Test the same RGB/height filter used on real packets, not just PCA.
        colors=np.tile(np.array([210,140,25],dtype=np.uint8),(1,len(q),1))
        class Packet:rgb=colors
        result=localizer._cookies(Packet(),q[None],box,source=False)
        return result,positions

    def test_target_detection_does_not_require_slot_alignment(self):
        for angle in (0.,.15):
            result,positions=self.target_detector(offset=(-.008,.009),angle=angle)
            self.assertEqual(len(result),5)
            actual=np.array([c.position[:2] for c in result])
            np.testing.assert_allclose(actual,positions,atol=.0007)

    def test_partial_and_stacked_ridges_do_not_count_as_placed_cookies(self):
        self.assertEqual(len(self.target_detector(length=.025)[0]),0)
        self.assertEqual(len(self.target_detector(height=.056)[0]),0)

    def test_sparse_disconnected_target_surfaces_are_measured_without_slot_seeds(self):
        spec=dict(schema='a3_rgbd_geometry_v2',cookie_half_size=[.025,.0031665,.0125],wall_thickness_m=.006)
        loc=RGBDLocalizer(spec)
        box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.057,.045]),.026,4,0.)
        # Alternating sampling rows split a surface into short components.
        patches=[]
        for y in np.arange(-2,3)*.006333:
            for j,x in enumerate(np.linspace(-.025,.025,65)):
                ys=np.linspace(-.002,.002,7)+(j%2)*.00025
                patches.append(np.c_[np.full(7,x-.008),ys+y+.009,np.full(7,.031)])
        result=loc._free_targets(np.concatenate(patches),box)
        self.assertEqual(len(result),5)
        np.testing.assert_allclose([c.position[0] for c in result],-.008,atol=.001)

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
