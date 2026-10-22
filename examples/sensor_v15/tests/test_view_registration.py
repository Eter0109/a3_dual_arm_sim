import unittest
from types import SimpleNamespace
import numpy as np
from a3_continuability.view_registration import box_alignment


def box(position,yaw=0,sides=4):
    c,s=np.cos(yaw),np.sin(yaw)
    return SimpleNamespace(position=np.array(position,dtype=float),
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]),supported_sides=sides)


class ViewRegistrationTests(unittest.TestCase):
    def test_observed_bin_alignment_corrects_relative_xy_without_moving_table_z(self):
        ref=box([.17,.31,0],.015);other=box([.173,.307,0],.023)
        rot,shift=box_alignment(ref,other)
        np.testing.assert_allclose(rot@other.position+shift,ref.position,atol=1e-12)
        np.testing.assert_allclose(rot@other.rotation,ref.rotation,atol=1e-12)
        self.assertEqual(shift[2],0)

    def test_underconstrained_or_large_alignment_is_rejected(self):
        ref=box([.17,.31,0])
        for other in (box([.17,.31,0],sides=2),box([.27,.31,0]),box([.17,.31,0],.15)):
            with self.assertRaises(ValueError):box_alignment(ref,other)
