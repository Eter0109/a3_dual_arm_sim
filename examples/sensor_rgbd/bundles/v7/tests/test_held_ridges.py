import unittest
import numpy as np
from a3_continuability.held_batch_vision import ridge_centers
from a3_continuability.batch_vision import LocalizationError

class RidgeTests(unittest.TestCase):
    def cloud(self,n):
        x,y=np.meshgrid(np.linspace(-.017,.017,30),np.linspace(-.018,.018,250))
        centers=(np.arange(n)-(n-1)/2)*.0063333
        distance=np.min(abs(y[:,:,None]-centers),axis=2)
        z=.0125-np.maximum(0,distance-.00065)
        keep=distance<.0031
        return np.c_[x[keep],y[keep],z[keep]]

    def test_five_geometric_ridges_without_any_color_information(self):
        np.testing.assert_allclose(ridge_centers(self.cloud(5)),
            (np.arange(5)-2)*.0063333,atol=.0005)

    def test_four_objects_or_flat_top_cannot_be_called_five(self):
        with self.assertRaises(LocalizationError):ridge_centers(self.cloud(4))
        cloud=self.cloud(5);cloud[:,2]=.0125
        with self.assertRaises(LocalizationError):ridge_centers(cloud)
