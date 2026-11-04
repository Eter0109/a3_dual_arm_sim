import io,json,unittest
from contextlib import redirect_stdout
import numpy as np
from a3_continuability.sensor_boundary import numeric_json

class SensorLoggingTests(unittest.TestCase):
    def test_numpy_diagnostics_do_not_abort_physical_control(self):
        event=dict(adjusted=np.bool_(True),error=np.float64(.001),xyz=np.array([1.,2.,3.]))
        row=json.loads(json.dumps(event,default=numeric_json))
        self.assertIs(row['adjusted'],True);self.assertEqual(row['xyz'],[1.,2.,3.])

    def test_unknown_objects_are_not_silently_stringified(self):
        with self.assertRaises(TypeError):json.dumps(object(),default=numeric_json)
