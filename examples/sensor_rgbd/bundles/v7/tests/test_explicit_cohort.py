import unittest
from a3_continuability.native_grid_reporting import cases,select_cases


class ExplicitCohortTest(unittest.TestCase):
    def test_selection_preserves_requested_order_and_parent_unchanged(self):
        plan=cases(24000,2)
        selected=select_cases(plan,[plan[-1],plan[0]])
        self.assertEqual(selected,[plan[-1],plan[0]])
        selected[0]['seed']=1
        self.assertEqual(plan[-1]['seed'],24001)

    def test_duplicate_or_out_of_grid_cannot_change_denominator(self):
        plan=cases(24000,1)
        for requested in [[],[plan[0],plan[0]],[dict(plan[0],seed=24001)],
                          [dict(plan[0],seed=True)],[dict(plan[0],score=10)]]:
            with self.subTest(requested=requested),self.assertRaises(ValueError):
                select_cases(plan,requested)
