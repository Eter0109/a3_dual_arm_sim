import math
import unittest

from a3_continuability.batch_vision import LocalizationError
from a3_continuability.release_planning import (
    ReleasePlanningError, plan_narrow_release, required_detach_height,
)


class ReleasePlanningTests(unittest.TestCase):
    def test_thin_pad_real_gap_offset_is_preserved(self):
        # Public 2F-85 geometry with 0.5mm half-thickness pads: the jaw gap
        # intercept is 5.3616mm, so opening*.085 is not a physical gap.
        width = 5 * (.019 / 3)
        offset = .0053616
        opening = (width-offset) / .085
        plan = plan_narrow_release(opening, width)
        self.assertAlmostEqual(plan.gap_offset_m, offset)
        self.assertAlmostEqual(plan.target_gap_m, width+.002)
        self.assertAlmostEqual(offset+.085*plan.opening, width+.002)
        self.assertAlmostEqual(plan.opening_increase, .002/.085)

    def test_minimum_release_avoids_known_adjacent_row_intrusion(self):
        # Adjacent batch has 8mm net free space. At the old fixed .49 opening,
        # each outside pad edge extends 8.6724667mm from a compressed five.
        width = 5 * (.019 / 3)
        offset, full_pad_thickness, neighboring_clearance = .0053616, .001, .008
        old_gap = offset + .085*.49
        old_extension = (old_gap-width)/2 + full_pad_thickness
        self.assertAlmostEqual(old_extension-neighboring_clearance, .0006724666666667)
        plan = plan_narrow_release((width-offset)/.085, width)
        new_extension = (plan.target_gap_m-width)/2 + full_pad_thickness
        self.assertAlmostEqual(new_extension, .002)
        self.assertLess(new_extension, neighboring_clearance)

    def test_measured_gap_not_a_fixed_cookie_count_controls_target(self):
        plan = plan_narrow_release(.4, .047, clearance_m=.003)
        self.assertAlmostEqual(plan.target_gap_m, .050)
        self.assertAlmostEqual(plan.opening, .4+.003/.085)

    def test_infeasible_cap_is_rejected_instead_of_clipped(self):
        for opening in [.48, .49, .8]:
            with self.subTest(opening=opening), self.assertRaises(ReleasePlanningError):
                plan_narrow_release(opening, .04)

    def test_exact_opening_limit_is_feasible(self):
        plan = plan_narrow_release(.49-.002/.085, .04)
        self.assertAlmostEqual(plan.opening, .49)
        self.assertAlmostEqual(plan.target_gap_m, .042)

    def test_invalid_release_measurements_are_rejected(self):
        cases = [(None,.03), (math.nan,.03), (math.inf,.03), (-.01,.03),
                 (1.01,.03), (.3,None), (.3,math.nan), (.3,-.001), (True,.03)]
        for opening, gap in cases:
            with self.subTest(opening=opening,gap=gap), self.assertRaises(ReleasePlanningError):
                plan_narrow_release(opening,gap)
        for keyword,value in [('gap_scale_m',0),('gap_scale_m',-1),('clearance_m',0),
                              ('clearance_m',-1),('max_opening',0),('max_opening',1.1),
                              ('clearance_m',math.inf)]:
            with self.subTest(keyword=keyword,value=value), self.assertRaises(ReleasePlanningError):
                plan_narrow_release(.3,.03,**{keyword:value})

    def test_already_clear_detach_needs_no_extra_travel(self):
        self.assertEqual(required_detach_height(.80,.78),0.)

    def test_detach_uses_lowest_pad_and_highest_protected_geometry(self):
        # The actual bottom pad, rather than tool center, must clear both old
        # cookies and box rims. The 2mm margin survives into the target.
        self.assertAlmostEqual(required_detach_height(.761,.790), .031)
        self.assertAlmostEqual(.761+required_detach_height(.761,.790), .792)

    def test_unreachable_detachment_is_rejected(self):
        with self.assertRaises(ReleasePlanningError):
            required_detach_height(.70,.85,max_detach_m=.127)
        self.assertAlmostEqual(required_detach_height(.70,.825,max_detach_m=.127),.127)

    def test_invalid_detach_evidence_is_rejected(self):
        for a,b in [(None,.79),(.76,None),(math.nan,.79),(.76,math.inf)]:
            with self.subTest(a=a,b=b), self.assertRaises(ReleasePlanningError):
                required_detach_height(a,b)
        for kwargs in [{'margin_m':-.001},{'max_detach_m':0},{'max_detach_m':math.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ReleasePlanningError):
                required_detach_height(.76,.79,**kwargs)

    def test_failure_uses_existing_localization_error_boundary(self):
        self.assertTrue(issubclass(ReleasePlanningError, LocalizationError))


if __name__ == '__main__':
    unittest.main()
