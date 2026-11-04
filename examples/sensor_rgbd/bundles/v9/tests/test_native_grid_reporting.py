import unittest
from a3_continuability.native_grid_reporting import cases,key,summarize


class NativeGridReportingTests(unittest.TestCase):
    def test_focused_grid_only_contains_requested_two_cells(self):
        plan=cases(14000,20,tasks=['2x5'],profiles=['medium','advanced'])
        self.assertEqual(len(plan),40)
        self.assertEqual(len({key(c) for c in plan}),40)
        self.assertEqual(len(summarize([],20,tasks=['2x5'],profiles=['medium','advanced'])),2)
        with self.assertRaises(ValueError):cases(tasks=['2x5'],profiles=['medium','medium'])
    def test_grid_has_six_cells_and_paired_seeds(self):
        plan=cases(13000,20)
        self.assertEqual(len(plan),120)
        self.assertEqual(len({key(c) for c in plan}),120)
        self.assertTrue(all(sum(c['seed']==s for c in plan)==6 for s in range(13000,13020)))

    def test_early_native_pass_does_not_mean_release_verified(self):
        row=dict(task='2x5',profile='basic',cookies_in_target=10,success=True,controller_phase='INSERT')
        result=summarize([row],20)[0]
        self.assertEqual(result['native_successes'],1)
        self.assertEqual(result['release_verified_successes'],0)

    def test_infrastructure_error_is_not_reported_as_zero_cookie_episode(self):
        row=dict(task='2x5',profile='basic',infrastructure_error='EGL failure')
        result=summarize([row],20)[0]
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['infrastructure_errors'],1)
        self.assertEqual(result['valid_episodes'],0)
        self.assertIsNone(result['mean_final_count'])

    def test_final_count_not_peak_used_for_aggregate(self):
        row=dict(task='2x10',profile='advanced',cookies_in_target=5,max_count=15,
                 success=False,controller_phase='FAILED',controller_failure='previous batch damaged')
        result=summarize([row],20)[5]
        self.assertEqual(result['mean_final_count'],5)
        self.assertEqual(result['failure_reasons'],{'previous batch damaged':1})
