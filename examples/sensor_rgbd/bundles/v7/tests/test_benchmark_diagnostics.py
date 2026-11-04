import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from a3_continuability.benchmark_diagnostics import (
    install_benchmark_diagnostics,native_failure_components,
)


class BenchmarkDiagnosticsTests(unittest.TestCase):
    def test_identity_actions_returns_and_observations_are_preserved(self):
        action=object();observation={'unmodified':object()}
        class Env:
            def step(self,command):
                self.command=command
                self.result=(observation,.25,False,False,self.info)
                return self.result
        env=Env();env.info=dict(success=False,cookies_in_target=10,
            count_fill=True,success_criterion='exact_slots',
            target_slot_occupancy=[1,-1],target_touches_all_walls=True,
            success_hold_count=0,required_success_hold_steps=5)
        original=Env.step
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            recorder=install_benchmark_diagnostics(folder,env_classes=(Env,))
            result=env.step(action)
            self.assertIs(result,env.result)
            self.assertIs(env.command,action);self.assertIs(result[0],observation)
            self.assertIs(result[4],env.info)
            env.info['target_slot_occupancy'][0]=99
            recorder.close();recorder.close()
            saved=json.loads((Path(folder)/'final_benchmark_info.json').read_text())
            self.assertEqual(saved['final_info']['target_slot_occupancy'],[1,-1])
            self.assertIn('unoccupied_target_slot',saved['final_failure_components'])
            self.assertEqual(saved['steps_observed'],1)
            self.assertFalse(saved['policy_receives_diagnostics'])
        self.assertIs(Env.step,original)

    def test_subclass_outermost_info_is_copied_once(self):
        class Base:
            def step(self,action):
                return action,0,False,False,dict(success=False,count_fill=True,
                    success_hold_count=0,required_success_hold_steps=5)
        class Large(Base):
            def step(self,action):
                result=super().step(action)
                result[4].update(target_layout='2x10',target_column_counts=(9,11))
                return result
        original_base=Base.step;original_large=Large.step
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            recorder=install_benchmark_diagnostics(folder,env_classes=(Base,Large))
            Large().step(None);recorder.close()
            saved=json.loads((Path(folder)/'final_benchmark_info.json').read_text())
        self.assertEqual(saved['steps_observed'],1)
        self.assertEqual(saved['final_info']['target_column_counts'],[9,11])
        self.assertIn('target_column_counts',saved['final_failure_components'])
        self.assertIs(Base.step,original_base);self.assertIs(Large.step,original_large)

    def test_native_hold_and_exact_slots_are_distinguished(self):
        base=dict(success=False,required_cookies=20,cookies_in_target=20,
            count_fill=True,target_layout='2x10',target_column_counts=[10,10],
            success_criterion='released_count',target_slot_occupancy=[-1],
            target_touches_all_walls=False,success_hold_count=3,
            required_success_hold_steps=5)
        self.assertEqual(native_failure_components(base),['native_success_hold_not_completed'])
        base['success']=True
        self.assertEqual(native_failure_components(base),[])

    def test_no_info_query_and_exception_propagates_unchanged(self):
        error=ValueError('original error')
        class Env:
            def step(self,action):raise error
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            recorder=install_benchmark_diagnostics(folder,env_classes=(Env,))
            with self.assertRaises(ValueError) as context:Env().step(None)
            self.assertIs(context.exception,error)
            recorder.close()
            saved=json.loads((Path(folder)/'final_benchmark_info.json').read_text())
            self.assertEqual(saved['steps_observed'],0)
            self.assertEqual(saved['final_failure_components'],['no_native_steps_observed'])


if __name__=='__main__':unittest.main()
