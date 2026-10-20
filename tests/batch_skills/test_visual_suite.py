import unittest
from collections import deque
from types import SimpleNamespace


def consume_cases(work,results):
    from a3_batch_skills.visual_suite import queued_cases
    results.put(list(queued_cases(work)))


class SharedModelTests(unittest.TestCase):
    def test_workers_started_before_feeder_receive_every_case_exactly_once(self):
        import multiprocessing as mp
        context=mp.get_context('spawn');work=context.Queue();results=context.Queue()
        workers=[context.Process(target=consume_cases,args=(work,results)) for _ in range(2)]
        try:
            for process in workers:process.start()
            for case in range(7):work.put(case)
            for _ in workers:work.put(None)
            ownership=[results.get(timeout=15) for _ in workers]
            self.assertEqual(sorted(x for group in ownership for x in group),list(range(7)))
            for process in workers:process.join(timeout=5);self.assertEqual(process.exitcode,0)
        finally:
            for process in workers:
                if process.is_alive():process.terminate();process.join(timeout=5)

    def test_sessions_share_weights_but_not_action_queues_or_preprocessing_state(self):
        import torch
        from a3_batch_skills.visual_suite import clone_session
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Linear(2,2);self.reset()
            def reset(self):self._action_queue=deque();self._queues={'action':deque()}
        base=SimpleNamespace(policy=Policy(),pre={'rows':[]},post={'rows':[]},steps=9)
        a=clone_session(base);b=clone_session(base)
        self.assertIs(a.policy.model,b.policy.model)
        a.policy._action_queue.append(3);a.pre['rows'].append('a')
        self.assertFalse(b.policy._action_queue);self.assertFalse(base.policy._action_queue)
        self.assertEqual(b.pre['rows'],[]);self.assertEqual(base.pre['rows'],[])


if __name__=='__main__':unittest.main()
