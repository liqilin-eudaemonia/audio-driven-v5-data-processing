import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import listen_j_annotation_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def test_explicit_three_gpu_scope_and_default_eight(self):
        self.assertEqual(supervisor.selected_gpu_ids({}),list(range(8)))
        self.assertEqual(supervisor.selected_gpu_ids({'gpu_ids':[0,1,2]}),[0,1,2])
        for ids in ([],[0,0],[8],[True],[-1]):
            with self.assertRaises(ValueError):supervisor.selected_gpu_ids({'gpu_ids':ids})

    def test_host_guard_requires_exact_explicit_destination(self):
        supervisor.validate_host({},'dedicated-developjob-lq-dev-machine-2-taaft')
        supervisor.validate_host({'expected_hostname':'node8'},'node8')
        for config,host in [({},'node8'),({'expected_hostname':'node8'},'node17'),({'expected_hostname':''},'')]:
            with self.assertRaises(ValueError):supervisor.validate_host(config,host)

    def test_repeat_selection_uses_combined_time_not_lucky_fastest(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);benchmark=work/'bench';summary={}
            for name,tokens,rate in [('a1',16384,.45),('a2',16384,.35),('b1',32768,.44),('b2',32768,.44)]:
                folder=benchmark/name/'results';folder.mkdir(parents=True)
                for i in range(50):
                    supervisor.write(folder/f'{i}.json',{'records':[{'view':'full','status':'valid'}]})
                summary[name]={'exit_code':0,'samples':50,'truncated_requests':0,'invalid_requests':0,
                    'clips_per_s':rate,'settings':{'batch':512,'tokens':tokens,'frames':16384,'clips':50,'memory':.85}}
            supervisor.write(benchmark/'summary.json',summary)
            result=supervisor.choose(work,benchmark)
            self.assertEqual(result['tokens'],32768)
            self.assertAlmostEqual(result['measured_clips_per_s'],.44)
            self.assertEqual(result['repeat_arms'],['b1','b2'])

    def test_identity_mismatch_is_not_alive(self):
        import os
        self.assertFalse(supervisor.alive({'phase':'annotation'}))
        self.assertFalse(supervisor.alive({'pid':0,'start_ticks':None}))
        self.assertTrue(supervisor.alive({'pid':os.getpid(),'start_ticks':supervisor.identity(os.getpid())}))
        self.assertFalse(supervisor.alive({'pid':os.getpid(),'start_ticks':'invalid'}))

    def test_rollout_rejects_old_heartbeat_until_live_new_worker_produces(self):
        import os
        from unittest.mock import patch
        selected={'batch_size':512,'tokens':32768,'frames':16384,'memory':.85}
        child={'pid':12345,'start_ticks':'expected'}
        h={'pid':12345,'revision':3,'execution':selected,'completed_clips':50}
        with patch.object(supervisor,'alive',return_value=False):
            self.assertFalse(supervisor.warm_ready(h,child,selected,3))
        with patch.object(supervisor,'alive',return_value=True):
            self.assertFalse(supervisor.warm_ready(dict(h,revision=2),child,selected,3))
            self.assertFalse(supervisor.warm_ready(dict(h,pid=12346),child,selected,3))
            self.assertFalse(supervisor.warm_ready(dict(h,completed_clips=0),child,selected,3))
            self.assertTrue(supervisor.warm_ready(h,child,selected,3))


if __name__=='__main__':unittest.main()
