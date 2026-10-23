import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_listen_v8_audio_governor import Governor,dimensions,ResilientSnapshot,validate_heartbeats


def node(t,cpu=80,free=80000,io=0):
    return dict(unix=t,cpu_busy_percent=cpu,iowait_percent=io,
        gpus=[dict(total_mib=140000,memory_mib=140000-free)],
        memory=dict(MemAvailable=500,MemTotal=1000))


class GovernorTests(unittest.TestCase):
    def test_sixteen_gib_free_does_not_drop_unchanged_gpu_processes(self):
        g=Governor()
        desired,reasons=g.consider('n',node(105,free=16000),dimensions(4,2),0,105)
        self.assertIsNone(desired)
        self.assertEqual(reasons,[])

    def test_transient_missing_node_keeps_recent_heartbeat(self):
        r=ResilientSnapshot(['a','b'])
        r.refresh_nodes(lambda name:dict(unix=100,state='running'))
        def missing(name):
            if name=='b':raise FileNotFoundError(name)
            return dict(unix=110,state='running')
        values=r.refresh_nodes(missing)
        self.assertEqual(set(values),{'a','b'})
        self.assertTrue(validate_heartbeats(values,['a','b'],110,0))

    def test_genuine_stale_node_still_stops(self):
        with self.assertRaisesRegex(RuntimeError,'stale heartbeat'):
            validate_heartbeats({'a':dict(unix=100,state='running')},['a'],230,0)

    def test_finished_node_need_not_publish_while_other_node_drains(self):
        nodes={'a':dict(unix=10,state='finished'),'b':dict(unix=200,state='running')}
        self.assertTrue(validate_heartbeats(nodes,['a','b'],210,0))

    def test_initial_missing_node_has_bounded_grace(self):
        self.assertFalse(validate_heartbeats({},['a'],100,0))
        with self.assertRaisesRegex(RuntimeError,'Missing node heartbeat'):
            validate_heartbeats({},['a'],130,0)

    def test_single_spike_does_not_drop_four_workers(self):
        g=Governor()
        for t in range(0,100,10):
            desired,_=g.consider('n',node(t,95 if t==50 else 76),dimensions(4,1),t,t)
            self.assertIsNone(desired)

    def test_sustained_pressure_reduces_prep_first(self):
        g=Governor();desired=None
        for t in range(0,70,10):
            value,_=g.consider('n',node(t,94),dimensions(4,2),t,t)
            if value:desired=value
        self.assertEqual(desired,dimensions(4,1))

    def test_sustained_pressure_steps_four_to_two(self):
        g=Governor();desired=None
        for t in range(0,70,10):
            value,_=g.consider('n',node(t,94),dimensions(4,1),t,t)
            if value:desired=value
        self.assertEqual(desired,dimensions(2,1))

    def test_memory_pressure_has_no_warmup_exemption(self):
        g=Governor();g.changed['n']=100
        desired,reasons=g.consider('n',node(105,free=10000),dimensions(4,1),0,105)
        self.assertEqual(desired,dimensions(1,1));self.assertIn('gpu_memory',reasons)

    def test_duplicate_telemetry_is_not_sustained_evidence(self):
        g=Governor()
        for t in range(0,100,10):
            desired,_=g.consider('n',node(0,99),dimensions(4,1),t,t)
            self.assertIsNone(desired)

    def test_stable_headroom_recovers_to_two(self):
        g=Governor();g.changed['n']=0
        desired=None
        for t in range(0,190,10):
            value,_=g.consider('n',node(t,65),dimensions(1,1),t,t)
            if value:desired=value
        self.assertEqual(desired,dimensions(2,1))

    def test_failed_prefetch_trial_rolls_back(self):
        g=Governor();g.changed['n']=0
        g.evaluations['n']=dict(setting=[4,2],started=0,baseline=0,time_baseline=0,previous_rate=60.)
        desired,reasons=g.consider('n',node(250,75),dimensions(4,2),200,250)
        self.assertEqual(desired,dimensions(4,1));self.assertIn('prefetch_trial_no_observed_gain',reasons)
        self.assertIn('n',g.prep_trials_disabled)


if __name__=='__main__':unittest.main()
