import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_listen_v8_audio_governor import Governor,dimensions


def node(t,cpu=80,free=80000,io=0):
    return dict(unix=t,cpu_busy_percent=cpu,iowait_percent=io,
        gpus=[dict(total_mib=140000,memory_mib=140000-free)],
        memory=dict(MemAvailable=500,MemTotal=1000))


class GovernorTests(unittest.TestCase):
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
