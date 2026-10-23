import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_local_health as health


class HealthTests(unittest.TestCase):
    def test_transient_timeout_retains_bounded_explicitly_stale_evidence(self):
        clock=Mock(side_effect=[100,120,130])
        sample=health.Sample(Mock(side_effect=[(2,{'gpus':[{'memory_mib':10}]}),subprocess.TimeoutExpired('nvidia-smi',15),(3,{'gpus':[{'memory_mib':20}]})]),clock)
        a,m=sample(1);self.assertFalse(m['telemetry_stale'])
        a,m=sample(2);self.assertEqual(a,2);self.assertTrue(m['telemetry_stale']);self.assertEqual(m['telemetry_age_seconds'],20)
        a,m=sample(2);self.assertEqual(a,3);self.assertFalse(m['telemetry_stale']);self.assertEqual(m['gpus'][0]['memory_mib'],20)

    def test_cached_metrics_expire_instead_of_claiming_health_forever(self):
        sample=health.Sample(Mock(side_effect=[(2,{}),subprocess.TimeoutExpired('nvidia-smi',15)]),Mock(side_effect=[100,161]))
        sample(1)
        with self.assertRaisesRegex(RuntimeError,'beyond grace'):sample(2)

    def test_startup_retries_are_bounded_and_never_invent_metrics(self):
        original=Mock(side_effect=subprocess.TimeoutExpired('nvidia-smi',15));sleep=Mock()
        with self.assertRaises(RuntimeError):health.Sample(original,clock=lambda:0,sleep=sleep)(1)
        self.assertEqual(original.call_count,3);self.assertEqual(sleep.call_count,2)

    def test_initial_timeout_can_recover(self):
        original=Mock(side_effect=[subprocess.TimeoutExpired('nvidia-smi',15),(2,{'gpus':[]})])
        self.assertFalse(health.Sample(original,clock=lambda:1,sleep=lambda _:None)(1)[1]['telemetry_stale'])

    def test_programming_errors_not_hidden(self):
        with self.assertRaises(KeyError):health.Sample(Mock(side_effect=KeyError('bug')))(1)

    def test_recovery_does_not_resume_manual_stops_or_quality_failures(self):
        self.assertTrue(health.recoverable(dict(state='stopping_after_error',error='node4 stale heartbeat beyond grace period')))
        self.assertFalse(health.recoverable(dict(state='stopped',error='stale heartbeat')))
        self.assertFalse(health.recoverable(dict(state='stopping_after_error',error='Ten consecutive processing errors')))

    def test_recovery_limits_and_resource_reserve(self):
        self.assertFalse(health.allowed([1,2,3],10));self.assertTrue(health.allowed([1,2,3],2000))
        self.assertFalse(health.allowed([1,2,3,4,5,6],10000))
        self.assertTrue(health.enough([dict(free_gpu_mib=28000,memory_fraction=.8)]*3))
        self.assertFalse(health.enough([dict(free_gpu_mib=20000,memory_fraction=.8)]*3))
        self.assertFalse(health.enough([dict(free_gpu_mib=28000,memory_fraction=.8)]*2))


if __name__=='__main__':unittest.main()
