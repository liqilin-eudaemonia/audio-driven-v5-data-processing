import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_listen_v8_audio_full import recoverable,ready_to_restart


class RecoveryGuardTests(unittest.TestCase):
    def test_only_known_stopped_scheduler_failures_are_retried(self):
        self.assertTrue(recoverable(dict(state='stopping_after_error',error='Missing node heartbeat')))
        self.assertFalse(recoverable(dict(state='cluster_screening',error='Missing node heartbeat')))
        self.assertFalse(recoverable(dict(state='stopping_after_error',error='Frozen screening code changed')))
        self.assertFalse(recoverable(dict(state='stopping_after_error',error='worker CUDA OOM')))

    def test_every_node_needs_reload_headroom(self):
        good=dict(free_gpu_mib=32000,memory_fraction=.5)
        self.assertTrue(ready_to_restart([good,good,good]))
        self.assertFalse(ready_to_restart([good,good]))
        self.assertFalse(ready_to_restart([good,good,dict(good,free_gpu_mib=24000)]))
        self.assertFalse(ready_to_restart([good,good,dict(good,memory_fraction=.05)]))


if __name__=='__main__':unittest.main()
