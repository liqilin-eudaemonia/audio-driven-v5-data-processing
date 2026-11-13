import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
from glare_cpu_service_pool import ServicePool


class CpuPoolTests(unittest.TestCase):
    def test_native_models_survive_bundle_close_and_receive_same_requests(self):
        servers=[]
        class Native:
            def __init__(self,gpu,slot):self.calls=[];self.closed=False;servers.append(self)
            def run(self,folder):self.calls.append(folder);return str(folder)+':native-result'
            def close(self):self.closed=True
        pool=ServicePool(Native,2,3)
        a=pool.factory(2,0);self.assertEqual(a.run('clipA'),'clipA:native-result');a.close()
        b=pool.factory(2,0);self.assertIs(a,b);self.assertFalse(servers[0].closed)
        self.assertEqual(b.run('clipB'),'clipB:native-result');self.assertEqual(servers[0].calls,['clipA','clipB'])
        self.assertEqual(pool.metrics()['created'],1);self.assertEqual(pool.metrics()['reused'],1)
        pool.close();self.assertTrue(servers[0].closed)
    def test_all_servers_close_once_at_worker_exit(self):
        closed=[]
        class Native:
            def __init__(self,gpu,slot):self.slot=slot
            def close(self):closed.append(self.slot)
        pool=ServicePool(Native,0,2);pool.factory(0,0);pool.factory(0,1);pool.close();pool.close()
        self.assertEqual(closed,[0,1])
    def test_failed_native_response_is_not_swallowed(self):
        class Native:
            def __init__(self,*a):pass
            def run(self,folder):raise RuntimeError('native input failure')
            def close(self):pass
        pool=ServicePool(Native,0,1)
        with self.assertRaises(RuntimeError):pool.factory(0,0).run('bad')
        pool.close()
    def test_concurrency_is_bounded_by_actual_cpu_budget_and_shared_memory(self):
        cfg=c.config();cfg['execution']=dict(cpu_detectors_per_gpu=6,cpu_decoders_per_gpu=4,cpu_inflight_per_gpu=16)
        node=c.node_by_name(cfg,'node8')
        with patch.object(c,'cpu_budget',return_value=160):
            caps=c.resources(cfg,node)
        self.assertEqual(caps['detector_services'],6);self.assertEqual(caps['decode_workers'],4)
        with patch.object(c,'cpu_budget',return_value=96):
            reduced=c.resources(cfg,node)
        self.assertLess(reduced['detector_services'],6)
        self.assertGreaterEqual(reduced['inflight_groups'],1)


if __name__=='__main__':unittest.main()
