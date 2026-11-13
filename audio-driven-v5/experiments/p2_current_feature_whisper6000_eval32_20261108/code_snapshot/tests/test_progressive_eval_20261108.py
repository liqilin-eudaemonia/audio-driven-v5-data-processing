import sys
import tempfile
import threading
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_progressive_eval_20261108 import ResourcePoller,ShardBinder,BASELINE,binding_inputs_ready
from p2_testsets_eval.protocol import read


class ProgressiveTests(unittest.TestCase):
    def test_partial_preparation_is_not_ready_for_binding(self):
        from p2_testsets_eval.protocol import save
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);save(work/'protocol.json',dict(workers=2))
            self.assertFalse(binding_inputs_ready(work))
            save(work/'CPU_ACCEPTANCE.json',dict(success=True))
            save(work/'shared_items.json',[]);save(work/'shards/00.json',[])
            self.assertFalse(binding_inputs_ready(work))
            save(work/'shards/01.json',[])
            self.assertTrue(binding_inputs_ready(work))

    def test_slow_node_does_not_block_fast_node_resources(self):
        block=threading.Event();fast=threading.Event()
        cluster=dict(nodes=[dict(name='node12'),dict(name='node11')])
        def probe(node):
            if node['name']=='node11':block.wait(5)
            else:fast.set()
            return {0:dict(free=100000,used=0,util=0)}
        poll=ResourcePoller(cluster,probe)
        try:
            poll.snapshot();self.assertTrue(fast.wait(2))
            result=poll.snapshot()
            self.assertIn('node12',result);self.assertNotIn('node11',result)
        finally:block.set();poll.close()

    def test_ready_shard_publishes_while_other_shard_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            b=ShardBinder.__new__(ShardBinder);b.work=Path(temp);b.arm='arm';b.p=dict(datasets=dict(seamless={},responsenet={}))
            b.source={};b.published=set();b.shards=[[
                dict(key='seamless/a',dataset='seamless',id='a'),dict(key='responsenet/b',dataset='responsenet',id='b')],
                [dict(key='seamless/c',dataset='seamless',id='c'),dict(key='responsenet/d',dataset='responsenet',id='d')]]
            b.committed={'seamless/a','responsenet/b'};b.rows=lambda ds,ids:[dict(id=i) for i in sorted(ids)]
            b.publish_ready();self.assertEqual(b.published,{0})
            self.assertTrue((b.work/'metric_manifests/arm/seamless/00.jsonl').exists())
            self.assertFalse((b.work/'metric_manifests/arm/seamless/01.jsonl').exists())

    def test_real_rows_equal_original_native_metric_manifests(self):
        b=ShardBinder(BASELINE);b.committed=set(b.by_key)
        for rank in [0,7]:
            for ds in b.p['datasets']:
                ids={x['id'] for x in b.shards[rank] if x['dataset']==ds}
                actual=b.rows(ds,ids)
                path=BASELINE/'metric_manifests'/b.arm/ds/f'{rank:02d}.jsonl'
                import json
                expected=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                self.assertEqual(actual,expected)

    def test_final_binding_cannot_omit_the_last_sample(self):
        b=ShardBinder.__new__(ShardBinder);b.committed=set(range(399))
        with self.assertRaises(AssertionError):b.canonical_audit()


if __name__=='__main__':unittest.main()
