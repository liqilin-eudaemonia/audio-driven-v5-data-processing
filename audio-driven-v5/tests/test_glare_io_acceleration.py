"""CPU tests for two-worker routing and durable async writes. No GPU models."""
import copy
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster
from glare_async_writes import BoundedWriter,LiaWrites
import run_glare_source_prosody_full as prosody
from glare_prosody_subset import ChunkCache,generation_rows,read


class AccelerationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_two_workers_split_per_gpu_resources_and_have_distinct_paths(self):
        cfg=cluster.config();cfg['execution']=dict(lia_workers_per_gpu=2)
        caps=dict(cpu_budget=128,decode_workers=4,detector_services=3,inflight_groups=7)
        node=cluster.node_by_name(cfg,'node11')
        with patch.object(cluster,'resources',return_value=caps):
            a=cluster.worker_resources(cfg,node,0);b=cluster.worker_resources(cfg,node,1)
        for key in ['decode_workers','detector_services','inflight_groups']:self.assertEqual(a[key]+b[key],caps[key])
        self.assertEqual((a['detector_services'],b['detector_services']),(2,1))
        self.assertNotEqual(cluster.slot_dir(cfg,node,0,0),cluster.slot_dir(cfg,node,0,1))
        with self.assertRaises(AssertionError):cluster.worker_resources(cfg,node,2)
    def test_shared_gpu_reservation_and_exclusive_worker_locks(self):
        gpu=self.root/'gpu.lock';slot=self.root/'worker0.lock'
        with cluster.locked(gpu,shared=True),cluster.locked(slot):
            with cluster.locked(gpu,shared=True),cluster.locked(self.root/'worker1.lock'):pass
            with self.assertRaises(BlockingIOError):
                with cluster.locked(slot):pass
            with self.assertRaises(BlockingIOError):
                with cluster.locked(gpu):pass
    def test_writer_has_real_backpressure_and_bounded_ownership(self):
        release=threading.Event();entered=threading.Event();third=threading.Event();writer=BoundedWriter(2)
        def hold():entered.set();assert release.wait(3)
        first=writer.submit(hold);self.assertTrue(entered.wait(2));second=writer.submit(lambda:2)
        def submit_third():writer.submit(lambda:3);third.set()
        t=threading.Thread(target=submit_third);t.start()
        try:self.assertFalse(third.wait(.05));self.assertEqual(writer.metrics()['pending'],2)
        finally:release.set();t.join(3);writer.close()
        self.assertTrue(third.is_set());self.assertEqual(writer.metrics()['succeeded'],3)
    def test_writer_failure_prevents_later_receipt_publication(self):
        release=threading.Event();entered=threading.Event();writer=BoundedWriter(3);done=[]
        def fail():entered.set();release.wait(3);raise OSError('disk failure')
        writer.submit(fail);self.assertTrue(entered.wait(2));writer.submit(lambda:done.append('published'));release.set()
        with self.assertRaises(RuntimeError):writer.close()
        self.assertEqual(done,[])
    def pipeline(self,gate=None,fail=False):
        loads=[]
        def load(path,**kw):loads.append(threading.current_thread().name);return pickle.loads(Path(path).read_bytes())
        t=types.SimpleNamespace(load=load)
        def tensor(path,values):
            if gate is not None:
                gate[0].set();assert gate[1].wait(3)
            if fail:raise OSError('write failed')
            Path(path).write_bytes(pickle.dumps(values));self.assertEqual(t.load(path,map_location='cpu',weights_only=True),values)
        def finish(job,kind,payload,p,**extra):
            self.assertTrue(Path(payload).exists())
            piper.save(Path(payload).with_suffix('.json'),dict(payload_sha256=cluster.sha(payload)))
        piper=types.SimpleNamespace(save_tensor=tensor,finish_stage=finish,save=cluster.save)
        return piper,t,loads
    def test_lia_ram_dependency_and_complete_receipt_wait_for_verified_disk_write(self):
        entered=threading.Event();release=threading.Event();p,t,loads=self.pipeline((entered,release))
        payload=self.root/'motion.pt';done=self.root/'sample.complete.json';status=self.root/'workers/gpu0.json'
        originals=(p.save_tensor,p.finish_stage,p.save,t.load)
        with LiaWrites(p,8,t):
            p.save_tensor(payload,dict(motion=[1.,2.,3.]));self.assertTrue(entered.wait(2))
            self.assertFalse(payload.exists());self.assertEqual(t.load(payload,map_location='cpu',weights_only=True)['motion'],[1.,2.,3.])
            self.assertEqual(loads,[])  # The actual disk roundtrip has not happened yet.
            p.finish_stage({},'lia',payload,{})
            p.save(done,dict(identity='same-source'))
            p.save(status,dict(gpu=0,stage='finished',completed=1))
            self.assertFalse(done.exists());self.assertEqual(read(status)['completed'],0)
            self.assertEqual(read(status)['stage'],'draining_writes');release.set()
        self.assertTrue(done.exists());self.assertTrue(payload.with_suffix('.json').exists())
        self.assertEqual(read(status)['completed'],1);self.assertEqual(read(status)['stage'],'finished')
        self.assertTrue(loads and loads[0].startswith('glare-write'))
        self.assertEqual((p.save_tensor,p.finish_stage,p.save,t.load),originals)
    def test_lia_write_failure_never_publishes_complete_and_restores_functions(self):
        p,t,_=self.pipeline(fail=True);original=p.save_tensor;done=self.root/'bad.complete.json'
        with self.assertRaises(RuntimeError):
            with LiaWrites(p,8,t):
                p.save_tensor(self.root/'bad.pt',dict(motion=[1]))
                p.save(done,dict(identity='source'))
        self.assertFalse(done.exists());self.assertIs(p.save_tensor,original)
    def source(self,cache,write_queue,overlap=False,limit=None):
        entered=threading.Event();release=threading.Event();inferences=[]
        track=dict(track_id='test-track',audio=dict(sha256='a'*64));indices=list(range(8))
        def tracks(_):yield track,indices
        def prepared(*args):return dict(track=track,indices=indices,reused={},prepare_s=0.,error=None,wave=np.zeros(8*38400,np.float32))
        original=cache.commit
        def commit(*args):
            if overlap and args[1]==0:entered.set();assert release.wait(3)
            return original(*args)
        def infer(waves):
            inferences.append(len(waves))
            if overlap and len(inferences)==2:
                self.assertTrue(entered.wait(2));release.set()
            gen=generation_rows([[8,99]],[99],64)[0];gen.update(eos_ids=[99],max_new_tokens=64)
            return [('NULL',dict(gen),'NULL') for _ in waves]
        runtime=types.SimpleNamespace(load=lambda:None,infer=infer)
        folder=self.root/('run-'+str(write_queue));folder.mkdir(exist_ok=True)
        with patch.object(prosody,'tracks_for',tracks),patch.object(prosody,'prepare_source',prepared),patch.object(cache,'commit',commit):
            state=prosody.run_engine(folder,dict(chunks=8,tracks=1),runtime,cache,limit=limit,
                                    batch_size=4,queue_size=16,write_queue=write_queue)
        return state,track,indices
    def test_prosody_next_inference_overlaps_pending_writes_and_final_counts_are_durable(self):
        cache=ChunkCache(self.root/'cache',dict(test=True))
        state,track,indices=self.source(cache,16,overlap=True)
        self.assertEqual(state['new'],8);self.assertEqual(state['scheduled_new'],8)
        self.assertEqual(state['async_writes']['pending'],0);self.assertEqual(state['stage'],'complete')
        self.assertTrue(all(cache.load(track,i) for i in indices))
    def test_prosody_sync_async_payload_values_and_raw_replies_match(self):
        a=ChunkCache(self.root/'sync',dict(test=True));b=ChunkCache(self.root/'async',dict(test=True))
        sa,track,indices=self.source(a,0);sb,_,_=self.source(b,16)
        self.assertEqual(sa['counts'],sb['counts'])
        for i in indices:
            x,y=a.load(track,i),b.load(track,i)
            self.assertEqual(x['analysis'],y['analysis']);self.assertEqual(x['generation'],y['generation'])
            with np.load(a.paths(track,i)[0]) as v,np.load(b.paths(track,i)[0]) as w:
                for key in v.files:np.testing.assert_array_equal(v[key],w[key])
    def test_prosody_limit_uses_scheduled_work_and_does_not_overshoot(self):
        state,track,indices=self.source(ChunkCache(self.root/'cache',dict(test=True)),16,limit=5)
        self.assertEqual(state['new'],5);self.assertEqual(state['scheduled_new'],5)
        self.assertEqual(state['stage'],'bounded_run_complete')
    def test_benchmark_arms_share_cohort_but_never_output_caches_or_gpu_launches(self):
        from prepare_glare_cache_benchmark import prepare
        cfg=cluster.config();plan=self.root/'plan';old=self.root/'old';plan.mkdir();old.mkdir()
        cfg.update(source_plan=str(plan),shared_lia=str(old))
        cluster.save(plan/'protocol.json',dict(index_sha256='index'))
        cluster.save(old/'protocol.json',dict(groups_sha256='groups'))
        db=sqlite3.connect(plan/'index.sqlite');db.execute('CREATE TABLE chunks(track_id TEXT,chunk_index INTEGER)')
        db.executemany('INSERT INTO chunks VALUES(?,?)',[(t,i) for t in ['a','b','c'] for i in range(3)]);db.commit();db.close()
        db=sqlite3.connect(old/'groups.sqlite');db.execute('CREATE TABLE groups(gid INTEGER,pairs INTEGER)')
        db.executemany('INSERT INTO groups VALUES(?,?)',[(0,1),(1,2),(2,1)]);db.commit();db.close()
        configs=[]
        def frozen(conf):configs.append(copy.deepcopy(conf));return dict(run_id='fake',assigned_units=dict(lia=2,prosody=3))
        with patch.object(cluster,'prepare',side_effect=frozen):result=prepare(cfg,self.root/'bench',lia_pairs=2,prosody_chunks=3)
        self.assertEqual(len(result['arms']),4)
        self.assertEqual(len({c['benchmark']['cohort'] for c in configs}),1)
        self.assertEqual(len({c['benchmark']['output_root'] for c in configs}),4)
        self.assertEqual({(cluster.execution(c)['lia_workers_per_gpu'],cluster.execution(c)['async_lia']) for c in configs},
                         {(1,False),(2,False),(1,True),(2,True)})
        self.assertTrue(all((Path(c['work'])/'STOP').exists() for c in configs))


if __name__=='__main__':unittest.main()
