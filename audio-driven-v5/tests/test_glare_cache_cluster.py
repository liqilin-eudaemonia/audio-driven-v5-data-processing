"""CPU-only operational checks. No model loading, source export or CUDA use."""
import copy
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
import glare_cache_cluster_worker as w


def fixture(work,count=4):
    db=sqlite3.connect(work/'routing.sqlite')
    db.execute('CREATE TABLE tasks(stage TEXT,task_id INTEGER,units INTEGER,weight REAL,items INTEGER)')
    db.executemany('INSERT INTO tasks VALUES(?,?,?,?,?)',[('prosody',i,3,3.,1) for i in range(count)])
    db.commit();db.close()


def crash_with_lease(work,signal):
    q=c.TaskQueue(work,'run','prosody');lease=q.claim('crashed');signal.send(lease.task['task_id']);signal.close()
    os._exit(19)  # No Python cleanup: the OS must release the lease.


def concurrent_worker(work,output):
    q=c.TaskQueue(work,'run','prosody');owned=[]
    while (lease:=q.claim(str(os.getpid()))) is not None:
        with lease:
            owned.append(lease.task['task_id'])
            lease.finish(dict(validated=True))
    output.put(owned)


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.work=Path(self.tmp.name);fixture(self.work)
    def tearDown(self):self.tmp.cleanup()
    def test_allocation_is_exact_and_node11_4_7_are_forbidden(self):
        cfg=c.config();self.assertEqual(sum(len(n['gpus']) for n in cfg['nodes']),28)
        self.assertEqual(c.node_by_name(cfg,'node11')['gpus'],[0,1,2,3])
        for name,change in [('node11',lambda n:n['gpus'].append(7)),('node8',lambda n:n.update(host='172.29.104.233'))]:
            bad=copy.deepcopy(cfg);change(c.node_by_name(bad,name));p=self.work/'bad.json';c.save(p,bad)
            with self.assertRaises(AssertionError):c.config(p)
    def test_worker_inherits_original_qwen_dependency_overlay(self):
        cfg=c.config();env=c.environment(c.node_by_name(cfg,'node11'),2,cfg)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
        self.assertEqual(env['PYTHONPATH'].split(':'),cfg['pythonpath'])
        self.assertIn('glare-deps',env['PYTHONPATH'])
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
    def test_balancing_keeps_indivisible_items_and_units_once(self):
        items=[('long',10,10),('a',5,5),('b',5,5),('c',1,1),('d',1,1)]
        assigned=list(c.balanced(items,2));self.assertEqual(len({a[0] for a in assigned}),5)
        costs=[sum(x[2] for x in assigned if x[1]==i) for i in range(2)]
        self.assertEqual(costs,[11,11]);self.assertEqual(sum(a[3] for a in assigned),22)
    def test_live_lease_excludes_another_worker(self):
        q=c.TaskQueue(self.work,'run','prosody');a=q.claim('A');b=q.claim('B')
        self.assertNotEqual(a.task['task_id'],b.task['task_id']);a.close();b.close()
    def test_crash_recovers_without_clock_expiry_or_queue_mutation(self):
        context=multiprocessing.get_context('spawn');a,b=context.Pipe(False)
        child=context.Process(target=crash_with_lease,args=(str(self.work),b));child.start()
        task=a.recv();child.join(20);self.assertEqual(child.exitcode,19)
        q=c.TaskQueue(self.work,'run','prosody');lease=q.claim('recovery')
        self.assertEqual(task,lease.task['task_id']);self.assertFalse(q.done(lease.task));lease.close()
    def test_four_workers_commit_every_task_exactly_once(self):
        context=multiprocessing.get_context('spawn');output=context.Queue()
        children=[context.Process(target=concurrent_worker,args=(str(self.work),output)) for _ in range(4)]
        for p in children:p.start()
        owned=sum([output.get(timeout=20) for _ in children],[])
        for p in children:p.join(20);self.assertEqual(p.exitcode,0)
        self.assertEqual(sorted(owned),[0,1,2,3]);self.assertTrue(c.TaskQueue(self.work,'run','prosody').summary()['complete'])
    def test_no_receipt_until_validated_commit_and_resume_skips_done(self):
        q=c.TaskQueue(self.work,'run','prosody')
        with q.claim('A') as lease:
            task=lease.task
            with self.assertRaises(AssertionError):lease.finish(dict(validated=False))
            self.assertFalse(q.done(task));lease.finish(dict(validated=True))
        next_lease=q.claim('resume');self.assertNotEqual(task['task_id'],next_lease.task['task_id']);next_lease.close()
    def test_stale_completion_receipt_is_rejected(self):
        q=c.TaskQueue(self.work,'run','prosody')
        with q.claim('A') as lease:lease.finish(dict(validated=True))
        with self.assertRaises(AssertionError):c.TaskQueue(self.work,'another-run','prosody').summary()
    def test_shared_cache_fence_blocks_legacy_exclusive_writer(self):
        p=self.work/'fence'
        with c.locked(p,shared=True):
            with c.locked(p,shared=True):pass
            with self.assertRaises(BlockingIOError):
                with c.locked(p):pass
        with c.locked(p):pass
    def test_changed_asset_stat_is_rejected(self):
        asset=self.work/'weight';asset.write_bytes(b'original')
        m=dict(run_id='run',assets_sha256={str(asset):c.sha(asset)})
        proof=self.work/'assets.json';c.save(proof,dict(**m,stats={str(asset):c.stat_key(asset)}))
        c.verify_assets(m,proof);asset.write_bytes(b'changed')
        with self.assertRaises(AssertionError):c.verify_assets(m,proof)
    def test_whole_source_tracks_are_routed_without_overlapping_chunks(self):
        plan=self.work/'plan';plan.mkdir()
        db=sqlite3.connect(plan/'index.sqlite')
        db.executescript('CREATE TABLE tracks(track_id TEXT,data TEXT);CREATE TABLE chunks(track_id TEXT,chunk_index INTEGER);')
        for tid in ['a','b','c']:
            db.execute('INSERT INTO tracks VALUES(?,?)',(tid,json.dumps(dict(track_id=tid))))
            db.executemany('INSERT INTO chunks VALUES(?,?)',[(tid,i) for i in range(3)])
        db.commit();db.close()
        db=sqlite3.connect(self.work/'routing.sqlite');db.execute('CREATE TABLE prosody_assign(track_id TEXT,task_id INTEGER)')
        db.executemany('INSERT INTO prosody_assign VALUES(?,?)',[('a',0),('b',1),('c',0)]);db.commit();db.close()
        cfg=dict(work=str(self.work),source_plan=str(plan))
        a=list(w.task_tracks(cfg,dict(task_id=0)));b=list(w.task_tracks(cfg,dict(task_id=1)))
        self.assertEqual([r[0]['track_id'] for r in a],['a','c']);self.assertEqual(b[0][0]['track_id'],'b')
        self.assertTrue(all(r[1]==[0,1,2] for r in a+b))
    def test_lia_native_query_routes_only_owned_groups_and_model_reused(self):
        old=self.work/'old';parent=self.work/'parent';slot=self.work/'slot'
        for p in (old,parent,slot/'workers'):p.mkdir(parents=True)
        c.save(old/'protocol.json',dict(frozen='shared'));c.save(parent/'protocol.json',dict(frozen='parent'))
        db=sqlite3.connect(slot/'groups.sqlite');db.execute('CREATE TABLE groups(gid INTEGER PRIMARY KEY,worker INTEGER,ordinals TEXT)')
        db.executemany('INSERT INTO groups VALUES(?,?,?)',[(0,3,'[0]'),(1,0,'[1]'),(2,2,'[2]')]);db.commit();db.close()
        (old/'groups.sqlite').symlink_to(slot/'groups.sqlite')
        db=sqlite3.connect(parent/'jobs.sqlite');db.execute('CREATE TABLE jobs(ordinal INTEGER PRIMARY KEY,sid TEXT)')
        db.executemany('INSERT INTO jobs VALUES(?,?)',[(i,'sample'+str(i)) for i in range(3)]);db.commit();db.close()
        db=sqlite3.connect(self.work/'routing.sqlite');db.execute('CREATE TABLE lia_assign(gid INTEGER,task_id INTEGER)')
        db.executemany('INSERT INTO lia_assign VALUES(?,?)',[(0,0),(1,1),(2,0)]);db.commit();db.close()
        loaded=[];decoders=[];routed=[]
        def load_model(p):loaded.append(p);return object()
        pipe=types.SimpleNamespace(path_for=lambda *a:None,base=types.SimpleNamespace(load_protocol=lambda *a:None,load_lia=load_model))
        def native_worker(gpu):
            pipe.base.load_lia(pipe.base.load_protocol(parent))
            db=pipe.sqlite3.connect((slot/'groups.sqlite').as_uri()+'?mode=ro',uri=True)
            rows=list(db.execute('SELECT * FROM groups WHERE worker=? ORDER BY gid',(gpu,)));db.close()
            db=pipe.sqlite3.connect(parent/'jobs.sqlite');owned=list(db.execute('SELECT * FROM jobs'));db.close()
            self.assertEqual([r[0] for r in owned],[r[0] for r in rows])
            routed.append([r[0] for r in rows])
            with pipe.ThreadPoolExecutor(max_workers=1):pass
            with pipe.ThreadPoolExecutor(max_workers=1):pass
            c.save(slot/'workers/gpu0.json',dict(stage='finished',completed=len(rows),reused=0,failed=0))
        pipe.worker=native_worker
        class FakeDecoder:
            def __init__(self,*args):decoders.append(1)
            def __enter__(self):return self
            def __exit__(self,*args):pass
        modules={'run_glare_shared_cache':pipe,'glare_decode_execution':types.SimpleNamespace(DecodeExecutor=FakeDecoder),
                 'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda *_:(0,0)))}
        cfg=dict(work=str(self.work),shared_lia=str(old),lia=dict(execution='baseline'))
        with patch.dict(sys.modules,modules),patch.object(c,'resources',return_value=dict(decode_workers=2,detector_services=1,inflight_groups=2)):
            execute=w.configure_lia(cfg,dict(parent=str(parent)),dict(name='node11'),0,slot)
            execute(dict(task_id=0));execute(dict(task_id=1))
        self.assertEqual(routed,[[0,2],[1]]);self.assertEqual(len(loaded),1);self.assertEqual(len(decoders),2)
        self.assertTrue(str(pipe.path_for('sample','lia.pt')).startswith(str(old/'cache')))
    def run_mock_prosody(self,stage,counts):
        # Model execution is mocked, but the real worker controls leases/commit.
        state=dict(stage=stage,producer_drained=True,counts=counts)
        engine=types.SimpleNamespace(run_engine=lambda *a,**k:state)
        cfg=dict(work=str(self.work),prosody=dict(batch_size=64,loaders=2,queue_chunks=128,max_memory_gib=96,reserve_memory_gib=4))
        m=dict(cache_contract={},cache_root='unused');node=dict(name='node11');folder=self.work/'slot'
        modules={'run_glare_source_prosody_full':engine,
                 'glare_official_capacity_runtime':types.SimpleNamespace(CapacityOfficialRuntime=lambda *a,**k:None),
                 'glare_prosody_subset':types.SimpleNamespace(ChunkCache=lambda *a:None)}
        with patch.dict(sys.modules,modules):w.run_prosody(cfg,m,node,0,folder,c.TaskQueue(self.work,'run','prosody'))
    def test_official_null_and_raw_diagnostic_failures_do_not_exclude_tasks(self):
        self.run_mock_prosody('complete',dict(refusal=1,truncated=1,explicit_null=1,uncovered_source_tail=1))
        self.assertTrue(c.TaskQueue(self.work,'run','prosody').summary()['complete'])
    def test_runtime_error_keeps_task_retryable(self):
        self.run_mock_prosody('complete',dict(input_error_exhausted=1))
        queue=c.TaskQueue(self.work,'run','prosody');summary=queue.summary()
        self.assertEqual(summary['done_tasks'],0);self.assertEqual(summary['failed_tasks'],4)
        self.assertTrue(summary['exhausted']);self.assertFalse(summary['complete'])
        self.assertEqual(queue.retry_errors(),[0,1,2,3]);self.assertEqual(queue.summary()['failed_tasks'],0)
        lease=queue.claim('retry');self.assertIsNotNone(lease);lease.close()
    def test_graceful_stop_never_marks_partially_written_task_done(self):
        self.run_mock_prosody('stopped',{})
        self.assertEqual(c.TaskQueue(self.work,'run','prosody').summary()['done_tasks'],0)


if __name__=='__main__':unittest.main()
