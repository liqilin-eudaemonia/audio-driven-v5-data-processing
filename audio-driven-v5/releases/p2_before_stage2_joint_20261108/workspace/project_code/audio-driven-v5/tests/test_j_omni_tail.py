import json
import os
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.j_annotation import JobQueue, read, write, result_path
from listen_benchmark.j_omni_stream import CrossSourceStream
from listen_benchmark.j_omni_tail import TailSharing, PROTOCOL


class TailTests(unittest.TestCase):
    def setup_work(self, root, size=24):
        rows = [dict(sample_id=f's{i}', source=dict(pair='a', duration_s=2)) for i in range(size)]
        write(root/'jobs/a.json', dict(job_id='a', rows=rows, source_result_sha256='sha'))
        write(root/'job_origins/a.json', dict(old_work='original-owner', source_result_sha256='sha'))
        (root/'ready.jsonl').write_text('{"job_id":"a"}\n')
        write(root/'producer_complete.json', {})
        write(root/'tail_sharing/ENABLED.json', {'protocol': PROTOCOL})
        return JobQueue(root, 'caption')

    def stream(self, root, q, pool, name='owner', packet=8):
        state = dict(pid=os.getpid(), visual_excluded_clips=0, visual_review_clips=0)
        cfg = dict(pipeline_id='p', tail_sharing=dict(enabled=True, worker_count=2,
                                                    packet_size=packet, poll_seconds=1))
        return CrossSourceStream(root, q, name, cfg, SimpleNamespace(pool=pool),
                                 'resources', dict(prefetch=2), None, state, threading.Event())

    def publish(self, stream):
        stream.tail.limit(4)
        stream.tail.advertise(stream.active)
        write(stream.work/'workers'/f'{stream.name}.json', dict(stream.state, heartbeat=time.time()))

    def terminal(self, root, row):
        write(result_path(root, 'caption', row['sample_id']),
              dict(source=row['source'], pipeline_id='p', generation_checks_passed=True))

    def patch_io(self):
        return patch('listen_benchmark.j_omni.verified_job', side_effect=read)

    def test_policy_bounded_lookahead_and_activation(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);q=self.setup_work(w);state={}
            t=TailSharing(w,q,'worker',dict(tail_sharing=dict(enabled=True,worker_count=2)),state)
            self.assertEqual(t.limit(4),1)
            (w/'ready.jsonl').write_text('{"job_id":"a"}\n'*20)
            t.last_check=float('-inf');self.assertEqual(t.limit(4),4)
            self.assertEqual(state['tail_ready_source_lower_bound'],4)
            (w/'ready.jsonl').write_text('')
            (w/'tail_sharing/ENABLED.json').unlink()
            t.last_check=float('-inf');self.assertEqual(t.limit(4),4)

    def test_helper_and_owner_claim_disjoint_rows_with_original_provenance(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w);owner=self.stream(w,q,pool)
            self.assertTrue(owner._claim());self.publish(owner)
            helper=self.stream(w,q,pool,'helper');helper.tail.limit(4)
            self.assertTrue(helper._help());self.assertEqual(len(helper.active['a']['rows']),8)
            self.assertEqual(helper.active['a']['origin']['old_work'],'original-owner')
            helped={r['sample_id'] for r in helper.active['a']['rows']}
            owned=set()
            while (row:=owner._next_row(owner.active['a'])) is not None:owned.add(row['sample_id'])
            self.assertFalse(helped & owned);self.assertEqual(len(helped|owned),24)
            self.assertEqual(helper.scheduling_for('s0')['role'],'helper')
            # No producer receipt while the helper still owns unfinished rows.
            owner._retire();self.assertFalse(q.done_path('a').exists())
            for s in [owner,helper]:
                for item in s.active.values():
                    for h in item['clip_locks'].values():h.close()
                    if item['handle']:item['handle'].close()

    def test_two_helpers_cannot_reserve_same_packet(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w);owner=self.stream(w,q,pool)
            owner._claim();self.publish(owner)
            helpers=[self.stream(w,q,pool,f'h{i}') for i in range(2)]
            ids=[]
            for h in helpers:
                h.tail.limit(4);self.assertTrue(h._help())
                ids.append(set(h.active['a']['clip_locks']))
            self.assertFalse(ids[0]&ids[1]);self.assertEqual(len(ids[0]|ids[1]),16)
            for s in [owner]+helpers:
                for item in s.active.values():
                    for h in item['clip_locks'].values():h.close()
                    if item['handle']:item['handle'].close()

    def test_live_lock_not_stolen_and_released_helper_rows_recover(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w,8);owner=self.stream(w,q,pool)
            owner._claim();self.publish(owner);helper=self.stream(w,q,pool,'helper')
            helper.tail.limit(4);helper._help()
            self.assertIsNone(owner._next_row(owner.active['a']))
            # Simulate process exit: OS releases flock, no output exists.
            for h in helper.active['a']['clip_locks'].values():h.close()
            recovered=[]
            for _ in range(8):
                row=owner._next_row(owner.active['a']);recovered.append(row['sample_id']);self.terminal(w,row)
            self.assertEqual(len(set(recovered)),8)
            owner._retire();self.assertEqual(read(q.path)['completed'],1)
            self.assertEqual(read(q.path)['completed_clips'],8)

    def test_stale_or_legacy_owner_never_offered_to_helpers(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w);owner=self.stream(w,q,pool);owner._claim();self.publish(owner)
            helper=self.stream(w,q,pool,'helper');helper.tail.limit(4)
            p=w/'workers/owner.json';state=read(p);state['pid']+=1;write(p,state)
            self.assertFalse(helper._help())
            state['pid']=os.getpid();state['tail_protocol']='old';write(p,state)
            helper.tail.last_help=float('-inf');self.assertFalse(helper._help())
            owner.active['a']['handle'].close()

    def test_source_owner_restart_preserves_helpers_exclusive_packet(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w,10);owner=self.stream(w,q,pool);owner._claim();self.publish(owner)
            helper=self.stream(w,q,pool,'helper');helper.tail.limit(4);helper._help()
            owner.active['a']['handle'].close()  # source owner exited, helper remains live
            original_packet=helper.active['a']
            self.assertFalse(helper._claim())  # even if the queue refills mid-packet
            self.assertIs(helper.active['a'],original_packet)
            self.assertEqual(len(helper.active['a']['clip_locks']),8)
            replacement=self.stream(w,q,pool,'replacement');self.assertTrue(replacement._claim())
            remaining=[]
            while (row:=replacement._next_row(replacement.active['a'])) is not None:
                remaining.append(row['sample_id']);self.terminal(w,row)
            self.assertEqual(len(remaining),2);replacement._retire()
            self.assertFalse(q.done_path('a').exists())
            for row in helper.active['a']['job']['rows']:self.terminal(w,row)
            helper.active['a']['rows'].clear();helper._retire()
            self.assertFalse(q.done_path('a').exists())  # a helper never commits a source
            self.assertIsNone(replacement._next_row(replacement.active['a']))
            replacement._retire();self.assertEqual(read(q.path)['completed_clips'],10)

    def test_end_to_end_slow_owner_helped_exactly_once(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(4) as pool, ThreadPoolExecutor(2) as runners, self.patch_io():
            w=Path(d);q=self.setup_work(w,40);owner=self.stream(w,q,pool);helper=self.stream(w,q,pool,'helper')
            owner._claim();self.publish(owner)
            seen=[];sources=[]
            def prepare(row,origin,*_):
                sources.append(origin);return row
            def consume(s,delay):
                try:
                    for row,f,_ in s:
                        f.result();time.sleep(delay);seen.append((s.name,row['sample_id']))
                        self.terminal(w,row);self.publish(s)
                finally:s.close()
            with patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=prepare):
                a=runners.submit(consume,owner,.015);b=runners.submit(consume,helper,.001)
                a.result(timeout=8);b.result(timeout=8)
            self.assertEqual(len(seen),40);self.assertTrue(all(v==1 for v in Counter(s for _,s in seen).values()))
            self.assertGreater(sum(n=='helper' for n,_ in seen),0)
            self.assertEqual(set(sources),{'original-owner'})
            self.assertEqual(read(q.path)['completed'],1);self.assertEqual(read(q.path)['completed_clips'],40)
            self.assertEqual(read(q.path)['inflight'],{})

    def test_helper_failure_does_not_finish_parent_and_cpu_drains_before_unlock(self):
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, ThreadPoolExecutor(1) as closer, self.patch_io():
            w=Path(d);q=self.setup_work(w,8);owner=self.stream(w,q,pool);owner._claim();self.publish(owner)
            helper=self.stream(w,q,pool,'helper');release=threading.Event();started=threading.Event()
            def prepare(row,*_):
                if row['sample_id']=='s1':started.set();release.wait(5)
                return row
            with patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=prepare):
                next(helper);self.assertTrue(started.wait(1));helper.record_failure('s0',ValueError('helper failure'))
                ending=closer.submit(helper.close);time.sleep(.05)
                self.assertFalse(ending.done());self.assertIsNone(owner.tail.reserve('s1'))
                release.set();ending.result(timeout=3)
            self.assertEqual(read(q.path)['attempts'],{})
            self.assertFalse(q.done_path('a').exists())
            self.assertTrue(helper.tail.helper_failed_path('s0').exists())
            lock=owner.tail.reserve('s1');self.assertIsNotNone(lock);lock.close()
            owner.active['a']['handle'].close()

    def test_reader_stop_keeps_claims_until_inference_consumer_finishes(self):
        from listen_benchmark.j_annotation import acquire
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(2) as pool, self.patch_io():
            w=Path(d);q=self.setup_work(w,8);s=self.stream(w,q,pool)
            with patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=lambda row,*_:row):
                row,f,_=next(s);f.result()  # now handed to GPU consumer
                sid=row['sample_id'];s.stop.set()
                with self.assertRaises(StopIteration):next(s)
                self.assertIsNone(s.tail.reserve(sid))
                self.assertIsNone(acquire(q.folder/'claims/a.lock',False))
                self.terminal(w,row);s.persisted(sid)  # GPU finished persisting
                h=s.tail.reserve(sid);self.assertIsNotNone(h);h.close()
                s.close()
                h=acquire(q.folder/'claims/a.lock',False);self.assertIsNotNone(h);h.close()


if __name__=='__main__':unittest.main()
