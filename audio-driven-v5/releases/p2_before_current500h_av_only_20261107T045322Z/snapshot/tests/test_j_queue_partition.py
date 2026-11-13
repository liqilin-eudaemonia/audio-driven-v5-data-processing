import json
from pathlib import Path
import tempfile
import unittest

from listen_benchmark.j_annotation import JobQueue,read,write
from listen_benchmark.j_queue_partition import split_journal,transfer_unclaimed


def line(job,count=10):return (json.dumps({'job_id':job,'count':count})+'\n').encode()


class PartitionTests(unittest.TestCase):
    def test_three_of_nineteen_capacity_preserves_protected_prefix(self):
        raw=b''.join(line(str(i),10) for i in range(192))
        cut=len(line('0')+line('1'))
        left,right,p=split_journal(raw,[{'offset':cut,'inflight':{'0':{}},'failed':{}}],3/19)
        self.assertEqual(p['moved_clips'],300)
        self.assertEqual(left[:cut],raw[:cut])
        self.assertFalse({'0','1'}&set(p['moved_sources']))

    def test_preserve_both_offsets_inflight_and_duplicate_prefix(self):
        raw=line('a')+line('b')+line('c',90)+line('d',50)+line('e',40)+line('f',20)+line('a')
        states=[{'offset':len(line('a')),'inflight':{'a':{}},'failed':{}},
                {'offset':len(line('a')+line('b')),'inflight':{'b':{}},'failed':{}}]
        left,right,p=split_journal(raw,states)
        self.assertEqual(left[:p['cut_offset']],raw[:p['cut_offset']])
        a={json.loads(x)['job_id'] for x in left.splitlines()};b={json.loads(x)['job_id'] for x in right.splitlines()}
        self.assertFalse(a&b);self.assertEqual(a|b,set('abcdef'));self.assertTrue({'a','b'}<=a)
        self.assertEqual(p['source_clips'],220)

    def test_reject_bad_offset_or_partial_journal(self):
        with self.assertRaises(ValueError):split_journal(line('a'),[{'offset':2}])
        with self.assertRaises(ValueError):split_journal(line('a')[:-1],[{'offset':0}])

    def setup_work(self,root):
        source=root/'source';dest=root/'destination';tx=root/'transaction';source.mkdir()
        write(source/'ingest_complete.json',{'complete':True})
        (source/'ready.jsonl').write_bytes(b''.join(line(str(i),i+1) for i in range(12)))
        queues={b:JobQueue(source,b) for b in ['vlm','nod']}
        return source,dest,tx,queues

    def test_live_claims_remain_and_each_branch_processes_disjoint_scope(self):
        with tempfile.TemporaryDirectory() as d:
            src,dst,tx,qs=self.setup_work(Path(d));owned=[]
            for b,q in qs.items():owned.append(q.claim(b))
            plan=transfer_unclaimed(src,dst,tx)
            self.assertEqual(plan['state'],'committed')
            self.assertEqual(transfer_unclaimed(src,dst,tx),plan)
            for b,q in qs.items():
                assigned=[];other=JobQueue(dst,b)
                for queue in [q,other]:
                    ids=[]
                    while (claim:=queue.claim(b)) is not None:
                        job,handle=claim;ids.append(job);queue.finish(job,{'clips':int(job)+1});handle.close()
                    assigned.append(set(ids))
                self.assertFalse(assigned[0]&assigned[1]);self.assertEqual(set.union(*assigned),set(map(str,range(1,12))))
            for _,handle in owned:handle.close()

    def test_incomplete_ingest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):transfer_unclaimed(Path(d)/'a',Path(d)/'b',Path(d)/'tx')

    def test_stale_prepared_transaction_not_reapplied_after_claim(self):
        with tempfile.TemporaryDirectory() as d:
            src,dst,tx,qs=self.setup_work(Path(d));transfer_unclaimed(src,dst,tx)
            (src/'ready.jsonl').write_bytes((tx/'original_ready.jsonl').read_bytes())
            job,h=qs['vlm'].claim('gpu-0')
            try:
                with self.assertRaisesRegex(ValueError,'stale'):transfer_unclaimed(src,dst,tx)
            finally:h.close()


if __name__=='__main__':unittest.main()
