import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import JobQueue,write,read
from listen_benchmark.simple_timeline import validate_timeline


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name);self.q=JobQueue(self.work,'vlm')
        (self.work/'ready.jsonl').write_text(''.join(json.dumps({'job_id':x})+'\n' for x in ['a','b']))

    def test_claims_exclude_live_owners_and_recover_dead_owner(self):
        a,ha=self.q.claim('one');b,hb=self.q.claim('two')
        self.assertEqual((a,b),('a','b'));self.assertIsNone(self.q.claim('three'))
        ha.close();a2,h2=self.q.claim('replacement');self.assertEqual(a2,'a')
        self.q.finish(a2,{'completed':True});h2.close();hb.close()
        b2,h3=self.q.claim('replacement');self.assertEqual(b2,'b');h3.close()

    def test_duplicate_journal_never_repeats_completed_job(self):
        with (self.work/'ready.jsonl').open('a') as f:f.write(json.dumps({'job_id':'a'})+'\n')
        for expected in ('a','b'):
            job,handle=self.q.claim('one');self.assertEqual(job,expected)
            self.q.finish(job,{'completed':True});handle.close()
        self.assertIsNone(self.q.claim('one'));self.assertEqual(read(self.q.path)['completed'],2)

    def test_receipt_commit_recovers_interrupted_queue_update(self):
        job,handle=self.q.claim('one');write(self.q.done_path(job),{'completed':True});handle.close()
        b,h=self.q.claim('two');self.assertEqual(b,'b');h.close()
        self.assertEqual(read(self.q.path)['completed'],1)

    def test_error_retry_exhaustion_is_visible_not_success(self):
        for _ in range(3):
            job,h=self.q.claim('one');self.assertEqual(job,'a');self.q.finish(job,error='bad');h.close()
        self.assertIn('a',read(self.q.path)['failed']);self.assertEqual(read(self.q.path)['completed'],0)
        job,h=self.q.claim('one');self.assertEqual(job,'b');h.close()

    def test_partial_journal_line_is_not_consumed(self):
        (self.work/'ready.jsonl').write_text('{"job_id":"a"')
        self.assertIsNone(self.q.claim('one'));self.assertEqual(read(self.q.path)['offset'],0)
        with (self.work/'ready.jsonl').open('a') as f:f.write('}\n')
        job,h=self.q.claim('one');self.assertEqual(job,'a');h.close()

    def test_branches_have_independent_claims(self):
        a,ha=self.q.claim('vlm');q2=JobQueue(self.work,'nod');b,hb=q2.claim('nod')
        self.assertEqual(a,b);ha.close();hb.close()


class TimePrecisionTests(unittest.TestCase):
    def test_roundoff_does_not_rewrite_original_model_response(self):
        event={'events':[{'start':0,'end':2.78,'modality':'visual','behavior':'The person smiles.'}]}
        original=copy.deepcopy(event);validate_timeline(event,2.7799999999999994)
        self.assertEqual(event,original)

    def test_actual_out_of_bounds_and_zero_length_still_rejected(self):
        for start,end in [(0,2.780001),(-.000001,1),(1,1),(2,1)]:
            with self.assertRaises(ValueError):
                validate_timeline({'events':[{'start':start,'end':end,'modality':'visual','behavior':'x'}]},2.78)


if __name__=='__main__':unittest.main()
