import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import validated_tail_endpoint
from listen_benchmark.j_annotation import JobQueue, read


class TailTests(unittest.TestCase):
    def test_adjacent_pts_resolve_real_oversized_duration_metadata(self):
        end, correction=validated_tail_endpoint(6.645,29.996753246753247,6.700000000000049,6.633333333333383,6.666666666666686)
        self.assertAlmostEqual(end,6.666666666666686)
        self.assertAlmostEqual(correction['reported_end_s'],6.7)

    def test_valid_endpoint_and_provenance_are_unchanged(self):
        self.assertEqual(validated_tail_endpoint(6.645,30,6.666,6.633,6.666),(6.666,None))

    def test_missing_tail_cannot_be_filled_from_a_later_frame(self):
        self.assertEqual(validated_tail_endpoint(6.645,30,6.4,6.366,6.666),(6.4,None))

    def test_missing_following_frame_or_large_pts_gap_cannot_mask_overrun(self):
        for inside,following in [(6.633,None),(6.4,6.666),(6.633,7.0)]:
            self.assertEqual(validated_tail_endpoint(6.645,30,6.7,inside,following),(6.7,None))


class RecoveryTests(unittest.TestCase):
    def test_failed_source_is_exclusively_claimed_and_history_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            (work/'ready.jsonl').write_text('{"job_id":"one"}\n{"job_id":"two"}\n')
            queue=JobQueue(work,'caption')
            job,handle=queue.claim('old-worker')
            for _ in range(3):queue.finish(job,error='incomplete interval')
            handle.close()
            recovered,owner=queue.claim_failed('one','repaired-worker')
            self.assertEqual(recovered,'one')
            self.assertNotIn('one',read(queue.path)['failed'])
            other,other_owner=queue.claim('unchanged-worker')
            self.assertEqual(other,'two')
            history=list((queue.folder/'recoveries').glob('one-*.json'))
            self.assertEqual(len(history),1)
            self.assertEqual(read(history[0])['previous_attempts'],3)
            self.assertEqual(read(history[0])['previous_failure']['error'],'incomplete interval')
            queue.finish('one',receipt={'clips':4});owner.close()
            queue.finish('two',receipt={'clips':2});other_owner.close()
            self.assertIsNone(queue.claim_failed('one','repaired-worker'))
            self.assertEqual(read(queue.path)['completed_clips'],6)

    def test_unfailed_source_cannot_be_taken_by_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            queue=JobQueue(Path(directory),'caption')
            with self.assertRaises(ValueError):queue.claim_failed('one','repaired-worker')


if __name__=='__main__':unittest.main()
