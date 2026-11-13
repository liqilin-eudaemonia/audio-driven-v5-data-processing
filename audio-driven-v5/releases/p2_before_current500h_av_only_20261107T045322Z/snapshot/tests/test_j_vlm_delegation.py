from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import JobQueue,write,read,result_path,bounded_claim_groups
from delegate_listen_j_local_vlm import delegate_nod,entries


class DelegationTests(unittest.TestCase):
    def test_remote_claims_visual_but_skips_delegated_nod(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Path(tmp);(w/'ready.jsonl').write_text('{"job_id":"pending","count":2}\n')
            delegate_nod(w,'pending')
            nod=JobQueue(w,'nod');visual=JobQueue(w,'vlm')
            self.assertIsNone(nod.claim('nod-0'))
            self.assertEqual(read(nod.path)['completed_clips'],0)
            job,handle=visual.claim('gpu-0');self.assertEqual(job,'pending');handle.close()
            self.assertEqual(read(w/'nod/completed/pending.json')['status'],'delegated_branch_not_measurement')

    def test_completed_partial_result_is_reused_by_existing_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=root/'remote';w.mkdir();old=root/'old.json';old.write_text('{"existing":true}\n')
            (w/'ready.jsonl').write_text('{"job_id":"pending","count":2}\n')
            write(w/'jobs/pending.json',{'rows':[{'sample_id':'done'},{'sample_id':'missing'}]})
            target=result_path(w,'vlm','done');target.parent.mkdir(parents=True);target.symlink_to(old)
            with closing(bounded_claim_groups(JobQueue(w,'vlm'),'gpu-0',w,50,threading.Event())) as groups:
                group=next(groups)
                self.assertEqual([r['sample_id'] for r in group['rows']],['missing'])
                self.assertEqual(len(group['finalize_claims']),1)
            self.assertEqual(old.read_text(),'{"existing":true}\n')

    def test_conflicting_source_counts_rejected(self):
        with self.assertRaises(ValueError):entries(b'{"job_id":"x","count":1}\n{"job_id":"x","count":2}\n')


if __name__=='__main__':unittest.main()
