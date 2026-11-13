import json
from contextlib import closing
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import (JobQueue,acquire,bounded_claim_groups,
                                          read,result_path,write)
from listen_benchmark.visual_schedule import prefetched


class BoundedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name);self.stop=threading.Event()
        self.q=JobQueue(self.work,'vlm')

    def jobs(self,sizes):
        for job,n in sizes:
            write(self.work/'jobs'/f'{job}.json',{'rows':[
                {'sample_id':f'{job}-{i}'} for i in range(n)]})
        (self.work/'ready.jsonl').write_text(''.join(
            json.dumps({'job_id':job})+'\n' for job,_ in sizes))

    def finish_group(self,group):
        for row in group['rows']:write(result_path(self.work,'vlm',row['sample_id']),row)
        for job,handle,data in group['finalize_claims']:
            self.assertTrue(all(result_path(self.work,'vlm',r['sample_id']).exists()
                                for r in data['rows']))
            self.q.finish(job,receipt={'clips':len(data['rows'])});handle.close()

    def test_327_clip_source_is_bounded_and_committed_only_at_end(self):
        self.jobs([('short',1),('large',327),('tail',3)])
        seen=[];sizes=[]
        with closing(bounded_claim_groups(self.q,'one',self.work,50,self.stop)) as groups:
            with closing(prefetched(groups,lambda g:g,1)) as prepared:
                for _,group in prepared:
                    if not group['claims']:break
                    sizes.append(len(group['rows']));self.assertLessEqual(sizes[-1],50)
                    for job,handle,_ in group['claims']:
                        self.assertFalse(handle.closed)
                        self.assertIsNone(acquire(self.q.folder/'claims'/f'{job}.lock',False))
                    seen.extend(r['sample_id'] for r in group['rows'])
                    if 'large-326' not in seen:self.assertFalse(self.q.done_path('large').exists())
                    self.finish_group(group)
        self.assertEqual(sizes,[50]*6+[31])
        self.assertEqual(seen,['short-0']+[f'large-{i}' for i in range(327)]+
                         [f'tail-{i}' for i in range(3)])
        self.assertEqual(read(self.q.path)['completed_clips'],331)
        self.assertEqual(read(self.q.path)['completed'],3)

    def test_stop_recovers_source_and_skips_saved_clips(self):
        self.jobs([('large',127)])
        with closing(bounded_claim_groups(self.q,'one',self.work,50,self.stop)) as groups:
            first=next(groups);self.finish_group(first)
            self.assertFalse(first['finalize_claims'])
            self.assertIsNone(self.q.claim('other'))
            self.stop.set()
        self.assertEqual(read(self.q.path)['completed'],0)
        self.assertEqual(read(self.q.path)['attempts'],{})
        self.stop.clear();seen=[]
        with closing(bounded_claim_groups(self.q,'replacement',self.work,50,self.stop)) as groups:
            for group in groups:
                if not group['claims']:break
                seen.extend(r['sample_id'] for r in group['rows']);self.finish_group(group)
        self.assertEqual(seen,[f'large-{i}' for i in range(50,127)])
        self.assertEqual(read(self.q.path)['completed_clips'],127)

    def test_saved_source_without_receipt_finalizes_with_no_duplicate_work(self):
        self.jobs([('saved',2),('new',1)])
        for i in range(2):write(result_path(self.work,'vlm',f'saved-{i}'),{})
        with closing(bounded_claim_groups(self.q,'one',self.work,50,self.stop)) as groups:
            group=next(groups)
            self.assertEqual([r['sample_id'] for r in group['rows']],['new-0'])
            self.assertEqual([j for j,_,_ in group['finalize_claims']],['saved','new'])
            self.finish_group(group)
        self.assertEqual(read(self.q.path)['completed_clips'],3)

    def test_exception_releases_prefetched_claims_without_admitting_source(self):
        self.jobs([('large',127)])
        with self.assertRaisesRegex(RuntimeError,'consumer error'):
            with closing(bounded_claim_groups(self.q,'one',self.work,50,self.stop)) as groups:
                with closing(prefetched(groups,lambda g:g,1)) as prepared:
                    for _,group in prepared:raise RuntimeError('consumer error')
        job,handle=self.q.claim('replacement');self.assertEqual(job,'large');handle.close()
        self.assertFalse(self.q.done_path('large').exists())


if __name__=='__main__':unittest.main()
