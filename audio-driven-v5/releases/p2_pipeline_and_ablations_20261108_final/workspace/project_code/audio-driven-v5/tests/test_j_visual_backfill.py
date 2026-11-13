import sys,tempfile,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_visual_backfill import refill

class BackfillTests(unittest.TestCase):
    def fixture(self,d):
        w=Path(d)/'quality';o=Path(d)/'omni';o.mkdir();journal=''.join(json.dumps(dict(job_id=f'j{i}'))+'\n' for i in range(20));(o/'ready.jsonl').write_text(journal)
        write(w/'config.json',dict(omni_work=str(o),priority_start_offset=len(journal)))
        write(w/'feed.json',dict(offset=len(journal),backfill_offset=0,published=10))
        write(w/'queues/measure/state.json',dict(offset=0,inflight={},failed={}))
        (w/'ready.jsonl').write_text('')
        for i in range(20):write(o/'caption/completed'/f'j{i}.json',dict(clips=1))
        cfg=dict(visual_quality_backfill_enabled=True,visual_quality_gate=dict(work=str(w)),nodes=dict(node4=dict(visual_quality_workers=64),node12=dict(visual_quality_workers=96)))
        return w,o,cfg
    def test_refills_all_cpu_capacity_and_skips_completed_seed_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            w,o,c=self.fixture(d);write(w/'measure/completed/j0.json',dict(clips=1))
            r=refill(c);self.assertEqual(r['configured_workers'],160);self.assertEqual(r['target_sources'],240);self.assertEqual(r['published'],19)
            ids=[json.loads(s)['job_id'] for s in (w/'ready.jsonl').read_text().splitlines()];self.assertNotIn('j0',ids);self.assertEqual(len(set(ids)),19)
            self.assertEqual(refill(c)['published'],0)
    def test_new_foreground_prevents_backfill(self):
        with tempfile.TemporaryDirectory() as d:
            w,o,c=self.fixture(d)
            with (o/'ready.jsonl').open('a') as f:f.write('{"job_id":"foreground"}\n')
            self.assertEqual(refill(c)['status'],'foreground_pending');self.assertEqual((w/'ready.jsonl').read_text(),'')
    def test_full_queue_is_not_extended(self):
        with tempfile.TemporaryDirectory() as d:
            w,o,c=self.fixture(d);write(w/'queues/measure/state.json',dict(offset=0,inflight={str(i):{} for i in range(240)},failed={}))
            self.assertEqual(refill(c)['published'],0);self.assertEqual(read(w/'feed.json')['backfill_offset'],0)

if __name__=='__main__':unittest.main()
