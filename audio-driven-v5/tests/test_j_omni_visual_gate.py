import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import JobQueue,read,write,result_path
from listen_benchmark.j_omni import pending_caption_rows
from listen_benchmark.j_visual_gate import QualityGate
from listen_visual_quality.io import canonical_hash

class GateTests(unittest.TestCase):
    def test_deferred_source_is_neither_lost_nor_failed(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);(w/'ready.jsonl').write_text('{"job_id":"a"}\n{"job_id":"b"}\n')
            q=JobQueue(w,'caption');allowed={'b'}
            jid,h=q.claim('worker',eligible=lambda x:x in allowed);self.assertEqual(jid,'b')
            q.finish(jid,receipt={'clips':1});h.close();s=read(q.path)
            self.assertIn('a',s['deferred']);self.assertEqual(s['attempts'],{});self.assertEqual(s['failed'],{})
            allowed.add('a');jid,h=q.claim('worker',eligible=lambda x:x in allowed);self.assertEqual(jid,'a')
            q.finish(jid,receipt={'clips':1});h.close();self.assertIsNone(q.claim('worker',eligible=lambda x:True))
            s=read(q.path);self.assertEqual(s['completed_clips'],2);self.assertFalse(s['deferred'])

    def test_interrupted_source_waits_for_quality_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);(w/'ready.jsonl').write_text('{"job_id":"a"}\n');q=JobQueue(w,'caption')
            jid,h=q.claim('old');h.close()
            self.assertIsNone(q.claim('new',eligible=lambda _:False));self.assertIn('a',read(q.path)['inflight'])
            jid,h=q.claim('new',eligible=lambda _:True);self.assertEqual(jid,'a');h.close()
            self.assertFalse(read(q.path)['failed'])

    def test_excluded_and_review_never_enter_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            rows=[{'sample_id':s,'source':{'sample_id':s,'duration_s':2.}} for s in ['yes','no','unknown']]
            quality={s:{'sample_id':s,'decision':v} for s,v in [('yes','retain'),('no','exclude'),('unknown','review')]}
            todo,added=pending_caption_rows(d,{'rows':rows},'p',quality)
            self.assertEqual([r['sample_id'] for r in todo],['yes']);self.assertEqual(added,{'exclude':1,'review':1})
            self.assertTrue(result_path(d,'visual_excluded','no').exists());self.assertTrue(result_path(d,'visual_review','unknown').exists())
            self.assertFalse(result_path(d,'caption','no').exists())
            _,again=pending_caption_rows(d,{'rows':rows},'p',quality);self.assertEqual(again,{'exclude':0,'review':0})

    def test_existing_caption_is_preserved_not_regenerated(self):
        with tempfile.TemporaryDirectory() as d:
            row={'sample_id':'old','source':{'sample_id':'old','duration_s':2.}}
            original=dict(source=row['source'],pipeline_id='p',caption='Original words')
            write(result_path(d,'caption','old'),original)
            todo,_=pending_caption_rows(d,{'rows':[row]},'p',{'old':{'decision':'exclude'}})
            self.assertEqual(todo,[]);self.assertEqual(read(result_path(d,'caption','old')),original)

    def test_gate_requires_complete_source_bound_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);write(w/'config.json',{'contract_sha256':'policy'})
            gate=QualityGate({'work':d,'contract_sha256':'policy'});self.assertFalse(gate.ready('job'))
            row={'sample_id':'s','source':{'sample_id':'s','duration_s':2.}}
            job={'job_id':'job','source_result_sha256':'source','rows':[row]}
            p=w/'measure/completed/job.json'
            receipt=dict(contract_sha256='policy',source_result_sha256='source',decisions=[dict(sample_id='s',source_sha256=canonical_hash(row['source']),decision='retain')])
            write(p,receipt);self.assertTrue(gate.ready('job'));self.assertEqual(gate.decisions(job)['s']['decision'],'retain')
            changed=json.loads(json.dumps(job));changed['rows'][0]['source']['duration_s']=3.
            with self.assertRaises(ValueError):gate.decisions(changed)
            receipt['decisions']=[];write(p,receipt)
            with self.assertRaises(ValueError):gate.decisions(job)

class EarlyGateTests(unittest.TestCase):
    def test_early_filter_reuses_measurements_without_completing_other_queues(self):
        from unittest.mock import patch
        from listen_benchmark.j_visual_quality_schedule import ensure_nod_quality
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);w=base/'quality';omni=base/'omni';old=base/'old'
            (omni/'jobs').mkdir(parents=True)
            write(w/'config.json',dict(omni_work=str(omni),contract_sha256='c'))
            row=dict(sample_id='s',source=dict(sample_id='s',duration_s=2.))
            job=dict(job_id='j',source_result_sha256='h',rows=[row]);p=old/'jobs/j.json';write(p,job)
            receipt=dict(source_result_sha256='h',contract_sha256='c',decisions=[dict(sample_id='s',source_sha256=canonical_hash(row['source']),decision='exclude')])
            with patch('listen_benchmark.j_visual_quality_schedule.measure_job',return_value=receipt) as measure:
                for _ in range(2):self.assertEqual(ensure_nod_quality(dict(work=str(w),contract_sha256='c'),p)['s']['decision'],'exclude')
                self.assertEqual(measure.call_count,1)
            self.assertFalse((w/'queues').exists());self.assertFalse((w/'measure/completed/j.json').exists())
            self.assertEqual((omni/'jobs/j.json').resolve(),p.resolve())
            job['rows'][0]['source']['duration_s']=3.;write(p,job)
            with self.assertRaises(ValueError):ensure_nod_quality(dict(work=str(w),contract_sha256='c'),p)

if __name__=='__main__':unittest.main()
