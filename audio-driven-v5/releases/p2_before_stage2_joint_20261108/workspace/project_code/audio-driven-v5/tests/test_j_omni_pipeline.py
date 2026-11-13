import copy
import json
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_omni import read_audio,SCHEMA,file_sha,nod_evidence,EvidenceUnavailable,producer_step
from listen_benchmark.j_annotation import write,read,JobQueue,result_path
from listen_benchmark.official_audio_gate import POLICY
from listen_audio_postprocess.recipes import make_recipe,read_listener_audio
from listen_j_annotation_supervisor import selected_gpu_ids


class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def recipe(self,official):
  path=self.p/'source.wav';wave=np.linspace(-.5,.5,1000,dtype=np.float32);sf.write(path,wave,1000,subtype='FLOAT')
  return dict(schema=SCHEMA,source=dict(path=str(path),sha256=file_sha(path),sample_rate=1000,channels=1,source_samples=1000,first_sample=100,last_sample=600),official_listener=official,extra_mute_intervals=[],gate_policy=POLICY)
 def test_segment_recipe_equal_reviewed_clip_reader(self):
  off=dict(words=[dict(id='w',start=.1,end=.2)],sentences=[],vad=[]);r=self.recipe(off)
  out,sr,identity=read_audio(r);raw,_=sf.read(r['source']['path'],dtype='float32',always_2d=True)
  path=self.p/'reference.wav';sf.write(path,raw[100:600],sr,subtype='FLOAT')
  ref=make_recipe('test',path,file_sha(path),off);expected,_,_=read_listener_audio(ref)
  np.testing.assert_array_equal(out,expected);self.assertEqual(out.shape,(500,1));self.assertTrue(np.all(out[300:]==0));self.assertFalse(identity['full_silence'])
 def test_empty_official_union_keeps_duration_and_mutes_all(self):
  r=self.recipe(dict(words=[],sentences=[],vad=[]));out,sr,identity=read_audio(r)
  self.assertEqual(out.shape,(500,1));self.assertTrue(identity['full_silence']);self.assertFalse(np.any(out))
 def test_missing_metadata_is_not_silence(self):
  r=self.recipe(dict(words=[],sentences=[]))
  with self.assertRaises(ValueError):read_audio(r)
 def test_source_change_rejected(self):
  r=self.recipe(dict(words=[],sentences=[],vad=[]));Path(r['source']['path']).write_bytes(b'changed')
  with self.assertRaises(ValueError):read_audio(r)
 def test_outside_clip_not_padded_into_clip(self):
  r=self.recipe(dict(words=[dict(start=-.2,end=-.01)],sentences=[],vad=[]))
  self.assertTrue(read_audio(r)[2]['full_silence'])
 def nod(self,status='ok'):
  row=dict(sample_id='sample',source=dict(duration_s=1.,listener_id='p',video_transform={'crop':1}))
  n=dict(sample_id='sample',source=row['source'],duration_s=1.,fps=25,profile='recall',status=status,events=[])
  write(result_path(self.p,'nod','sample'),n);return row,n
 def test_empty_nod_success_allowed(self):
  row,_=self.nod();self.assertEqual(nod_evidence(row,self.p)['events'],[])
 def test_unavailable_nod_not_empty(self):
  row,_=self.nod('unavailable')
  with self.assertRaises(EvidenceUnavailable):nod_evidence(row,self.p)
 def test_nod_time_crop_identity_rejected(self):
  row,n=self.nod();n=copy.deepcopy(n);n['source']['video_transform']={'crop':2};write(result_path(self.p,'nod','sample'),n)
  with self.assertRaises(ValueError):nod_evidence(row,self.p)
 def test_unfinished_nod_not_empty(self):
  with self.assertRaises(FileNotFoundError):nod_evidence({'sample_id':'missing'},self.p)
 def test_queue_claim_exclusion_and_resume(self):
  (self.p/'ready.jsonl').write_text('{"job_id":"one","count":1}\n{"job_id":"one","count":1}\n{"job_id":"two","count":1}\n')
  q=JobQueue(self.p,'caption');a,ha=q.claim('node1');b,hb=q.claim('node2');self.assertEqual((a,b),('one','two'))
  self.assertIsNone(q.claim('node3'));ha.close();recovered,h=q.claim('node3');self.assertEqual(recovered,'one')
  q.finish('one',dict(clips=1));h.close();q.finish('two',dict(clips=1));hb.close();self.assertIsNone(q.claim('node1'));self.assertEqual(read(q.path)['completed_clips'],2)
 def test_producer_waits_for_nod_receipt(self):
  old=self.p/'old';work=self.p/'new';write(work/'config.json',dict(nodes={'node':dict(old_work=str(old),scope_sources=2),'compute_only':dict(gpu_ids=[4,5,6,7])},backlog_sources=10))
  lines='{"job_id":"one","count":1}\n{"job_id":"two","count":1}\n';(old/'queues/nod').mkdir(parents=True)
  (old/'ready.jsonl').write_text(lines);write(old/'queues/nod/state.json',dict(offset=len(lines),failed={}))
  write(old/'jobs/one.json',{});write(old/'jobs/two.json',{})
  write(old/'nod/completed/one.json',dict(job_id='one',clips=1,source_result_sha256='x'))
  s=producer_step(work);self.assertEqual(s['published_sources'],1);self.assertIn('two',s['nodes']['node']['pending'])
  write(old/'nod/completed/two.json',dict(job_id='two',clips=1,source_result_sha256='y'))
  s=producer_step(work);self.assertEqual(s['published_sources'],2);self.assertTrue(s['all_sources_published'])
  self.assertNotIn('compute_only',s['nodes'])
  producer_step(work);self.assertEqual(len((work/'ready.jsonl').read_text().splitlines()),2)
 def test_cpu_only_supervision_explicitly_enabled(self):
  self.assertEqual(selected_gpu_ids(dict(visual_enabled=False)),[])
  with self.assertRaises(ValueError):selected_gpu_ids(dict(gpu_ids=[]))
  self.assertEqual(selected_gpu_ids(dict(gpu_ids=[0,2])),[0,2])


if __name__=='__main__':unittest.main()
