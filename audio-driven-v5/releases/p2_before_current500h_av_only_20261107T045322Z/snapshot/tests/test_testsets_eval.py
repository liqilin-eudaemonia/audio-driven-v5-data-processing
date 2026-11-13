import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import geometry,partition
from evaluation.video.paired_metrics.media import load_pair
from evaluation.video.paired_metrics.run import summarize
from evaluation.video.paired_metrics.core import frechet

class TestEvaluationProtocol(unittest.TestCase):
 def test_responsenet_size_changes_without_retiming(self):
  r=dict(duration_s=7.84,media={'listener':{'video':dict(frames=237,fps=30.,width=640,height=640)}})
  g=geometry(r,'responsenet')
  self.assertEqual((g['width'],g['height'],g['fps'],g['frames']),(512,512,30.,237))
  self.assertEqual((g['native_frames']-1)%8,0);self.assertGreaterEqual(g['native_frames'],g['frames'])
  self.assertAlmostEqual(g['gt_frame_grid_offset_s'],.06)
 def test_partition_has_exactly_the_original_items(self):
  items=[dict(key=str(i),geometry={'native_frames':i+1,'width':512,'height':512}) for i in range(40)]
  shards=partition(items,8)
  self.assertEqual(sorted(x['key'] for ss in shards for x in ss),sorted(x['key'] for x in items))
  self.assertTrue(all(shards));self.assertEqual(shards,partition(items,8))
 def test_resize_requires_declaration_and_only_changes_gt(self):
  from PIL import Image
  g=np.random.default_rng(5).integers(0,256,(3,32,32,3),dtype=np.uint8)
  resized=np.stack([np.array(Image.fromarray(x).resize((16,16),Image.Resampling.BICUBIC)) for x in g])
  with patch('evaluation.video.paired_metrics.media.decode',side_effect=[(g,25.,True),(resized,25.,True)]),patch('evaluation.video.paired_metrics.media.sha256',return_value='test'):
   a,b,m=load_pair(dict(gt='gt',generated='gen',gt_resize=[16,16],gt_resize_method='pil_bicubic'))
   np.testing.assert_array_equal(a,b);self.assertEqual(m['gt_resize']['original_shape'],[32,32,3])
  with patch('evaluation.video.paired_metrics.media.decode',side_effect=[(g,25.,True),(resized,25.,True)]),patch('evaluation.video.paired_metrics.media.sha256',return_value='test'):
   with self.assertRaisesRegex(ValueError,'Spatial sizes'):load_pair(dict(gt='gt',generated='gen'))
 def test_global_frechet_uses_all_features_not_mean_shard_score(self):
  a=np.array([[0.,0.],[1,0],[6,0],[8,0]]);b=np.array([[0.,1.],[2,1],[4,3],[9,3]])
  records=[dict(metrics={},errors={}) for _ in range(4)]
  result=summarize(records,{'fid':[[a[:2],a[2:]],[b[:2],b[2:]]],'fvd':[[],[]]}, {'fid'},'test',['x'],4,{})
  self.assertAlmostEqual(result['metrics']['fid'],frechet(a,b))
  self.assertNotAlmostEqual(result['metrics']['fid'],(frechet(a[:2],b[:2])+frechet(a[2:],b[2:]))/2)
class TestSeparateMetricHost(unittest.TestCase):
 def test_adopted_worker_success_and_failure(self):
  from p2_testsets_eval.metrics import AdoptedMetricProcess
  with tempfile.TemporaryDirectory() as folder:
   marker=Path(folder)/'done.json'
   with patch('p2_testsets_eval.metrics.Path') as proc:
    proc.return_value.read_text.return_value='123 (worker) S 0 0'
    self.assertIsNone(AdoptedMetricProcess(123,marker).poll())
    proc.return_value.read_text.side_effect=FileNotFoundError
    self.assertEqual(AdoptedMetricProcess(123,marker).poll(),1)
    marker.write_text(json.dumps({'status':'complete'}))
    self.assertEqual(AdoptedMetricProcess(123,marker).poll(),0)
 def test_inference_hosts_wait_for_generation_completion(self):
  from p2_testsets_eval.metrics import metric_hosts
  with tempfile.TemporaryDirectory() as folder:
   w=Path(folder)
   (w/'metric.json').write_text(json.dumps({'nodes':[{'name':'evaluation'}]}))
   (w/'infer.json').write_text(json.dumps({'nodes':[{'name':'node8'}]}))
   p=dict(metrics_cluster=str(w/'metric.json'),cluster=str(w/'infer.json'),
          metrics_workers_per_gpu=8,metrics_inference_workers_per_gpu=2,
          metrics_use_released_inference_nodes=True)
   self.assertEqual(set(metric_hosts(w,p)),{'evaluation'})
   (w/'GENERATION_COMPLETE.json').write_text('{}')
   self.assertEqual(set(metric_hosts(w,p)),{'evaluation','node8'})
   self.assertEqual(metric_hosts(w,p)['node8'][2],2)
   p['metrics_use_released_inference_nodes']=False
   self.assertEqual(set(metric_hosts(w,p)),{'evaluation'})
 def test_only_idle_gpus_are_acquired(self):
  from p2_testsets_eval.metrics import idle_gpus
  self.assertEqual(idle_gpus('0, 70000, 143771, 100\n1, 0, 143771, 0\n2, 1200, 143771, 90'),[1])
 def test_utmos_only_does_not_request_sync_or_video(self):
  from evaluation.listener_metrics.run import windows_quality,quality_summary
  from unittest.mock import Mock
  options=dict(quality_context_s=.1,utmos_min_s=3.01,sync_min_s=1.44,quality_min_generated_speech_fraction=.5,quality_kinds=['utmos'])
  worker=Mock();worker.call.return_value=dict(status='success',scores={'gt':{'value':4.},'generated':{'value':3.5}})
  with tempfile.TemporaryDirectory() as folder:
   result=windows_quality(np.ones(80000),np.ones(80000),{'speech':[[.5,4.5]]},{'speech':[[.5,4.5]]},None,None,Path(folder),options,worker)
  self.assertEqual(worker.call.call_count,1);self.assertEqual(worker.call.call_args.args[0]['kind'],'utmos')
  self.assertEqual(result[0]['sync']['reason'],'not_requested')
  report=quality_summary([{'id':'test','audio':{'quality_windows':result}}],('utmos',))
  self.assertEqual(set(report),{'utmos'});self.assertEqual(report['utmos']['generated']['mean'],3.5)
  self.assertEqual(report['utmos']['scored_windows'],1)

if __name__=='__main__':unittest.main()
