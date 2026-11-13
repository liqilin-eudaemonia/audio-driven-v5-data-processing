import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .core import event_scores,activity_comparison,aggregate_activity,intervals_from_mask
from .media import grid25,activity,read_audio
from .suite import enrich


class MetricsTests(unittest.TestCase):
    def test_silent_not_perfect_f1(self):
        r=activity_comparison([],[],2)
        self.assertIsNone(r['strict']['f1']);self.assertFalse(r['false_activity_on_silent_clip'])
        self.assertEqual(aggregate_activity([r])['false_sound_clip_rate'],0)
    def test_false_positive_on_silence(self):
        r=activity_comparison([],[[.4,.8]],2)
        self.assertEqual(r['strict']['fp'],1);self.assertTrue(r['false_activity_on_silent_clip'])
        self.assertAlmostEqual(r['extra_active_s'],.4)
    def test_missing_response(self):
        r=activity_comparison([[.2,.8]],[],2)
        self.assertEqual(r['strict']['fn'],1);self.assertEqual(r['strict']['f1'],0);self.assertTrue(r['missed_entire_response'])
    def test_repeated_responses_not_merged(self):
        gt=[[.2,.4],[.55,.75]];pred=[[.2,.75]]
        r=event_scores(gt,pred,.2)
        self.assertLessEqual(r['tp'],1);self.assertGreaterEqual(r['fn'],1)
    def test_collar_separate_from_tiou(self):
        g=[[.2,.3]];p=[[.35,.45]]
        self.assertEqual(event_scores(g,p)['tp'],0)
        self.assertEqual(event_scores(g,p,.2)['tp'],1)
    def test_perfect_alignment(self):
        r=event_scores([[.1,.4],[.8,1.2]],[[.1,.4],[.8,1.2]])
        self.assertEqual(r['f1'],1);self.assertEqual(r['onset_mae_s'],0)
    def test_aggregate_counts_not_average_f1(self):
        rows=[activity_comparison([[0,1]],[[0,1]],2),activity_comparison([],[[.2,.5],[.9,1.2]],2)]
        self.assertAlmostEqual(aggregate_activity(rows)['strict']['f1'],.5)
    def test_energy_does_not_imply_speech(self):
        wave=np.ones(32000,np.float32)*.01
        r=activity(wave,dict(energy_dbfs=-50,energy_min_s=.04,energy_gap_s=.03),lambda x:[])
        self.assertTrue(r['acoustic']);self.assertFalse(r['speech'])
    def test_near_silence_does_not_become_speech(self):
        r=activity(np.ones(16000,np.float32)*1e-5,dict(energy_dbfs=-50,energy_min_s=.04,energy_gap_s=.03),lambda x:[[0,1]])
        self.assertEqual(r['acoustic'],[]);self.assertEqual(r['speech'],[])
    def test_gap_preserved(self):
        x=np.r_[np.ones(10),np.zeros(8),np.ones(10)]
        self.assertEqual(len(intervals_from_mask(x)),2)
    def test_numpy_aggregate_json(self):
        from ..video.paired_metrics.core import json_safe
        self.assertEqual(json.loads(json.dumps(json_safe(dict(count=np.int64(2),flag=np.bool_(True))))),dict(count=2,flag=True))
    def test_time_grid(self):
        x=np.arange(30)[:,None,None,None]
        y,a=grid25(x,30,1)
        self.assertEqual(len(y),25);self.assertEqual(y[0,0,0,0],0)
        self.assertLessEqual(np.max(np.abs(y[:,0,0,0]/30-np.arange(25)/25)),1/60+1e-9)
    def test_missing_audio_not_silence(self):
        with self.assertRaises(Exception):read_audio('/missing.wav',2)
    def test_enrich_prevents_target_audio_leak(self):
        ref=dict(id='a',duration_s=1,listener=dict(audio='/target.wav',asr='/asr.json'))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'target.wav';p.touch();ref['listener']['audio']=str(p)
            with self.assertRaises(ValueError):enrich([dict(id='a')],[ref],[dict(id='a',generated_audio=str(p))],Path(td))
    def test_enrich_generated_audio_optional(self):
        ref=dict(id='a',duration_s=1,listener=dict(audio='/target.wav',asr='/asr.json'))
        r=enrich([dict(id='a',generated='/prediction.mp4')],[ref],[dict(id='a')],Path('/'))[0]
        self.assertNotIn('generated_audio',r);self.assertEqual(r['gt_audio'],'/target.wav')


if __name__=='__main__':unittest.main()
