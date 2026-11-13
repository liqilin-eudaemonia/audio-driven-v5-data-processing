import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

from p2_listen.contracts import sha,digest,atomic_json
from p2_listen.cache import save_part
from listen_audio_postprocess.recipes import make_recipe,read_listener_audio,masks
from listen_audio_postprocess.overlay import compose,build_one,VERSION
from listen_audio_postprocess.prepare import adapt_canonical
from test_p2_listen_core import label,obs


def official():return dict(words=[dict(id='w',start=.2,end=.6)],sentences=[],vad=[])


class Recipes(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'source.wav';self.wave=np.linspace(-.5,.5,1000,dtype=np.float32)
        sf.write(self.path,self.wave,1000,subtype='FLOAT')
    def tearDown(self):self.tmp.cleanup()
    def recipe(self):return make_recipe('s1',self.path,sha(self.path),official(),[dict(start=.3,end=.4)])
    def test_matches_sequential_physical_pipeline_and_writes_nothing(self):
        r=self.recipe();before=list(self.root.iterdir());actual,sr,meta=read_listener_audio(r)
        g,m,_=masks(r);np.testing.assert_array_equal(actual[:,0],(self.wave*g)*m)
        self.assertTrue(np.all(actual[300:400]==0));self.assertEqual(actual.shape,(1000,1));self.assertEqual(sr,1000)
        self.assertEqual(before,list(self.root.iterdir()));self.assertTrue(meta['waveform_changed'])
    def test_source_tampering_rejected(self):
        r=self.recipe();sf.write(self.path,self.wave+.01,1000,subtype='FLOAT')
        with self.assertRaisesRegex(ValueError,'Source audio changed'):read_listener_audio(r)
    def test_policy_or_invalid_interval_rejected(self):
        r=self.recipe();r['gate_policy']=dict(r['gate_policy'],padding_seconds=.3)
        with self.assertRaisesRegex(ValueError,'Unsupported'):masks(r)
        r=self.recipe();r['extra_mute_intervals']=[dict(start=.6,end=.2)]
        with self.assertRaisesRegex(ValueError,'Invalid'):masks(r)
    def test_recipe_changes_identity_and_silence_is_normal_waveform(self):
        r=self.recipe();s=copy.deepcopy(r);s['official_listener']=dict(words=[],sentences=[],vad=[])
        wave,_,meta=read_listener_audio(s);self.assertFalse(wave.any());self.assertNotEqual(digest(r),digest(s));self.assertTrue(meta['waveform_changed'])
    def test_stereo_and_noop(self):
        sf.write(self.path,np.stack([self.wave,-self.wave],axis=1),1000,subtype='FLOAT')
        r=make_recipe('s1',self.path,sha(self.path),dict(words=[],sentences=[],vad=[dict(start=0,end=1)]))
        wave,_,meta=read_listener_audio(r);self.assertEqual(wave.shape,(1000,2));self.assertFalse(meta['waveform_changed'])


class Composition(unittest.TestCase):
    def base(self):
        return dict(av=dict(latents=dict(latents=torch.ones(1,2,3,4,4)),audio_latents=dict(latents=torch.zeros(1,2,3,4))),
                    support=dict(video=torch.ones(1,3,dtype=torch.bool),audio=torch.ones(1,3,dtype=torch.bool)),geometry={'duration_s':1})
    def audio(self):
        return dict(audio_latents=dict(latents=torch.full((1,2,3,4),.25)),audio_support=torch.ones(1,3,dtype=torch.bool),geometry={'duration_s':1},audio_support_margin_s=.05,audio_endpoint_tokens_removed=0)
    def test_video_is_shared_and_source_unmodified(self):
        b=self.base();p=self.audio();r=compose(b,p)
        self.assertIs(r['av']['latents'],b['av']['latents']);self.assertIs(r['av']['latents']['latents'],b['av']['latents']['latents'])
        self.assertTrue(torch.equal(r['av']['audio_latents']['latents'],p['audio_latents']['latents']));self.assertFalse(b['av']['audio_latents']['latents'].any())
        self.assertIs(compose(b,{'reuse_base_audio':True}),b)
    def test_shape_mismatch_fails(self):
        p=self.audio();p['audio_latents']['latents']=torch.zeros(1,2,2,4)
        with self.assertRaisesRegex(ValueError,'geometry'):compose(self.base(),p)
    def test_noop_skips_encoder_and_resume_skips_reading_waveform(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'base';expected={'part':'base'};save_part(base,expected,self.base())
            recipe={'test':'frozen'};identity={'part':'test'};contract=(root,base,expected,recipe,{},identity)
            factory=lambda:(_ for _ in ()).throw(AssertionError('encoder should not load'))
            with patch('listen_audio_postprocess.overlay.sample_contract',return_value=contract),patch('listen_audio_postprocess.overlay.read_listener_audio',return_value=(np.zeros((10,1)),10,dict(waveform_changed=False))) as reader:
                first=build_one({},dict(sample_id='s1'),{},factory);second=build_one({},dict(sample_id='s1'),{},factory)
                self.assertFalse(first['audio_encoded']);self.assertTrue(second['reused']);self.assertEqual(reader.call_count,1)
            payload=torch.load(root/'cache/audio_postprocess/s1/payload.pt',weights_only=True)
            self.assertNotIn('av',payload);self.assertNotIn('latents',payload)
    def test_training_routes_only_when_explicitly_configured(self):
        from p2_listen.training_data import load_training_sample
        with patch('listen_audio_postprocess.overlay.load_training_sample',return_value='overlay') as fn:
            self.assertEqual(load_training_sample({'data':{'audio_postprocess':{'schema':VERSION}}},{},{},'cpu'),'overlay');fn.assert_called_once()


class CanonicalLabels(unittest.TestCase):
    def test_disjoint_audio_ranges_preserve_visual_text_and_parent(self):
        original=label();original['events'][0]['source']['event_id']='raw-voice';before=copy.deepcopy(original)
        record=dict(audio_events=[dict(event_id='raw-voice',description='Yeah',text=None,intervals=[dict(start=.125,end=.17),dict(start=.2,end=.25)])])
        cfg={'events':{'mapping_version':original['provenance']['mapping_version'],'admission_policy':original['provenance']['admission_policy']}}
        result=adapt_canonical(original,record,{'version':'test'},obs(),cfg)
        self.assertEqual(original,before);self.assertEqual(result['events'][-1],original['events'][-1]);self.assertEqual(len(result['events']),3)
        self.assertNotEqual(result['events'][0]['event_id'],result['events'][1]['event_id'])
        self.assertEqual(result['events'][0]['source']['audio_postprocess']['original_training_event_id'],'voice1')
        self.assertEqual(result['coverage'],original['coverage']);self.assertEqual(result['unknown'],original['unknown'])
    def test_text_rewrite_or_unmapped_event_fails(self):
        original=label();original['events'][0]['source']['event_id']='raw-voice'
        cfg={'events':{k:original['provenance'][k] for k in ('mapping_version','admission_policy')}}
        r=dict(audio_events=[dict(event_id='raw-voice',description='changed',intervals=[dict(start=.1,end=.2)])])
        with self.assertRaisesRegex(ValueError,'preserve'):adapt_canonical(original,r,{},obs(),cfg)
        r['audio_events'][0]['event_id']='unknown'
        with self.assertRaisesRegex(ValueError,'missing'):adapt_canonical(original,r,{},obs(),cfg)


if __name__=='__main__':unittest.main()
