import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.reuse_scores import feature_blocks
spec=importlib.util.spec_from_file_location('rn200',ROOT/'scripts/p2_responsenet_random200.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def population(n=586):
 refs=[];audio={}
 for i in range(n):
  refs.append(dict(id=str(i),duration_s=3.+i%9,group='official' if i%7 else 'local',sample_id=str(i//4)))
  audio[str(i)]=dict(audio=dict(gt=dict(peak=.1 if i<48 else 0.,acoustic=[[0.,.2]] if i<34 else [])))
 return refs,audio


def test_sample_fixed_without_model_scores_and_quota_rounding():
 refs,audio=population()
 a,strata=m.stratified_sample(refs,audio)
 b,_=m.stratified_sample(list(reversed(refs)),audio)
 assert [r['id'] for r in a]==[r['id'] for r in b]
 assert len(a)==len({r['id'] for r in a})==200
 assert sum(x['selected'] for x in strata)==200
 assert all(abs(x['selected']-x['ideal'])<1.00001 for x in strata)
 for r in refs:r['model_score']=1e9 if int(r['id'])%2 else -1e9
 c,_=m.stratified_sample(refs,audio)
 assert [r['id'] for r in a]==[r['id'] for r in c]
 d,_=m.stratified_sample(refs,audio,seed=99)
 assert [r['id'] for r in a]!=[r['id'] for r in d]


def test_feature_rows_are_exactly_partitioned_and_bad_offsets_rejected():
 records=[dict(id='a',fid=dict(status='features_ready',feature_count=2),fvd=dict(status='features_ready',feature_count=1)),
          dict(id='b',fid=dict(status='features_ready',feature_count=3),fvd=dict(status='features_ready',feature_count=1))]
 features={k:np.arange(n*4,dtype=np.float32).reshape(n,4) for k,n in [('fid_gt',5),('fid_generated',5),('fvd_gt',2),('fvd_generated',2)]}
 found=feature_blocks(records,features)
 assert np.array_equal(found['b']['fid'][0],features['fid_gt'][2:])
 assert np.array_equal(found['a']['fvd'][1],features['fvd_generated'][:1])
 features['fid_gt']=features['fid_gt'][:-1]
 with pytest.raises(ValueError,match='row mismatch'):feature_blocks(records,features)


def test_response_whisper_source_wave_is_exact_base_condition(tmp_path):
 import soundfile as sf
 from scipy.signal import resample_poly
 from p2_testsets_eval.protocol import save
 source=tmp_path/'speaker.wav';wave=np.zeros(16000,np.float32);wave[8000:10000]=.125
 sf.write(source,wave,8000,subtype='FLOAT')
 save(tmp_path/'input_audit.json',dict(speaker_audio=str(source),source_only=True,
  target_description_in_model_input=False,listener_future_media_in_model_input=False))
 item=dict(geometry=dict(duration_s=2.),reference=dict(speaker=dict(audio=str(source)),media=dict(speaker=dict(audio=dict(samples=16000,sample_rate=8000)))))
 found,meta=m.rn_speaker_wave(item,tmp_path)
 assert np.array_equal(found,resample_poly(wave,2,1).astype(np.float32))
 assert len(found)==32000 and meta['origin_s']==0 and not meta['silence_removed']
 assert not found[:15980].any()
