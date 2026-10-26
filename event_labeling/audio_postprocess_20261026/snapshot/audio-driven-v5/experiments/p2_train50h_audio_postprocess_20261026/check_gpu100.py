import sys,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.config import load,activate
from p2_listen.contracts import read,atomic_json
from p2_listen.training_data import load_training_sample
from listen_audio_postprocess.encoder import NativeAudioEncoder
from listen_audio_postprocess.recipes import read_listener_audio
run=Path(__file__).resolve().parent;c=load(run/'view/config.json');activate(c);torch.set_num_threads(2)
obs={x['sample_id']:x for x in read(run/'view/manifests/observations.json')};targets={x['sample_id']:x for x in read(run/'view/manifests/targets.json')}
review=read(ROOT/'experiments/listen_official_chinese100_20261026/results.json');encoder=NativeAudioEncoder(c,'cuda:0');reports=[];start=time.monotonic()
for case in review['cases']:
 sid=case['sample_id'];recipe=read(run/'view/audio_postprocess/recipes'/f'{sid}.json')
 virtual,sr,meta=read_listener_audio(recipe);physical,psr=sf.read(ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100'/case['media']['processed'],dtype='float32',always_2d=True)
 assert sr==psr;np.testing.assert_array_equal(virtual,physical)
 _,_,_,current,_=load_training_sample(c,obs[sid],targets[sid],'cpu')
 encoded=encoder.encode(physical,psr,current['geometry'],current['av']['latents']['latents'].shape)
 equal=torch.equal(encoded['audio_latents']['latents'],current['av']['audio_latents']['latents'])
 # No-op samples intentionally retain the original cache; new encodings must match.
 if meta['waveform_changed']:assert equal,(sid,'GPU cache differs from reviewed WAV encoding')
 reports.append(dict(sample_id=sid,waveform_bit_exact=True,cache_matches_gpu_physical_encoding=equal,updated_audio=meta['waveform_changed']))
 if len(reports)%20==0:print('PASS',len(reports),flush=True)
result=dict(status='passed',samples=len(reports),waveforms_bit_exact=100,updated_caches_checked=sum(x['updated_audio'] for x in reports),seconds=time.monotonic()-start,records=reports)
atomic_json(run/'gpu100_acceptance.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}))
