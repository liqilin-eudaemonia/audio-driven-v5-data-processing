import copy,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.config import load,activate
from p2_listen.contracts import read,sha,digest
from p2_listen.cache import load_part
from p2_listen.training_data import load_training_sample
from listen_audio_postprocess.overlay import build_one,sample_contract,compose
from listen_audio_postprocess.encoder import NativeAudioEncoder
from listen_audio_postprocess.recipes import read_listener_audio
RUN=Path(__file__).resolve().parent;c=load(RUN/'view100/config.json');activate(c);torch.set_num_threads(2)
obs=read(RUN/'view100/manifests/observations.json');targets={x['sample_id']:x for x in read(RUN/'view100/manifests/targets.json')}
review=read(ROOT/'experiments/listen_official_chinese100_20261026/results.json');preview=ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100'
encoder=NativeAudioEncoder(c,'cpu');reports=[]
for index in [0,3,22,76]:
 o=obs[index];sid=o['sample_id'];t=targets[sid];start=time.monotonic()
 result=build_one(c,o,t,lambda:encoder)
 root,folder,expected,recipe,label,identity=sample_contract(c,o,t)
 base=load_part(folder,expected);updated=load_training_sample(c,o,t,'cpu');actual=updated[3]
 assert updated[2] is None
 assert torch.equal(base['av']['latents']['latents'],actual['av']['latents']['latents'])
 assert torch.equal(base['support']['video'],actual['support']['video'])
 patch=load_part(root/'cache/audio_postprocess'/sid,identity)
 assert 'av' not in patch and 'video' not in patch
 raw,sr=sf.read(preview/review['cases'][index]['media']['processed'],dtype='float32',always_2d=True)
 expected_audio=encoder.encode(raw,sr,base['geometry'],base['av']['latents']['latents'].shape)
 assert torch.equal(expected_audio['audio_latents']['latents'],actual['av']['audio_latents']['latents'])
 assert torch.equal(expected_audio['audio_support'],actual['support']['audio'])
 for key in ['geometry','audio_support_margin_s','audio_endpoint_tokens_removed']:assert expected_audio[key]==actual[key]
 for e in label['events']:
  assert e['type']!='listener_vocalization' or any(e['start']>=q['start']-1e-8 and e['end']<=q['end']+1e-8 for a in review['cases'][index]['aligned_audio_events'] for q in a['intervals'])
 before=sha(root/'cache/audio_postprocess'/sid/'payload.pt');resume=build_one(c,o,t,lambda:(_ for _ in ()).throw(AssertionError('resume loaded encoder')));assert resume['reused'] and sha(root/'cache/audio_postprocess'/sid/'payload.pt')==before
 reports.append(dict(sample_rank=index+1,sample_id=sid,physical_and_virtual_latents_bit_exact=True,video_bit_exact=True,teacher_not_loaded=True,resume_reused=True,
                     audio_shape=list(actual['av']['audio_latents']['latents'].shape),base_payload_bytes=(folder/'payload.pt').stat().st_size,audio_only_payload_bytes=(root/'cache/audio_postprocess'/sid/'payload.pt').stat().st_size,seconds=time.monotonic()-start))
 print(reports[-1],flush=True)
assert not list((RUN/'view100').rglob('*.wav'))
frozen=read(RUN/'frozen_code_before.json');assert all(sha(ROOT/name)==value for name,value in frozen.items())
report=dict(real_encoder='native LTX audio VAE FP32',device='cpu',cuda_initialized=torch.cuda.is_initialized(),samples=reports,physical_virtual_latents_exact=True,frozen_feature_identity_code_unchanged=True,wav_files_written=0,video_cache_copies=0)
(RUN/'real_encoder_check.json').write_text(json.dumps(report,indent=2));print('PASS: real VAE/loader/resume, 4 cases; no GPU, no WAV, video exact')
