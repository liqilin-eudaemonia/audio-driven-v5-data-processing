import sys,json,copy,time
from pathlib import Path
from unittest.mock import patch
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path[:0]=[str(ROOT/'releases/p2_audio_postprocess_v1/src'),str(ROOT/'src')]
from p2_listen.config import load,activate
from p2_listen.contracts import read,atomic_json,digest
from p2_listen.training_data import load_training_sample,audio_cache_identity
from p2_listen.cache import load_part
from p2_listen import features
from listen_audio_postprocess import overlay

def rows(path,full):
 if full:
  with path.open() as f:
   for line in f:
    line=line.strip().rstrip(',')
    if line.startswith('{'):yield json.loads(line)
 else:yield from read(path)

results=[]
for name in ['p2_train50h_audio_postprocess_20261026','p2_full_audio_postprocess_crop384_20261026']:
 prior=OUT/(name+'.json')
 if prior.exists():
  report=read(prior);assert report['status']=='passed';results.extend(report['rows']);continue
 view=ROOT/'experiments'/name/'view';c=load(view/'config.json');activate(c);full=name.startswith('p2_full');counts={'changed':0,'reused':0,'rejected_unadmitted':0};scanned=0
 for o,t in zip(rows(view/'manifests/observations.json',full),rows(view/'manifests/targets.json',full)):
  assert o['sample_id']==t['sample_id'];sid=o['sample_id'];scanned+=1
  label=read(view/'labels/canonical'/f'{sid}.json')
  if not label['training_admitted']:
   if counts['rejected_unadmitted']<2:
    try:load_training_sample(c,o,t)
    except ValueError:counts['rejected_unadmitted']+=1
    else:raise AssertionError('Unadmitted row loaded')
   if counts['changed']==16 and counts['reused']==16 and counts['rejected_unadmitted']>=1:break
   continue
  if scanned>=5000:raise RuntimeError('Insufficient branch coverage')
  cache=view/'cache/audio_postprocess'/sid
  # Pick both physical branches without ever reading/processing a waveform.
  raw=torch.load(cache/'payload.pt',map_location='cpu',weights_only=True);kind='reused' if raw.get('reuse_base_audio') else 'changed'
  if counts[kind]>=16:continue
  root,base_folder,expected,recipe,canonical,ident=overlay.sample_contract(c,o,t)
  with patch.object(overlay,'read_listener_audio',side_effect=AssertionError('Training must not regenerate WAV')),patch.object(features,'load_training_sample',side_effect=AssertionError('No old teacher path')):
   sample=load_training_sample(c,o,t,'cpu')
  base=load_part(base_folder,expected);av=sample[3];events=load_part(features.folder(c,'events',sid),features.identity(c,'events',o,label=canonical))
  assert sample[2] is None;torch.testing.assert_close(av['av']['latents']['latents'],base['av']['latents']['latents'],rtol=0,atol=0)
  expect_audio=base['av']['audio_latents'] if kind=='reused' else raw['audio_latents']
  torch.testing.assert_close(av['av']['audio_latents']['latents'],expect_audio['latents'],rtol=0,atol=0)
  torch.testing.assert_close(av['support']['audio'],base['support']['audio'] if kind=='reused' else raw['audio_support'],rtol=0,atol=0)
  for mod in events:
   for k,v in events[mod].items():
    if isinstance(v,torch.Tensor):torch.testing.assert_close(sample[4][mod][k],v,rtol=0,atol=0)
    else:assert sample[4][mod][k]==v
  results.append(dict(dataset=name,sample_id=sid,audio=kind,events_from_postprocessed=True,teacher_read=False,waveform_read=False));counts[kind]+=1
  atomic_json(OUT/'real_progress.json',dict(dataset=name,counts=counts,scanned=scanned))
  if counts['changed']==16 and counts['reused']==16 and (not full or counts['rejected_unadmitted']>=1):break
  if scanned>=5000:raise RuntimeError('Insufficient branch coverage')
 print(name,counts,'scanned',scanned,flush=True)
 assert counts['changed']==counts['reused']==16
 atomic_json(OUT/(name+'.json'),dict(status='passed',counts=counts,scanned=scanned,rows=[r for r in results if r['dataset']==name]))
atomic_json(OUT/'real_validation.json',dict(status='passed',rows=results,samples=len(results),training_started=False,cache_rebuilt=False))
