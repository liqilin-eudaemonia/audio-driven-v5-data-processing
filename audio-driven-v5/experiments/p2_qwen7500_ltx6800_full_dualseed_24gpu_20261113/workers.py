"""Thin adapters around the validated native cache, renderer and decoder."""
import copy,os,socket,subprocess,time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from common import *

def device(rank):
 s=slots()[rank];assert os.environ['CUDA_VISIBLE_DEVICES']==str(s['gpu'])
 assert s['node']['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 value=subprocess.check_output(['nvidia-smi','-i',str(s['gpu']),'--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert value[0].strip()==read(BASE/'ACTIVATED.json')['uuids'][str(rank)]
 return dict(rank=rank,node=s['node']['name'],gpu=s['gpu'],uuid=value[0].strip(),total_mib=float(value[1]))

def cache_worker(dataset,rank,shard,stack,rt,d,cfg):
 import torch
 from PIL import Image
 from p2_testsets_eval import whisper_runtime as wr
 import p2_j_sft_hidden as hidden
 cd=conditions(dataset);p=read(cd/'protocol.json');part=read(cd/'shards'/f'{shard:02d}.json')
 identities=read(dataset_work(dataset)/'INPUT_IDENTITIES.json')
 old_load=hidden.load_session
 def reader(path):
  assert path=='inference-only-adapter:7500'
  return SimpleNamespace(source=SimpleNamespace(manifest={'config':d}),protocol={'adapter':p['planner_adapter_identity']})
 def load(session):
  result=old_load(session)
  from peft import get_peft_model_state_dict
  from safetensors.torch import load_file
  actual=get_peft_model_state_dict(result[2]);expected=load_file(str(Path(p['planner_adapter'])/'adapter_model.safetensors'))
  assert len(actual)==504 and actual.keys()==expected.keys() and all(torch.equal(actual[k].cpu(),expected[k]) for k in actual)
  save(cd/'adapter_acceptance'/f'{shard:02d}.json',dict(at=time.time(),adapter=p['planner_adapter_identity'],all504_tensors_equal=True));return result
 stack.enter_context(patch.object(hidden,'Reader',reader));stack.enter_context(patch.object(hidden,'load_session',load))
 memo={};native_sha=wr.sha
 def memo_sha(path):
  f=Path(path);s=f.stat();key=(str(f),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  if key not in memo:memo[key]=native_sha(f)
  return memo[key]
 stack.enter_context(patch.object(wr,'sha',memo_sha))
 if dataset=='responsenet':
  import p2_responsenet_random200 as rn
  stack.enter_context(patch.object(rn,'sha',memo_sha));wave_fn=rn.rn_speaker_wave
 else:wave_fn=wr.speaker_wave
 def conditioning(x,folder):
  identity=identities[x['key']]
  assert sha(folder/'first_frame.png')==identity['image_sha256'] and sha(folder/'input_audit.json')==identity['audit_sha256']
  wave,audit=wave_fn(x,folder);frozen=identity['whisper_source']
  assert audit['waveform_sha256']==frozen['waveform_sha256'] and audit['source_sha256']==frozen['source_sha256']
  with Image.open(folder/'first_frame.png') as image:image=image.convert('RGB')
  return dict(sample_id=x['id'],duration=x['geometry']['duration_s'],image=image,audio=wave)
 stack.enter_context(patch.object(rt,'conditioning',conditioning))
 native=import_file('recorded_native_cache',RECORDED_CACHE)
 native.cache(cd,part,0,p)
 for x in part:
  f=cd/'conditions'/x['key'];assert rt.ready(f/'qwen') and rt.ready(f/'text')
  assert read(f/'qwen/plan.json')['adapter']==p['planner_adapter_identity'] and sha(f/'input_audit.json')==identities[x['key']]['audit_sha256']
 save(cd/f'CACHE_COMPLETE_{shard}.json',dict(success=True,at=time.time(),keys=[x['key'] for x in part],adapter=p['planner_adapter_identity']))

def worker(dataset,seed,stage,rank,shard):
 verify();g=device(rank);w=work(dataset,seed);cd=conditions(dataset)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(CAPS[stage]*1024/g['total_mib'])
 from p2_testsets_eval import runtime as rt
 from p2_listen.config import extractor_config
 d=read(BASE/'DATA_CONFIG.json');cfg=extractor_config(d);items=read(dataset_work(dataset)/'shared_items.json');p=read(w/'protocol.json')
 record=dict(at=time.time(),pid=os.getpid(),hostname=socket.gethostname(),stage=stage,dataset=dataset,seed=seed,shard=shard,**g)
 save(w/'devices'/f'{stage}_{rank}_{shard}.json',record);progress=rt.progress
 with ExitStack() as stack:
  stack.enter_context(patch.object(rt,'data_config',lambda:(d,cfg)))
  stack.enter_context(patch.object(rt,'progress',lambda work,st,unused,**kw:progress(work,st,rank,**kw)))
  if stage=='cache':cache_worker(dataset,rank,shard,stack,rt,d,cfg)
  elif stage=='render':
   assert (cd/'CACHE_COMPLETE.json').exists()
   import p2_all_lora_followups_eval_20261109 as native
   import p2_ltx_events.inference as old
   from p2_ltx_all_lora import inference,checkpoint,config
   loading=import_file('verified_ltx_checkpoint',TRUSTED/'run.py');plan=read(BASE/'PLAN.json');identity=plan['identity'];checked_validate=config.validate
   def load(path,device='cuda:0'):
    assert str(path)==p['arms'][0]['checkpoint'];v=loading.pinned(6800)
    def normalize(c):assert c==identity['experiment'];return checked_validate(copy.deepcopy(plan['normalized_config']))
    with patch.object(checkpoint,'checked',lambda _:v),patch.object(config,'source_identity',lambda:identity['source']),patch.object(config,'validate',normalize):result=inference.load_model(path,device)
    assert len(result[0].text_targets)==1632 and list(result[0].state_dict())==list(v['model'])
    save(w/'model_acceptance'/f'{rank}.json',dict(**record,step=6800,strict_tensors=4809,targets=1632,checkpoint_sha256=p['arms'][0]['original_sha256']));return result
   native.STEP=6800;native.ARM=ARM;native.WORK=w
   with patch.object(old,'load_model',load):native.render(w,items,0,p)
  elif stage=='decode':
   index=CompletionIndex(w,items)
   with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']):
    while not stopped(dataset,seed):
     remaining,backlog=index.refresh()
     if len(index.done)==len(items):break
     if backlog:rt.decode(w,items,rank,p,p['arms'])
     else:time.sleep(3)
 save(w/'worker_complete'/f'{stage}_{rank}_{shard}.json',dict(**record,finished=time.time()))

def publish(dataset,seed,shard):
 from evaluation.listener_metrics.suite import enrich
 w=work(dataset,seed);p=read(w/'protocol.json');items=read(w/'shards'/f'{shard:02d}.json');paired=[];mapping=[]
 for x in items:
  r=x['reference'];f=w/'generated'/ARM/x['key'];m=read(f/'metadata.json');assert valid_media(m,p,x,sha(w/'conditions'/x['key']/'qwen/plan.json'))
  if dataset=='seamless':v=dict(id=r['id'],dataset='seamless',identity='listener',gt_reference=r['gt_reference'])
  else:v=dict(id=r['id'],dataset='responsenet-j-paired-evaluation-v1',identity='listener',gt=r['gt'],gt_resize=[512,512],gt_resize_method='pil_bicubic')
  v.update(generated=str(f/'generated.mp4'),listening_intervals=r['listening_intervals'],source_start=r['source_start'],source_end=r['source_end'],timeline='same_pair_window');paired.append(v)
  mapping.append(dict(id=r['id'],generated_audio=str(f/'listener.wav')))
 out=w/'metric_manifests'/ARM/dataset/f'{shard:02d}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
 if out.is_symlink():
  assert {r['id'] for r in rows(out)}=={x['id'] for x in items};return
 values=enrich(paired,[x['reference'] for x in items],mapping,w);out.write_text(''.join(json.dumps(v)+'\n' for v in values))

def audit_inference(dataset,seed):
 from p2_testsets_eval.runtime import ready
 w=work(dataset,seed);p=read(w/'protocol.json');items=read(dataset_work(dataset)/'shared_items.json')
 for x in items:
  f=w/'generated'/ARM/x['key'];assert ready(f)
  assert valid_media(read(f/'metadata.json'),p,x,sha(w/'conditions'/x['key']/'qwen/plan.json'))
 save(w/'INFERENCE_COMPLETE.json',dict(success=True,at=time.time(),dataset=dataset,seed=seed,samples=len(items),qwen=7500,ltx=6800,media_exported=True,scoring_complete=False))
