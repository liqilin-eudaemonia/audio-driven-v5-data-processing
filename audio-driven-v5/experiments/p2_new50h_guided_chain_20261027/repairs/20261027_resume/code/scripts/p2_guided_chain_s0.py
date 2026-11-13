#!/usr/bin/env python3
"""Real GPU test of all eight arms, initialization, gradient ownership and resume."""
from pathlib import Path
import argparse,copy,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
from p2_guided_chain import backend


def run(root,out):
 import torch
 from p2_matrix16.training import train
 from p2_matrix16.checkpoint import checked
 rows=read(root/'view/observations.json')
 # Exercise the longest actual media, and a long Gemma/Qwen schedule.
 longest=max(rows,key=lambda o:o['requested_duration_s'])['sample_id']
 p=read(root/'plans/protocol.json')
 longest_text=max(p['sample_ids'],key=lambda sid:(root/'plans/samples'/(sid+'.json')).stat().st_size)
 chosen=list(dict.fromkeys([longest,longest_text]))
 class SmokeDataset(backend.Dataset):
  def __init__(self,c):
   super().__init__(c);mapping={o['sample_id']:o for o in self.rows};self.rows=[mapping[sid] for sid in chosen]
 class SmokeBackend:
  Dataset=SmokeDataset
  data_config=staticmethod(backend.data_config);data_identity=staticmethod(backend.data_identity);source_identity=staticmethod(backend.source_identity);initialize_stage2=staticmethod(backend.initialize_stage2)
 stats=read(root/'CACHE_VALIDATED.json')['statistics'];results=[]
 free,_=torch.cuda.mem_get_info();os.environ['P2_TRAIN_MEMORY_GIB']='128' if free>120*2**30 else '64'
 for loss in ('av','av_event','av_feature','av_event_feature'):
  parent=None
  for stage in (1,2):
   name=f'guided_{loss}_s{stage}';folder=out/name;c=read(root/'configs'/(name+'.json'));c['output']=str(folder);c['training'].update(steps=2,global_batch=1,warmup_steps=1)
   c['s0_sample_ids']=chosen
   atomic_json(folder/'config.json',c,immutable=True)
   pre=dict(schema='p2-matrix-preflight-v1',data_identity=backend.data_identity(c),experiment_mode=['guided',loss,stage],samples=len(chosen),statistics=stats,source=backend.source_identity())
   atomic_json(folder/'preflight.json',pre)
   cp2=folder/'step_000002.pt';cp1=folder/'step_000001.pt';cp0=folder/'step_000000.pt'
   if not cp2.with_suffix('.complete.json').exists():
    if not cp1.with_suffix('.complete.json').exists():
     resume=cp0 if cp0.with_suffix('.complete.json').exists() else None
     if resume is None and (folder/'run.json').exists():(folder/'run.json').unlink()
     train(c,parent=parent if resume is None else None,resume_path=resume,stop_after=1,audit=True,backend=SmokeBackend)
    train(c,resume_path=cp1,audit=True,backend=SmokeBackend)
   saved=checked(cp2)
   if saved['step']!=2:raise ValueError('S0 resume failed')
   metrics=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
   if [m['step'] for m in metrics]!=[1,2]:raise ValueError('Resume duplicated/lost metrics')
   if not all(m['grad_norm']['connector']>0 for m in metrics):raise ValueError('AV gradient missing')
   if stage==2 and not any(m['grad_norm'].get('qwen_lora',0)>0 for m in metrics):raise ValueError('Qwen LoRA gradients missing')
   for metric in metrics:
    paths=metric['gradient_paths']
    if 'event' not in loss and any(paths[g]['event']!=0 for g in ('connector','qwen_lora')):raise ValueError('Event probe leaked')
    if 'feature' not in loss and 'features' in metric['grad_norm']:raise ValueError('Disabled feature optimizer exists')
   results.append(dict(id=name,checkpoint=str(cp2),sha256=sha(cp2),metrics=metrics));parent=str(cp2)
   atomic_json(out/'progress.json',dict(completed=len(results),total=8,latest=name))
   del saved
   import gc
   gc.collect();torch.cuda.empty_cache()
 atomic_json(out/'COMPLETE.json',dict(success=True,selected=chosen,runs=results,script_sha256=sha(__file__),source=backend.source_identity(),checks=['real_AV_video_audio_forward_backward','all_8_arms','guided_cached_and_online_conditions','same_stage1_parent','strict_resume','gradient_ownership','finite_checkpoint']),immutable=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--chunk',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root if a.chunk=='final' else Path(a.chunk),a.output)
