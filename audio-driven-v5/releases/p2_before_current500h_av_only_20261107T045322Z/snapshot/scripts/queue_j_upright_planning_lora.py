#!/usr/bin/env python3
"""After full audited SFT selection, train fresh Omni-3B LoRA on node8 GPU7."""
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_planning_sft_upright_3b_gpu7_20261105'
FILTER=ROOT/'experiments/listen_j_upright_response_sft_20261105'
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
EXPECTED='0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389'
UUID='GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc'

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def ensure(ok,msg):
 if not ok:raise ValueError(msg)
def status(stage,**kw):save(WORK/'QUEUE_STATUS.json',dict(stage=stage,at=time.time(),pid=os.getpid(),**kw))
def environment(gpu=''):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(WORK/'vendor')+':'+str(ROOT/'src')+':'+str(ROOT/'scripts'),PYTHONUNBUFFERED='1')

def initialize():
 WORK.mkdir(exist_ok=True);(WORK/'logs').mkdir(exist_ok=True)
 if (WORK/'PREPARED.json').exists():
  for p,h in read(WORK/'PREPARED.json')['code'].items():ensure(sha(p)==h,'Frozen training code changed: '+p)
  return
 vendor=WORK/'vendor';vendor.mkdir(exist_ok=True);package=vendor/'p2_planning_sft';package.mkdir(exist_ok=True)
 originals={}
 for p in (ROOT/'src/p2_planning_sft').glob('*.py'):
  originals[str(p)]=sha(p);shutil.copy2(p,package/p.name)
 shutil.copy2(ROOT/'scripts/upright_lora_media.py',vendor/'upright_lora_media.py')
 # Only new source-reference adaptation; native model, collator, loss, Trainer,
 # equivalent AUTO4 decoder and timeout/retry implementation remain unchanged.
 p=package/'reference_data.py';text=p.read_text()
 text=text.replace('from listen_response_sft import load_example','from upright_lora_media import load_example')
 text=text.replace('ROOT=Path(__file__).resolve().parents[2]',"ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')")
 old="    return {n:sha(ROOT/'src'/n) for n in names}"
 new="    result={n:sha(ROOT/'src'/n) for n in names}\n    extra=[Path(__file__),Path(__file__).parent/'fast_media.py',Path(__file__).parents[1]/'upright_lora_media.py',ROOT/'scripts/j_upright_response_sft_data.py',ROOT/'scripts/p2_j_current_qwen_hidden.py']\n    result.update({str(p):sha(p) for p in extra})\n    return result"
 ensure(text.count(old)==1,'Reader identity hook moved');p.write_text(text.replace(old,new))
 p=package/'fast_media.py';text=p.read_text();old='from listen_response_sft.data import resolve_example, require, first_frame';ensure(text.count(old)==1,'Fast input source hook moved');p.write_text(text.replace(old,'from upright_lora_media import resolve_example\nfrom listen_response_sft.data import require, first_frame'))
 for name in ['train.py','model.py','data.py','isolated_decode.py','global_batch.py']:
  ensure(sha(package/name)==originals[str(ROOT/'src/p2_planning_sft'/name)],'Native training math/decoder changed')
 entry='''import argparse\nfrom p2_planning_sft.train import train\np=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--stop-after',type=int);a=p.parse_args()\ntrain(a.config,a.resume,a.smoke,a.output,a.stop_after)\n'''
 (vendor/'train_entry.py').write_text(entry)
 template=read(ROOT/'configs/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102.json')
 template.pop('performance_resume',None);template.pop('extend_from',None)
 template.update(output=str(WORK/'training'),initialization='original_base_model_new_lora')
 template['training'].update(micro_batch=8,gradient_accumulation=32,epochs=1,workers=4,prefetch_factor=4,evaluate=False,save_steps=500)
 # The historical fixed-global-batch shard emits 256 examples on one rank;
 # use native microbatches and accumulation instead, keeping effective batch256.
 template['training'].pop('global_batch_size',None)
 template['data']['seal']=str(WORK/'data_seal.json')
 template['data']['input_pipeline']['decode_log_dir']=str(WORK/'decoder_logs')
 legacy=Path(template['data']['input_pipeline']['legacy_seal']);ensure(sha(legacy)==template['data']['input_pipeline']['legacy_seal_sha256'],'Legacy frame identity changed')
 save(WORK/'config.json',template)
 shutil.copy2(Path(__file__),vendor/'queue_driver.py')
 code={str(p):sha(p) for p in vendor.rglob('*.py')}
 save(WORK/'PREPARED.json',dict(status='prepared_waiting_for_audited_selection',filter=str(FILTER),expected_labels_sha256=EXPECTED,model=template['model'],physical_gpu=7,gpu_uuid=UUID,global_batch=256,micro_batch=8,gradient_accumulation=32,training_epochs=1,code=code,original_sources=originals,config_sha256=sha(WORK/'config.json'),fresh_adapter=True,no_old_optimizer_or_adapter_resume=True,dev_test_excluded_from_gradients=True,evaluation_enabled=False,at=time.time()))
 print(json.dumps(dict(prepared=True,work=str(WORK),physical_gpu=7,effective_batch=256)),flush=True)

def upstream():
 marker=read(FILTER/'COMPLETE.json');ensure(marker['status']=='complete' and marker['processed']==1107313,'Upstream not fully finished')
 final=FILTER/'final';summary=read(final/'summary.json')
 ensure(sha(final/'summary.json')==marker['summary_sha256'] and summary['labels_sha256']==EXPECTED,'Wrong filtered labels')
 ensure(summary['group_split_overlap']==0 and summary['source_interval_overlaps']==0,'Upstream split/time audit failed')
 ensure(sha(final/'MEDIA_READBACK.json')==marker['media_readback_sha256'] and read(final/'MEDIA_READBACK.json')['passed'],'Upstream media readback failed')
 ensure(sum(sum(s['counts'].values()) for s in summary['splits'].values())==1107313,'Upstream partition incomplete')
 return final,summary

def link(source,dest):
 if dest.exists():ensure(dest.stat().st_size==source.stat().st_size and sha(dest)==sha(source),'Existing training input differs');return
 try:os.link(source,dest)
 except OSError:shutil.copy2(source,dest)

def seal_data():
 if (WORK/'data_seal.json').exists():return
 final,summary=upstream();view=WORK/'data';view.mkdir(exist_ok=True)
 for s in ('train','dev'):link(final/(s+'.jsonl'),view/(s+'.jsonl'))
 link(final/'crop_config.json',view/'crop_config.json')
 dbfile=view/'index.sqlite';dbfile.unlink(missing_ok=True);src=sqlite3.connect(f'file:{final}/index.sqlite?mode=ro',uri=True);db=sqlite3.connect(dbfile)
 db.execute(src.execute("SELECT sql FROM sqlite_master WHERE name='samples'").fetchone()[0])
 cursor=src.execute("SELECT * FROM samples WHERE split IN ('train','dev')")
 while True:
  batch=cursor.fetchmany(8192)
  if not batch:break
  db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',batch)
 db.commit();ensure(db.execute("SELECT count(*) FROM samples WHERE split NOT IN ('train','dev')").fetchone()[0]==0,'Test leaked into training view');db.close();src.close()
 adapted=dict(summary)
 adapted['splits']={s:dict(counts=summary['splits'][s]['counts'],clip_hours=summary['splits'][s]['hours']) for s in ('train','dev')}
 adapted['manifests']={s+'.jsonl':summary['manifests'][s+'.jsonl'] for s in ('train','dev')}
 adapted['upstream_summary_sha256']=sha(final/'summary.json');adapted['scope']='filtered train/dev references only; test stays in independent upstream export'
 save(view/'summary.json',adapted)
 cmd=[str(PY),str(WORK/'vendor/queue_driver.py'),'seal-worker']
 with (WORK/'logs/data_seal.log').open('a') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=subprocess.STDOUT,check=True)
 save(WORK/'DATA_READY.json',dict(passed=True,seal_sha256=sha(WORK/'data_seal.json'),selection_complete_sha256=sha(FILTER/'COMPLETE.json'),selection_summary_sha256=sha(final/'summary.json'),train=summary['splits']['train'],dev=summary['splits']['dev'],test_not_in_training_index=True,total_optimizer_steps=math.ceil(summary['splits']['train']['counts']['keep']/256),at=time.time()))

def seal_worker():
 sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
 from p2_planning_sft.reference_data import prepare
 final,summary=upstream()
 prepare(WORK/'data',WORK/'data_seal.json',expected_train=summary['splits']['train']['counts']['keep'],expected_dev=summary['splits']['dev']['counts']['keep'])

def verify_inputs():
 if (WORK/'INPUT_EQUIVALENCE.json').exists():ensure(read(WORK/'INPUT_EQUIVALENCE.json')['passed'],'Input verification failed');return
 cmd=[str(PY),str(WORK/'vendor/queue_driver.py'),'verify-worker']
 with (WORK/'logs/input_verification.log').open('a') as f:subprocess.run(cmd,env=environment(),stdout=f,stderr=subprocess.STDOUT,check=True)

def verify_worker():
 sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
 import numpy as np
 import torch
 from p2_planning_sft.reference_data import ReferencePlanningDataset
 from p2_planning_sft.fast_media import FastInputs
 from p2_planning_sft.data import PlanningCollator
 from upright_lora_media import load_example
 torch.set_num_threads(2);cfg=read(WORK/'config.json');collator=PlanningCollator(cfg['model'],cfg['max_length'],cfg['max_pixels']);tokenizer=collator.get_processor().tokenizer;fast=FastInputs(cfg['data']['input_pipeline']);records=[]
 for split,n in [('train',8),('dev',4)]:
  ds=ReferencePlanningDataset(WORK/'data_seal.json',split,limit=n,seed=cfg['seed'],input_pipeline=cfg['data']['input_pipeline'])
  for index in range(len(ds)):
   row=ds.record(index);original=load_example(row,ds.crop_config);accelerated=fast(row,ds.crop_config)
   ensure(np.array_equal(np.asarray(original['inputs']['listener_first_frame']),np.asarray(accelerated['inputs']['listener_first_frame'])),'Accelerated RGB differs')
   ensure(original['audit']['source_frame_time_s']==accelerated['audit']['source_frame_time_s'],'Accelerated frame PTS differs')
   ensure(np.array_equal(original['inputs']['speaker_audio'],accelerated['inputs']['speaker_audio']),'Speaker audio differs')
   item=ds[index];batch=collator([item]);labels=batch['labels'][0];supervised=labels[labels!=-100].tolist();target=tokenizer.encode(item['target'],add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
   ensure(supervised==target,'Assistant text/EOS mask changed');prefix=collator.source_inputs([item]);ids=prefix['input_ids'][0][prefix['attention_mask'][0].bool()]
   ensure(bool((labels[:len(ids)]==-100).all()) and torch.equal(batch['input_ids'][0,:len(ids)],ids),'Input prefix/mask changed')
   ensure(bool((batch['labels'][batch['attention_mask']==0]==-100).all()),'Padding supervised')
   ensure(len(item['audio'])==round(item['duration']*16000) and not item['audit']['future_listener_media_in_inputs'],'Invalid model conditions')
   records.append(dict(sample_id=item['sample_id'],split=split,duration=item['duration'],frame_read_path=item['audit']['first_frame_read_path'],input_tokens=int(batch['attention_mask'].sum()),target_tokens=len(supervised)))
  ds.close()
 save(WORK/'INPUT_EQUIVALENCE.json',dict(passed=True,cases=records,checks=['source and accelerated RGB/PTS/audio exact equality','real 16k speaker resampling','assistant-only labels and EOS','no future listener media in conditions'],at=time.time()))

def reload_worker():
 sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
 import torch
 from safetensors.torch import load_file
 from peft import get_peft_model_state_dict
 from p2_planning_sft.model import build_model
 torch.set_num_threads(2);checkpoint=WORK/'smoke/checkpoint-1'
 model=build_model(read(WORK/'config.json'),adapter=str(checkpoint),trainable=False)
 expected=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu');actual=get_peft_model_state_dict(model)
 ensure(set(actual)==set(expected),'Reloaded adapter tensor names differ')
 ensure(all(torch.equal(actual[k].cpu(),expected[k]) for k in expected),'Reloaded adapter tensor values differ')
 save(WORK/'SMOKE_RELOAD.json',dict(passed=True,tensors=len(expected),adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),fresh_base_plus_saved_adapter=True,at=time.time()))

def wait_gpu():
 query=['nvidia-smi','-i','7','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 for _ in range(90):
  fields=[v.strip() for v in subprocess.check_output(query,text=True).split(',')];ensure(fields[0]==UUID,'Wrong node/GPU')
  if int(fields[1])<=1024:return
  status('waiting_for_GPU7_release',gpu_memory_mib=int(fields[1]));time.sleep(10)
 raise RuntimeError('GPU7 did not become free; no other process was stopped')

def child(command,name,stage):
 with (WORK/'logs'/name).open('a') as f:
  p=subprocess.Popen(command,env=environment('7'),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  save(WORK/(stage+'_process.json'),dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()[19],command=command,log=f.name,started_at=time.time()))
  while p.poll() is None:
   status(stage,child_pid=p.pid,log=f.name);time.sleep(10)
  ensure(p.returncode==0,stage+' failed: '+f.name)

def control():
 initialize();lock=(WORK/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  prep=read(WORK/'PREPARED.json');ensure(sha(WORK/'config.json')==prep['config_sha256'],'Frozen config changed')
  for p,h in prep['code'].items():ensure(sha(p)==h,'Frozen training source changed')
  while not (FILTER/'COMPLETE.json').exists():
   if (FILTER/'CONTROLLER_ERROR.json').exists():raise RuntimeError('Selection controller failed; training will not start')
   current=read(FILTER/'RUN_STATUS.json');status('waiting_for_full_selection_and_audit',upstream_stage=current['stage']);time.sleep(15)
  upstream();status('sealing_latest_filtered_train_dev');seal_data();status('checking_native_and_fast_inputs');verify_inputs()
  if (WORK/'COMPLETE.json').exists():return
  wait_gpu()
  smoke=WORK/'smoke';base=[str(PY),str(WORK/'vendor/train_entry.py'),'--config',str(WORK/'config.json')]
  if not (WORK/'SMOKE_PASSED.json').exists():
   child(base+['--smoke','--output',str(smoke),'--stop-after','1'],'smoke.log','GPU_smoke')
   ensure(read(smoke/'gradient_audit.json')['finite'] and not read(smoke/'gradient_audit.json')['frozen_gradient'],'Smoke gradient audit failed')
   ck=smoke/'checkpoint-1';ensure((ck/'adapter_model.safetensors').exists() and (ck/'optimizer.pt').exists() and (ck/'scheduler.pt').exists(),'Smoke complete checkpoint missing')
   with (WORK/'logs/smoke_reload.log').open('a') as f:subprocess.run([str(PY),str(WORK/'vendor/queue_driver.py'),'reload-worker'],env=environment(),stdout=f,stderr=subprocess.STDOUT,check=True)
   ensure(read(WORK/'SMOKE_RELOAD.json')['passed'],'Saved adapter reload failed')
   save(WORK/'SMOKE_PASSED.json',dict(passed=True,gradient=read(smoke/'gradient_audit.json'),checkpoint=str(ck),at=time.time()))
  wait_gpu();out=WORK/'training';cmd=list(base)
  if (out/'run.json').exists():
   candidates=sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
   complete=[p for p in candidates if all((p/n).exists() for n in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth'])]
   ensure(complete,'Existing run has no complete resume checkpoint; preserve and inspect')
   cmd+=['--resume',str(complete[-1])]
  if not (out/'COMPLETE.json').exists():child(cmd,'training.log','training')
  done=read(out/'COMPLETE.json');ensure(done['success'] and not done['smoke'] and done['validation_enabled'] is False,'Training not fully complete')
  ensure((out/'final_adapter/adapter_model.safetensors').exists(),'Final adapter missing')
  save(WORK/'COMPLETE.json',dict(status='complete',training=done,training_complete_sha256=sha(out/'COMPLETE.json'),data_seal_sha256=sha(WORK/'data_seal.json'),final_adapter=str(out/'final_adapter'),at=time.time()));status('complete',global_step=done['global_step'])
 except BaseException as exc:
  save(WORK/'QUEUE_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));status('failed',error=repr(exc));raise
 finally:lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','control','seal-worker','verify-worker','reload-worker']);a=p.parse_args()
 {'initialize':initialize,'control':control,'seal-worker':seal_worker,'verify-worker':verify_worker,'reload-worker':reload_worker}[a.action]()
