"""Reuse seven completed arms; add7500/8000/8256 with unchanged native generation."""
import argparse,collections,copy,difflib,importlib.util,json,os,socket,subprocess,time,traceback
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
OLD=ROOT/'experiments/p2_planning_ckpt_seamless200_20261113'
spec=importlib.util.spec_from_file_location('original_seamless200',OLD/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
native=old.native;read,save,sha=native.read,native.save,native.sha
NEW=ROOT/'experiments/p2_planning_sft_continue2epochs20_20261113/training'

def prepare():
 if (WORK/'PREPARED.json').exists():verify();return
 p=copy.deepcopy(read(OLD/'protocol.json'));assert sha(native.__file__)==p['native_entry_sha256']
 assert sha(OLD/'samples.json')==p['samples_sha256'];samples=read(OLD/'samples.json');assert len(samples)==200
 for s in samples:
  for f in s['files'].values():assert sha(OLD/f['path'])==f['sha256']
 for arm in p['arms']:
  folder=OLD/'results'/arm['name'];assert read(folder/'COMPLETE.json')['success']
  for s in samples:
   result=read(folder/s['id']/'original.json');assert result['arm']==arm['name'] and result['sample_id']==s['id'] and result['ended_with_eos'] and not result['target_in_prompt']
  dest=WORK/'results'/arm['name'];dest.parent.mkdir(exist_ok=True)
  if not dest.exists():dest.symlink_to(folder,target_is_directory=True)
 original_config=read(Path(p['arms'][-1]['adapter'])/'adapter_config.json')
 for step in (7500,8000,8256):
  folder=NEW/f'checkpoint-{step}';assert read(folder/'trainer_state.json')['global_step']==step
  assert read(folder/'adapter_config.json')==original_config
  p['arms'].append(dict(name=f'sft{step}',step=step,adapter=str(folder),adapter_sha256=sha(folder/'adapter_model.safetensors'),adapter_config_sha256=sha(folder/'adapter_config.json')))
 if not (WORK/'media').exists():(WORK/'media').symlink_to(OLD/'media',target_is_directory=True)
 (WORK/'samples.json').write_bytes((OLD/'samples.json').read_bytes())
 p.update(schema='qwen-ten-checkpoints-fixed-seamless200-v1',at=time.time(),total_outputs=2000,added_outputs=600,reused_outputs=1400,physical_gpus=[0,1,2],slots=3,entry_sha256=sha(__file__),report_entry_sha256=sha(OLD/'run.py'),config_sha256=sha(native.CFG),reuse_source=str(OLD))
 p.pop('physical_gpu',None);save(WORK/'protocol.json',p);save(WORK/'PREPARED.json',dict(at=time.time(),success=True,source_media_verified=200,reused_outputs=1400,new_outputs=600,adapter_configs_equal=True,old_results_unmodified=True))

def verify():
 p=read(WORK/'protocol.json');assert sha(__file__)==p['entry_sha256'] and sha(native.__file__)==p['native_entry_sha256'] and sha(OLD/'run.py')==p['report_entry_sha256'] and sha(native.CFG)==p['config_sha256']
 assert sha(WORK/'samples.json')==p['samples_sha256']
 for a in p['arms']:
  if a['adapter']:assert sha(Path(a['adapter'])/'adapter_model.safetensors')==a['adapter_sha256'] and sha(Path(a['adapter'])/'adapter_config.json')==a['adapter_config_sha256']
 return p

def worker(slot):
 p=verify();assert slot in (0,1,2) and os.environ['CUDA_VISIBLE_DEVICES']==str(slot)
 assert '172.29.252.37' in subprocess.check_output(['hostname','-I'],text=True).split()
 uuid=subprocess.check_output(['nvidia-smi','-i',str(slot),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 save(WORK/f'DEVICE_{slot}.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),physical_gpu=slot,uuid=uuid))
 native.WORK=WORK;native.generate(slot,slots=3)

def report():
 p=verify();old.WORK=WORK;old.report()
 path=WORK/'index.html';page=path.read_text().replace('七组Qwen','十组Qwen').replace('七组Qwen','十组Qwen').replace('原模型和SFT500／1000／1500／2000／2500／2752。','原模型和SFT500／1000／1500／2000／2500／2752／7500／8000／8256。');path.write_text(page)
 # Original presentation helper hardcodes seven arms; correct only completion bookkeeping.
 c=read(WORK/'COMPLETE.json');c.update(outputs=2000,reused_outputs=1400,new_outputs=600,physical_gpus=[0,1,2]);c.pop('physical_gpu',None);save(WORK/'COMPLETE.json',c)
 samples=read(WORK/'samples.json');arms=[a['name'] for a in p['arms']];texts={a:[read(WORK/'results'/a/s['id']/'original.json')['text'] for s in samples] for a in arms}
 comparisons=[]
 for a,b in [('sft2752','sft7500'),('sft2752','sft8000'),('sft2752','sft8256'),('sft7500','sft8000'),('sft7500','sft8256'),('sft8000','sft8256')]:
  sims=[difflib.SequenceMatcher(None,x.split(),y.split(),autojunk=False).ratio() for x,y in zip(texts[a],texts[b])]
  comparisons.append(dict(a=a,b=b,exact_same=sum(x==y for x,y in zip(texts[a],texts[b])),token_edit_similarity_mean=sum(sims)/200,scope='Surface similarity only, not semantic correctness',lowest_similarity=[dict(index=i+1,id=samples[i]['id'],similarity=sims[i]) for i in sorted(range(200),key=lambda i:sims[i])[:10]]))
 save(WORK/'TEXT_COMPARISONS.json',dict(at=time.time(),pairs=comparisons))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','report']);ap.add_argument('--slot',type=int);a=ap.parse_args()
 try:
  if a.action=='prepare':prepare()
  elif a.action=='worker':worker(a.slot)
  else:report()
 except BaseException:
  save(WORK/f'FAILURE_{a.action}_{a.slot}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
