"""Freeze existing checkpoints and reuse the accepted Qwen7500 test64 inputs."""
import copy,importlib.util,json,time,traceback
from pathlib import Path

def runtime(root):
 p=root/'experiments/p2_all_lora_ablations28_20261111/runtime.py'
 s=importlib.util.spec_from_file_location('ablation_training_runtime',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(a):
 if (a.WORK/'PREPARED.json').exists():a.verify();return
 import torch
 torch.set_num_threads(2)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 from p2_testsets_eval.runtime import ready
 from p2_current_component_ablation.config import flags
 prior=a.ROOT/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113'
 full=prior/'combinations/qwen7500_ltx6800'
 items=a.read(prior/'shared_items.json');assert len(items)==64
 c=a.read(a.ROOT/'experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/cluster.json');c.update(training_script=str(a.SELF),purpose='12 ablation checkpoints / Qwen7500 / fixed64 / three nodes24 GPUs')
 a.save(a.WORK/'cluster.json',c);a.slots(c)
 a.link(prior/'shared_items.json',a.WORK/'shared_items.json');a.link(prior/'DATA_CONFIG.json',a.WORK/'DATA_CONFIG.json')
 protocol=a.read(full/'protocol.json');cache=a.cache_dir(7500);proof=[]
 assert protocol['seed']==137 and protocol['planner_adapter_identity']==a.adapter(7500)
 for x in items:
  f=cache/'conditions'/x['key'];q=a.read(f/'qwen/plan.json')
  assert ready(f/'qwen') and ready(f/'text') and q['adapter']==a.adapter(7500) and not q['target_description_access']
  audit=a.read(f/'input_audit.json');assert audit['source_only'] and not audit['listener_future_media_in_model_input']
  proof.append(dict(key=x['key'],plan=a.sha(f/'qwen/plan.json'),qwen=a.sha(f/'qwen/complete.json'),text=a.sha(f/'text/complete.json'),input=a.sha(f/'input_audit.json')))
 a.save(a.WORK/'CONDITIONS.json',proof)
 r=runtime(a.ROOT);combinations=[];identities={};checkpoints=[]
 for component,last in [('no_whisper',6643),('no_feature',6643),('no_event',6573)]:
  output=r.output(component);identity=a.read(output/'run.json');cfg=identity['experiment']
  rr=r.noevent_runtime() if component=='no_event' else r
  rr.validate(cfg);normal=rr.normalized(cfg)
  current=rr.source();assert current==identity['source'],(component,'training source drift')
  identities[component]=dict(identity=identity,normalized_config=normal)
  for step in [5500,6000,6500,last]:
   path=output/f'step_{step:06d}.pt';marker=a.read(path.with_suffix('.complete.json'));stamp=a.base.stamp(path)
   v=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
   assert v['step']==marker['step']==step and v['identity']==identity and len(v['rank_rng'])==28
   assert identity['experiment']['component_arm']==component
   assert len(identity['text_targets'])==1632 and a.base.stamp(path)==stamp
   key=f'{component}_step{step}_qwen7500';w=a.child(key);w.mkdir(parents=True,exist_ok=True)
   rec=dict(key=key,component=component,step=step,path=str(path),sha256=marker['sha256'],stamp=stamp,schema=v['schema'],model_tensors=len(v['model']))
   checkpoints.append(rec);del v
   p=copy.deepcopy(protocol)
   arm=dict(name=key,kind='ltx_all_lora_'+component,step=step,checkpoint=str(path),original_checkpoint=str(path),checkpoint_sha256=marker['sha256'],original_sha256=marker['sha256'],planner=p['planner'])
   p.update(schema='component-ablation-qwen7500-test64-v1',arms=[arm],component=component,checkpoint_identity=identity,component_flags=flags(normal),workers=4,total_samples=64,total_outputs=64,cluster=str(a.WORK/'cluster.json'),metrics_cluster=str(a.WORK/'cluster.json'),inference_nodes=['node12','node8','node11'],authorized_scope='node12/node8/node11 GPU0-7',physical_gpus=list(range(8)),shared_item_queue=True,shared_items_manifest=str(a.WORK/'shared_items.json'),shared_items_sha256=a.sha(a.WORK/'shared_items.json'),selection_purpose='test_checkpoint_tuning_not_independent_dev',entry_sha256=a.sha(a.SELF),metrics_decoded_gt_cache=str(a.PARENT/'decoded_gt'))
   for name,src in [('conditions',cache/'conditions'),('negative',full/'negative'),('shared_items.json',a.WORK/'shared_items.json'),('DATA_CONFIG.json',a.WORK/'DATA_CONFIG.json')]:a.link(src,w/name)
   if component!='no_whisper':a.link(full/'whisper',w/'whisper')
   a.save(w/'protocol.json',p)
   for i in range(4):a.save(w/'shards'/f'{i:02d}.json',a.read(full/'shards'/f'{i:02d}.json'))
   combinations.append(dict(key=key,qwen=7500,ltx=step,component=component,work=str(w),reused=False))
 a.save(a.WORK/'CHECKPOINTS.json',checkpoints)
 a.save(a.WORK/'PLAN.json',dict(at=time.time(),combinations=combinations,identities=identities,total_combinations=12,total_outputs=768,samples=64,qwen=7500,seed=137,generation=protocol['generation'],physical_gpus=24,scope='Fixed test64 exploratory checkpoint screening. Retain all results. GLARE selection and256 advancement pending user confirmation.',cache_reused=64,checkpoint_integrity='Sequential CPU SHA verification, per-checkpoint admission; original native strict model loading; no duplicate weight exports',scheduler='At most2 render workers/checkpoint; native shared claims; decode/scoring overlap; admit by free memory, stop no unrelated jobs'))
 frozen=[a.SELF,a.WORK/'preparation.py',a.WORK/'model_loading.py',a.WORK/'render.py',a.WORK/'cluster.json',a.WORK/'PLAN.json',a.WORK/'CHECKPOINTS.json',a.WORK/'CONDITIONS.json',a.WORK/'shared_items.json',a.WORK/'DATA_CONFIG.json',a.PARENT/'run.py',a.ROOT/'src/p2_testsets_eval/runtime.py',a.GUARD]
 frozen += [a.child(x['key'])/'protocol.json' for x in combinations]
 a.save(a.WORK/'PREPARED.json',dict(success=True,at=time.time(),files={str(p):a.sha(p) for p in frozen},physical_gpus=24,nodes=['node12','node8','node11'],new_gpu_runs=0))
 print('Prepared12 checkpoints,768 outputs,Qwen7500 conditions reused64',flush=True)

def integrity(a):
 try:
  a.verify()
  for rec in a.read(a.WORK/'CHECKPOINTS.json'):
   if a.stopped():return
   dest=a.WORK/'integrity'/f'{rec["key"]}.json'
   if dest.exists():assert a.read(dest)['stamp']==a.base.stamp(rec['path']);continue
   assert a.base.stamp(rec['path'])==rec['stamp'];start=time.time()
   h=a.sha(rec['path']);assert h==rec['sha256'] and a.base.stamp(rec['path'])==rec['stamp']
   a.save(dest,dict(success=True,at=time.time(),seconds=time.time()-start,**rec));print('Verified',rec['key'],round(time.time()-start,1),flush=True)
  a.save(a.WORK/'INTEGRITY_COMPLETE.json',dict(success=True,at=time.time(),checkpoints=12))
 except BaseException:
  a.save(a.WORK/'INTEGRITY_FAILED.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
