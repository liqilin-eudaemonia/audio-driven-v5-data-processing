"""Expand the disclosed test64 majority screen to exact existing test256."""
import copy,time
from pathlib import Path
SELECTED=[('no_event',6500),('no_event',6573),('no_feature',5500),('no_feature',6643),('no_whisper',5500),('no_whisper',6643)]
METRICS=[('psnr',1),('ssim',1),('lpips',-1),('fid',-1),('fvd',-1),('emo_acc',1),('emo_div_distance',-1),('delta_resp',-1),('missed_response_rate',-1),('false_sound_clip_rate',-1)]
def value(r,k):
 if k=='emo_div_distance':return abs(r['metrics']['emo_div_generated']-r['metrics']['emo_div_gt'])
 if k in ('missed_response_rate','false_sound_clip_rate'):return r['audio']['activity']['acoustic'][k]
 return r['metrics'][k]
def compare(r,b):
 states={}
 for k,d in METRICS:
  delta=(value(r,k)-value(b,k))*d;states[k]='worse' if delta < -1e-10 else 'better' if delta>1e-10 else 'tie'
 return dict(metrics=states,worse=sum(s=='worse' for s in states.values()),better=sum(s=='better' for s in states.values()),tie=sum(s=='tie' for s in states.values()),visual_reaction_worse=sum(states[k]=='worse' for k,_ in METRICS[:8]))

def prepare(a):
 if (a.WORK/'PREPARED.json').exists():a.verify();return
 from p2_testsets_eval.runtime import ready
 source=a.ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113';old=a.ROOT/'experiments/p2_ablation12_test64_qwen7500_24gpu_20261113'
 assert a.read(old/'COMPLETE.json')['success'] and a.read(source/'COMPLETE.json')['success']
 full64=a.ROOT/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113/combinations/qwen7500_ltx6800'
 base=next(iter(a.read(full64/'RESULTS.json').values()));scores=a.read(old/'RESULTS.json')['results'];screen={k:compare(v['scores'],base) for k,v in scores.items()}
 selected=[f'{arm}_step{step}_qwen7500' for arm,step in SELECTED]
 assert all(screen[k]['worse']>=6 and screen[k]['visual_reaction_worse']>=5 for k in selected)
 a.save(a.WORK/'SELECTION.json',dict(at=time.time(),reference='Qwen7500+LTX6800, identical64,seed137',reference_results=str(full64/'RESULTS.json'),all12_results=str(old/'RESULTS.json'),metrics=METRICS,criterion='strict majority numerically worse: >=6 of10, also >=5 of8 visual/reaction; ties not counted as worse',selected=selected,all12=screen,eligible=[k for k,v in screen.items() if v['worse']>=6],policy='Retain five prior candidates that satisfy the clarified majority rule; replace no_event6000 with no_event6500. Keep all12 results. GLARE comparison deferred by user.',interpretation='Outcome-conditioned exploratory test-subset screening, not a significance test or independent validation. Do not claim components improve all metrics. No guarantee of same order at256.'))
 items=a.read(source/'shared_items.json');olditems=a.read(old/'shared_items.json');byid={x['id']:x for x in items};assert len(items)==256 and len(olditems)==64
 for x in olditems:
  y=byid[x['id']];assert x['key']==y['key'] and x['geometry']==y['geometry']
  aa=x['reference']['gt_reference'];bb=y['reference']['gt_reference']
  assert {k:v for k,v in aa.items() if k not in ['manifest','crop_config']}=={k:v for k,v in bb.items() if k not in ['manifest','crop_config']}
  assert {k:v for k,v in x['reference'].items() if k!='gt_reference'}=={k:v for k,v in y['reference'].items() if k!='gt_reference'}
 c=a.read(old/'cluster.json');c['nodes']=[n for n in c['nodes'] if n['name'] in ['node12','node11']];c.update(training_script=str(a.SELF),purpose='6 ablation checkpoints selected by disclosed majority rule; Qwen7500,test256,seed137; node8 excluded')
 a.save(a.WORK/'cluster.json',c);a.slots(c)
 a.link(source/'shared_items.json',a.WORK/'shared_items.json');a.link(source/'DATA_CONFIG.json',a.WORK/'DATA_CONFIG.json')
 full=source/'combinations/qwen7500_ltx6800';protocol=a.read(full/'protocol.json');cache=a.cache_dir(7500)
 assert protocol['seed']==137 and protocol['planner_adapter_identity']==a.adapter(7500)
 def same_file(x,y):return Path(x).resolve()==Path(y).resolve() or a.sha(x)==a.sha(y)
 for x in items:
  folder=cache/'conditions'/x['key'];plan=a.read(folder/'qwen/plan.json')
  assert ready(folder/'qwen') and ready(folder/'text') and plan['adapter']==a.adapter(7500) and not plan['target_description_access']
 for x in olditems:
  new=cache/'conditions'/x['key'];prev=full64/'conditions'/x['key']
  for relative in ['qwen/plan.json','qwen/hidden.pt','text/payload.pt','first_frame.png','input_audit.json']:assert same_file(new/relative,prev/relative),(x['id'],relative)
 oldplan=a.read(old/'PLAN.json');oldchecks={r['key']:r for r in a.read(old/'CHECKPOINTS.json')};combos=[];checks=[];reuse=[]
 for component,step in SELECTED:
  key=f'{component}_step{step}_qwen7500';w=a.child(key);w.mkdir(parents=True,exist_ok=True);prev=old/'combinations'/key
  rec=oldchecks[key];assert a.base.stamp(rec['path'])==rec['stamp'];proof=old/'integrity'/f'{key}.json';assert a.read(proof)['sha256']==rec['sha256']
  a.link(proof,a.WORK/'integrity'/f'{key}.json');checks.append(rec)
  oldp=a.read(prev/'protocol.json');p=copy.deepcopy(protocol)
  p.update(schema='component-ablation-qwen7500-test256-v1',arms=oldp['arms'],component=component,component_flags=oldp['component_flags'],checkpoint_identity=oldp['checkpoint_identity'],workers=10,total_samples=256,total_outputs=256,cluster=str(a.WORK/'cluster.json'),metrics_cluster=str(a.WORK/'cluster.json'),inference_nodes=['node12','node11'],authorized_scope='node12/node11 GPU0-7 only; node8 excluded',physical_gpus=list(range(8)),shared_item_queue=True,shared_items_manifest=str(a.WORK/'shared_items.json'),shared_items_sha256=a.sha(a.WORK/'shared_items.json'),selection_purpose='outcome_conditioned_test_checkpoint_screening_not_independent_dev',entry_sha256=a.sha(a.SELF),reused_generated_ids=[x['id'] for x in olditems])
  for name,src in [('conditions',cache/'conditions'),('negative',full/'negative'),('shared_items.json',a.WORK/'shared_items.json'),('DATA_CONFIG.json',a.WORK/'DATA_CONFIG.json')]:a.link(src,w/name)
  if component!='no_whisper':
   a.link(full/'whisper',w/'whisper')
   for x in olditems:assert same_file(w/'whisper'/x['key']/'payload.pt',prev/'whisper'/x['key']/'payload.pt')
  for x in olditems:
   f=prev/'generated'/key/x['key'];assert ready(f);meta=a.read(f/'metadata.json')
   assert meta['seed']==137 and meta['step']==step and meta['checkpoint_sha256']==rec['sha256'] and meta['planning_adapter']==p['planner_adapter_identity'] and meta['generation']==p['generation']
   assert meta['geometry']==byid[x['id']]['geometry'] and not meta['listener_future_input'] and not meta['target_description_access']
   a.link(f,w/'generated'/key/x['key']);reuse.append(dict(key=key,id=x['id'],receipt=str(f/'complete.json'),receipt_sha256=a.sha(f/'complete.json')))
  a.save(w/'protocol.json',p)
  for i in range(10):a.save(w/'shards'/f'{i:02d}.json',a.read(full/'shards'/f'{i:02d}.json'))
  combos.append(dict(key=key,qwen=7500,ltx=step,component=component,work=str(w),reused=False,reused_outputs=64))
 a.save(a.WORK/'CHECKPOINTS.json',checks);a.save(a.WORK/'REUSED_OUTPUTS.json',reuse)
 a.save(a.WORK/'PLAN.json',dict(at=time.time(),combinations=combos,identities=oldplan['identities'],total_combinations=6,total_outputs=1536,new_outputs=1152,reused_outputs=384,samples=256,qwen=7500,seed=137,generation=protocol['generation'],physical_gpus=16,nodes=['node12','node11'],excluded_nodes=['node8'],scope='test checkpoint screening; retain all results; GLARE comparison deferred',selection=str(a.WORK/'SELECTION.json'),new_qwen_extractions=0,scheduler='Dynamic shares across6 remaining models; interleaved native generation/decode/scoring; per-GPU single owned worker'))
 a.save(a.WORK/'INTEGRITY_COMPLETE.json',dict(success=True,at=time.time(),checkpoints=6,verified_original_SHA_reused_after_exact_stat_check=True,upstream=str(old/'INTEGRITY_COMPLETE.json')))
 frozen=[a.SELF,a.WORK/'preparation.py',a.WORK/'model_loading.py',a.WORK/'render.py',a.WORK/'cluster.json',a.WORK/'PLAN.json',a.WORK/'CHECKPOINTS.json',a.WORK/'SELECTION.json',a.WORK/'REUSED_OUTPUTS.json',a.WORK/'shared_items.json',a.WORK/'DATA_CONFIG.json',old/'PLAN.json',old/'PREPARED.json',old/'RESULTS.json',full64/'RESULTS.json',a.PARENT/'run.py',a.GUARD]
 frozen += [a.child(k)/'protocol.json' for k in selected]
 a.save(a.WORK/'PREPARED.json',dict(success=True,at=time.time(),files={str(p):a.sha(p) for p in frozen},physical_gpus=16,nodes=['node12','node11'],reused=384,new=1152))
 print('Prepared6x256:384 exact reusable outputs+1152 new. All256 Qwen7500 inputs cached.',flush=True)
