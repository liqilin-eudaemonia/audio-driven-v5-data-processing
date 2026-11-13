"""Second generation seed: same256 and same conditions; no old media/scores reused."""
import copy,time
from pathlib import Path

def prepare(m):
 w=m.WORK;read,save,sha,link=m.read,m.save,m.sha,m.link
 if (w/'PREPARED.json').exists():m.verify();return
 source=m.ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113'
 prior=read(source/'PLAN.json');assert prior['seed']==137 and prior['samples']==256 and prior['total_combinations']==4
 items=read(source/'shared_items.json');assert len(items)==256
 for name in ['shared_items.json','DATA_CONFIG.json']:link(source/name,w/name)
 c=read(source/'cluster.json');c.update(training_script=str(m.SELF),purpose='Same four nested256 combinations; second AV generation seed238 after seed137 completes');m.slots(c);save(w/'cluster.json',c)
 combos=[]
 for record in prior['combinations']:
  key=record['key'];q=record['qwen'];old=Path(record['work']);dest=m.child(key);dest.mkdir(parents=True,exist_ok=True)
  p=copy.deepcopy(read(old/'protocol.json'));assert p['seed']==137
  p.update(seed=238,cluster=str(w/'cluster.json'),metrics_cluster=str(w/'cluster.json'),shared_items_manifest=str(w/'shared_items.json'),shared_items_sha256=sha(w/'shared_items.json'),entry_sha256=sha(m.SELF),schema='joint-qwen-ltx-test256-second-seed-v1',execution_order='Reuse exact deterministic planner conditions; fresh seed238 render/decode/scoring; full256 pooled metric merge',paired_seed_experiment=dict(first_seed=137,second_seed=238,first_work=str(source),same_conditions=True,same_samples=True,no_media_or_metric_reuse=True))
  save(dest/'protocol.json',p)
  cd=m.cache_dir(q);cd.mkdir(parents=True,exist_ok=True);save(cd/'protocol.json',p)
  for name in ['conditions','negative','DATA_CONFIG.json','shards']:
   link(source/'conditioning'/f'qwen{q}'/name,cd/name)
  for i in range(4):
   receipt=source/'conditioning'/f'qwen{q}'/f'CACHE_COMPLETE_{i}.json';assert read(receipt)['success'];link(receipt,cd/receipt.name)
  for name in ['conditions','negative','whisper','shards']:
   link(old/name,dest/name)
  for name in ['shared_items.json','DATA_CONFIG.json']:link(w/name,dest/name)
  combos.append(dict(key=key,qwen=q,ltx=record['ltx'],work=str(dest),reused=False,reused_pairs=0,new_pairs=256,conditions_from=str(old),first_seed_work=str(old)))
 plan=copy.deepcopy(prior);plan.update(at=time.time(),combinations=combos,seed=238,reused_outputs=0,new_outputs=1024,cache_counts={str(q):dict(reused=256,new=0) for q in m.QWENS},first_seed=137,first_seed_work=str(source),first_plan_sha256=sha(source/'PLAN.json'),no_old_media_or_metrics=True)
 save(w/'PLAN.json',plan)
 frozen=[m.SELF,w/'prepare_impl.py',w/'cache_recorded_truncation.py',m.PARENT/'run.py',m.GUARD,w/'cluster.json',w/'PLAN.json',w/'shared_items.json',w/'DATA_CONFIG.json',source/'PLAN.json']
 frozen += [m.ROOT/'src'/f for f in ['p2_testsets_eval/runtime.py','p2_all_lora_followups_eval_20261109.py','p2_j_sft_hidden.py','p2_plan_cache/encoding.py','p2_testsets_eval/cluster.py','p2_testsets_eval/metrics.py']]
 frozen += [m.cache_dir(q)/'protocol.json' for q in m.QWENS]+[m.child(x['key'])/'protocol.json' for x in combos]
 save(w/'PREPARED.json',dict(success=True,at=time.time(),files={str(p):sha(p) for p in frozen},physical_gpus=16,excluded_nodes=['node11','node4']))
 print('Prepared seed238:4x256 fresh generation/scoring;exact1024 conditions reused;await first-seed completion',flush=True)
