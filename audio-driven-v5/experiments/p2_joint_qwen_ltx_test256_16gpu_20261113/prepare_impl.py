"""Frozen nested256 scope and exact per-item/per-shard reuse from the completed64."""
import copy,json,time,shutil
from pathlib import Path

def prepare(m):
 w=m.WORK;read,save,sha,link=m.read,m.save,m.sha,m.link
 if (w/'PREPARED.json').exists():m.verify();return
 from p2_testsets_eval.runtime import ready
 from p2_testsets_eval.protocol import rows
 prior=m.ROOT/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113'
 selected=m.ROOT/'experiments/seamless_test_tuning256_20261113';prepared=selected/'prepared_256'
 assert read(prior/'COMPLETE.json')['success'] and read(selected/'COMPLETE.json')['success']
 refs=rows(prepared/'references.jsonl');olditems=read(prior/'shared_items.json');oldkeys={x['key'] for x in olditems}
 full=m.ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112';byid={x['id']:x for x in read(full/'shared_items.json')}
 items=[]
 for ref in refs:
  x=copy.deepcopy(byid[ref['id']]);assert x['reference']['gt_reference']['row_sha256']==ref['gt_reference']['row_sha256'];x['reference']=ref;items.append(x)
 assert len(items)==len({x['key'] for x in items})==256 and oldkeys<={x['key'] for x in items}
 for x in olditems:
  cur=next(r for r in items if r['key']==x['key']);assert cur['geometry']==x['geometry']
 added=[x for x in items if x['key'] not in oldkeys];assert len(added)==192
 save(w/'shared_items.json',items);link(prior/'DATA_CONFIG.json',w/'DATA_CONFIG.json')
 c=read(prior/'cluster.json');c.update(training_script=str(m.SELF),purpose='Four approved Qwen/LTX candidates on nested test256; node8/node12 only');m.slots(c);save(w/'cluster.json',c)
 combos=[];reuse=[];cache_counts={};normal=read(prior/'PLAN.json');oldplans={c['key']:c for c in normal['combinations']}
 for q,step in [(8256,6800),(7500,6800),(2752,6000),(0,6800)]:
  key=f'qwen{q}_ltx{step}';oldw=Path(oldplans[key]['work']);oldp=read(oldw/'protocol.json');oldarm=oldp['arms'][0]['name'];cd=m.cache_dir(q);cd.mkdir(parents=True,exist_ok=True)
  p=copy.deepcopy(oldp);p.update(schema='joint-qwen-ltx-test256-v1',datasets={'seamless':dict(prepared=str(prepared),references_sha256=sha(prepared/'references.jsonl'),count=256,face_padding=0)},cluster=str(w/'cluster.json'),metrics_cluster=str(w/'cluster.json'),inference_nodes=['node12','node8'],authorized_scope='node12/node8 GPU0-7 only',physical_gpus=list(range(8)),shared_items_manifest=str(w/'shared_items.json'),shared_items_sha256=sha(w/'shared_items.json'),total_samples=256,total_outputs=256,workers=10,sft_hidden_cache=f'inference-only-adapter:{q}',conditioning='Each Qwen own text/hidden/Gemma; fixed speaker audio and listener first frame',execution_order='Native cache ->14 render slots with dynamic per-combination load ->decode/score; pooled256 merge',selection_purpose='nested_test256_checkpoint_tuning',entry_sha256=sha(m.SELF),planner_truncation_policy=read(prior/'TRUNCATION_POLICY.json'))
  p['arms'][0]['name']=key;p['arms'][0]['planner']=p['planner']
  for stale in ['worker_cli','acceleration','comparison','expanded_to_32_at']:p.pop(stale,None)
  save(cd/'protocol.json',p);link(w/'DATA_CONFIG.json',cd/'DATA_CONFIG.json');link(oldw/'negative',cd/'negative')
  # Frozen full2752 conditions and Whisper caches exist for all256; new Qwens
  # only reuse their own completed64 conditions. No writes into a prior cache.
  reuse_count=0
  for x in items:
   source=(full if q==2752 else oldw)/'conditions'/x['key'];target=cd/'conditions'/x['key']
   if source.exists():
    assert ready(source/'qwen') and ready(source/'text') and read(source/'qwen/plan.json')['adapter']==p['planner_adapter_identity']
    link(source,target);reuse_count+=1
   baseline=full/'conditions'/x['key'];wp=full/'whisper'/x['key'];assert ready(wp) and ready(baseline/'qwen') and ready(baseline/'text')
   assert read(wp/'metadata.json')['source']['input_audit_sha256']==sha(baseline/'input_audit.json')
  cache_counts[q]=dict(reused=reuse_count,new=256-reuse_count)
  for i in range(4):
   part=olditems[16*i:16*(i+1)]+added[48*i:48*(i+1)];save(cd/'shards'/f'{i}.json',part)
   if all(ready(cd/'conditions'/x['key']/'qwen') and ready(cd/'conditions'/x['key']/'text') for x in part):
    for x in part:assert sha(cd/'conditions'/x['key']/'input_audit.json')==sha(full/'conditions'/x['key']/'input_audit.json')
    save(cd/f'CACHE_COMPLETE_{i}.json',dict(success=True,at=time.time(),items=[x['key'] for x in part],adapter=p['planner_adapter_identity'],source_audit_exact=True,reused=True))
  dest=m.child(key);dest.mkdir(parents=True,exist_ok=True);save(dest/'protocol.json',p)
  for name,source in [('conditions',cd/'conditions'),('negative',cd/'negative'),('whisper',full/'whisper'),('shared_items.json',w/'shared_items.json'),('DATA_CONFIG.json',w/'DATA_CONFIG.json')]:link(source,dest/name)
  for x in olditems:
   g=oldw/'generated'/oldarm/x['key'];assert ready(g);meta=read(g/'metadata.json')
   assert meta['step']==step and meta['checkpoint_sha256']==p['arms'][0]['original_sha256'] and meta['planning_adapter']==p['planner_adapter_identity'] and meta['seed']==137 and meta['generation']==p['generation']
   assert meta['export']['frames']==x['geometry']['frames'] and not meta['listener_future_input'] and not meta['target_description_access']
   link(g,dest/'generated'/key/x['key']);reuse.append(dict(key=key,id=x['id'],source=str(g),complete_sha256=sha(g/'complete.json')))
  # Preserve the four original16-pair feature blocks. New32-pair blocks are
  # ordered longest-first to allow scoring while later generation continues.
  for i in range(4):
   shard=read(oldw/'shards'/f'{i:02d}.json');lookup={x['key']:x for x in items};save(dest/'shards'/f'{i:02d}.json',[lookup[x['key']] for x in shard])
   src=oldw/'metric_shards'/oldarm/'seamless'/f'{i:02d}';assert (src/'features.npz').is_file()
   assert {r['id'] for r in rows(src/'pairs.jsonl')}=={x['id'] for x in shard}
   assert {r['id'] for r in rows(src/'audio/pairs.jsonl')}=={x['id'] for x in shard}
   link(src,dest/'metric_shards'/key/'seamless'/f'{i:02d}');link(oldw/'metric_manifests'/oldarm/'seamless'/f'{i:02d}.jsonl',dest/'metric_manifests'/key/'seamless'/f'{i:02d}.jsonl')
   save(dest/'progress/metrics'/f'{key}_{i:02d}.json',dict(status='complete',at=time.time(),reused_from=str(src),source_contract_sha256=sha(src/'contract.json'),source_features_sha256=sha(src/'features.npz')))
  order=sorted(added,key=lambda x:(-x['geometry']['native_frames'],x['key']))
  for i in range(6):save(dest/'shards'/f'{i+4:02d}.json',order[32*i:32*(i+1)])
  combos.append(dict(key=key,qwen=q,ltx=step,work=str(dest),reused=False,reused_pairs=64,new_pairs=192,source64=str(oldw),source64_results_sha256=sha(oldw/'RESULTS.json')))
 assert sum(v['new'] for v in cache_counts.values())==576
 save(w/'REUSE.json',dict(at=time.time(),media=reuse,conditions=cache_counts,old_metric_blocks=16,pooled_features_reused=True,shard_scores_not_averaged=True))
 save(w/'PLAN.json',dict(at=time.time(),combinations=combos,total_combinations=4,samples=256,total_outputs=1024,reused_outputs=256,new_outputs=768,cache_counts=cache_counts,seed=137,generation=normal['generation'],normalized_config=normal['normalized_config'],identity=normal['identity'],selection_sha256=sha(selected/'selection.jsonl'),old64_sha256=sha(prior/'shared_items.json'),metric_shards=10,render_budget=14,dynamic_worker_cap='max(4,ceil(14/unfinished_combinations))',planner_truncation_policy=read(prior/'TRUNCATION_POLICY.json'),scope='nested test256 tuning; no independent-test claim'))
 frozen=[m.SELF,w/'prepare_impl.py',w/'cache_recorded_truncation.py',m.PARENT/'run.py',m.GUARD,w/'cluster.json',w/'PLAN.json',w/'shared_items.json',w/'DATA_CONFIG.json',selected/'PROTOCOL.json',prepared/'references.jsonl',prepared/'seal.json']
 frozen += [m.ROOT/'src'/f for f in ['p2_testsets_eval/runtime.py','p2_all_lora_followups_eval_20261109.py','p2_j_sft_hidden.py','p2_plan_cache/encoding.py','p2_testsets_eval/cluster.py','p2_testsets_eval/metrics.py']]
 frozen += [m.cache_dir(q)/'protocol.json' for q in m.QWENS]+[m.child(x['key'])/'protocol.json' for x in combos]
 save(w/'PREPARED.json',dict(success=True,at=time.time(),files={str(p):sha(p) for p in frozen},physical_gpus=16,excluded_nodes=['node11','node4']))
 print('Prepared4x256;256 reused+768 new generations;576 new planner conditions;16 metric blocks reused',flush=True)
