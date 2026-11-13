"""CPU-only preparation: sealed full sets and strict compatible reuse."""
import copy,time
from concurrent.futures import ThreadPoolExecutor
from common import *

def prepare():
 if (BASE/'PREPARED.json').exists():return verify()
 from p2_testsets_eval.runtime import ready
 from p2_j_sft_hidden import adapter_identity
 template=read(SOURCE256[137]/'combinations'/ARM/'protocol.json');normal=read(SOURCE256[137]/'PLAN.json')
 assert adapter_identity(template['planner_adapter'])==template['planner_adapter_identity']
 check=next(x for x in read(TRUSTED/'CHECKPOINTS.json') if x['step']==6800)
 old=import_file('checked_checkpoint_header',TRUSTED/'run.py')
 assert old.stamp(check['path'])==check['stamp'] and read(TRUSTED/'integrity/006800.json')['sha256']==check['sha256']==template['arms'][0]['original_sha256']
 assert read(Path(check['path']).with_suffix('.complete.json'))['sha256']==check['sha256']
 save(BASE/'DATA_CONFIG.json',read(SOURCE256[137]/'DATA_CONFIG.json'))
 report={};frozen=[BASE/n for n in ['run.py','common.py','prepare.py','workers.py','cluster.json','DATA_CONFIG.json']]
 for path,digest in read(SOURCE256[137]/'PREPARED.json')['files'].items():
  assert sha(path)==digest,('Validated256 dependency changed',path);frozen.append(Path(path))
 frozen += [RECORDED_CACHE,TRUSTED/'run.py',TRUSTED/'CHECKPOINTS.json',TRUSTED/'PLAN.json',TRUSTED/'integrity/006800.json',GUARD]
 frozen += [ROOT/f for f in ['src/p2_all_lora_followups_eval_20261109.py','src/p2_testsets_eval/runtime.py','src/p2_testsets_eval/shared_items.py','src/p2_testsets_eval/whisper_runtime.py','src/p2_testsets_eval/cluster.py','src/p2_testsets_eval/metrics.py','src/p2_j_sft_hidden.py','src/p2_plan_cache/encoding.py','scripts/p2_pipelined_testsets_eval.py','scripts/p2_pipelined_video_metrics.py','scripts/p2_responsenet_random200.py']]
 for dataset,count in COUNTS.items():
  dw=dataset_work(dataset);cd=conditions(dataset);dw.mkdir(parents=True,exist_ok=True)
  source=INPUTS[dataset];sp=read(source/'protocol.json');spec=copy.deepcopy(sp['datasets'][dataset]);refs=rows(Path(spec['prepared'])/'references.jsonl')
  assert len(refs)==len({x['id'] for x in refs})==count and sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
  assert spec['face_padding']==(0 if dataset=='seamless' else 64)
  original={x['id']:x for x in read(source/'shared_items.json')};items=[copy.deepcopy(original[r['id']]) for r in refs]
  assert all(x['reference']==r for x,r in zip(items,refs))
  save(dw/'shared_items.json',items);frozen += [dw/'shared_items.json',Path(spec['prepared'])/'references.jsonl',Path(spec['prepared'])/'seal.json']
  small={x['key']:x for x in read(SOURCE256[137]/'shared_items.json')} if dataset=='seamless' else {}
  media_source=SOURCE256[137]/'combinations'/ARM
  def one(x):
   src=source/'conditions'/x['key'];dst=cd/'conditions'/x['key'];audit=read(src/'input_audit.json');wp=source/'whisper'/x['key']
   assert audit['source_only'] and not audit['target_description_in_model_input'] and not audit['listener_future_media_in_model_input']
   assert ready(wp);wm=read(wp/'metadata.json');assert wm['encoder']==template['whisper_encoder'] and wm['feature_code']==template['feature_code'] and wm['source']['input_audit_sha256']==sha(src/'input_audit.json')
   for n in ['first_frame.png','input_audit.json']:link(src/n,dst/n)
   reused=False
   if x['key'] in small:
    assert same_source(small[x['key']],x);own=media_source/'conditions'/x['key']
    assert sha(own/'input_audit.json')==sha(src/'input_audit.json') and sha(own/'first_frame.png')==sha(src/'first_frame.png')
    assert ready(own/'qwen') and ready(own/'text');q=read(own/'qwen/plan.json');assert q['adapter']==template['planner_adapter_identity'] and not q['target_description_access']
    for n in ['qwen','text']:link(own/n,dst/n)
    reused=True
   return dict(key=x['key'],reused_qwen7500=reused,source=str(src),image_sha256=sha(src/'first_frame.png'),audit_sha256=sha(src/'input_audit.json'),whisper_source=wm['source'])
  records=[]
  with ThreadPoolExecutor(4) as pool:
   for rec in pool.map(one,items):
    records.append(rec)
    if len(records)%256==0:save(BASE/'prepare_status.json',dict(dataset=dataset,checked=len(records),total=count,at=time.time()))
  save(dw/'INPUT_IDENTITIES.json',{x['key']:x for x in records});frozen.append(dw/'INPUT_IDENTITIES.json')
  link(source/'whisper',cd/'whisper');link(media_source/'negative',cd/'negative');link(BASE/'DATA_CONFIG.json',cd/'DATA_CONFIG.json')
  missing=[x for x in items if not ready(cd/'conditions'/x['key']/'text') or not ready(cd/'conditions'/x['key']/'qwen')]
  for i in range(24):save(cd/'shards'/f'{i:02d}.json',missing[i::24]);frozen.append(cd/'shards'/f'{i:02d}.json')
  base=copy.deepcopy(template)
  for stale in ['execution_code','adapter_code','reused_generated_ids','reused_condition_ids','selection_purpose','source_version','changed_from_historical_full_ids','planner_truncation_policy','metrics_decoded_gt_cache']:
   base.pop(stale,None)
  base.update(schema='qwen7500-ltx6800-full-dualseed-24gpu-v1',datasets={dataset:spec},total_samples=count,total_outputs=count,
   entry_sha256=sha(BASE/'run.py'),created_at=time.time(),cache_memory_gib=80,metrics_workers_per_gpu=1,metrics_inference_workers_per_gpu=1,
   cluster=str(BASE/'cluster.json'),metrics_cluster=str(BASE/'cluster.json'),worker_cli=str(BASE/'run.py'),
   inference_nodes=['node12','node8','node11'],physical_gpus=list(range(8)),authorized_scope='node12/node8/node11 GPU0-7; prepare-only until explicit run',
   shared_item_queue=True,shared_items_manifest=str(dw/'shared_items.json'),shared_items_sha256=sha(dw/'shared_items.json'),
   sft_hidden_cache='inference-only-adapter:7500',inference_qwen_hidden_source='SFT7500',
   conditioning='Own Qwen7500 planning/hidden/Gemma, same speaker Whisper and listener first frame; shared between seeds',
   execution_order='All four inference/export stages first; scoring is a separate manual action; no dual-seed aggregation',
   planner_truncation_policy=read(SOURCE256[137]/'PLAN.json')['planner_truncation_policy'])
  reused_outputs={}
  for seed in SEEDS:
   w=work(dataset,seed);w.mkdir(exist_ok=True);p=copy.deepcopy(base);p['seed']=seed
   for n,src in [('shared_items.json',dw/'shared_items.json'),('DATA_CONFIG.json',BASE/'DATA_CONFIG.json'),('conditions',cd/'conditions'),('whisper',cd/'whisper'),('negative',cd/'negative')]:link(src,w/n)
   shards=[];reused=[]
   if dataset=='seamless':
    sw=SOURCE256[seed]/'combinations'/ARM;prior=read(sw/'protocol.json');assert prior['seed']==seed and prior['generation']==p['generation'];bykey={x['key']:x for x in items}
    for i in range(10):
     part=[bykey[x['key']] for x in read(sw/'shards'/f'{i:02d}.json')];shards.append(part)
     out=sw/'metric_shards'/ARM/dataset/f'{i:02d}';assert {x['id'] for x in rows(out/'pairs.jsonl')}=={x['id'] for x in part}
     assert {x['id'] for x in rows(out/'audio/pairs.jsonl')}=={x['id'] for x in part} and (out/'features.npz').is_file()
     link(out,w/'metric_shards'/ARM/dataset/f'{i:02d}');link(sw/'metric_manifests'/ARM/dataset/f'{i:02d}.jsonl',w/'metric_manifests'/ARM/dataset/f'{i:02d}.jsonl')
     save(w/'progress/metrics'/f'{ARM}_{i:02d}.json',dict(status='complete',at=time.time(),reused_from=str(out)))
     for x in part:
      g=sw/'generated'/ARM/x['key'];assert ready(g);assert valid_media(read(g/'metadata.json'),p,x,sha(cd/'conditions'/x['key']/'qwen/plan.json'))
      link(g,w/'generated'/ARM/x['key']);reused.append(x['key'])
   pending=sorted([x for x in items if x['key'] not in set(reused)],key=lambda x:(-x['geometry']['native_frames'],x['key']))
   shards += [pending[i:i+32] for i in range(0,len(pending),32)]
   assert sum(map(len,shards))==count and len({x['key'] for sh in shards for x in sh})==count
   p['workers']=len(shards);save(w/'protocol.json',p);frozen.append(w/'protocol.json')
   for i,part in enumerate(shards):save(w/'shards'/f'{i:02d}.json',part);frozen.append(w/'shards'/f'{i:02d}.json')
   reused_outputs[seed]=len(reused)
  cp=copy.deepcopy(base);cp.update(seed=137,workers=24);save(cd/'protocol.json',cp);frozen.append(cd/'protocol.json')
  report[dataset]=dict(samples=count,cached_qwen7500=count-len(missing),new_qwen7500=len(missing),whisper_reused=count,reused_outputs=reused_outputs,scoring_shards=len(shards),fps_counts={str(f):sum(x['geometry']['fps']==f for x in items) for f in sorted({x['geometry']['fps'] for x in items})})
 save(BASE/'PLAN.json',dict(at=time.time(),datasets=report,stage_order=[dict(dataset=d,seed=s) for d,s in ORDER],qwen=7500,ltx=6800,
  identity=normal['identity'],normalized_config=normal['normalized_config'],generation=template['generation'],
  default_action='inference_and_export_only',automatic_scoring=False,automatic_dualseed_aggregation=False,physical_gpus=24,
  total_outputs=2*sum(COUNTS.values()),new_outputs=2*sum(COUNTS.values())-512,cache_workers=24,render_workers=22,decode_workers=2,
  checkpoint=check,existing_test_tuning_overlap='Seamless256 used for checkpoint selection; disclose in full-test reporting'))
 frozen.append(BASE/'PLAN.json')
 save(BASE/'PREPARED.json',dict(success=True,at=time.time(),files={str(f):sha(f) for f in frozen},prepared_only=True,gpu_jobs_started=False,scope=report))
 return verify()
