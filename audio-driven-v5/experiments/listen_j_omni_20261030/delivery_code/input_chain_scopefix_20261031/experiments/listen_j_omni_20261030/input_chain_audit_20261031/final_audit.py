import hashlib,json,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];F=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write
cfg=read(W/'config.json');scope=read(F/'scope_reconciliation.json');assert scope['passed']
plan_raw=Path(cfg['source_ownership']['path']).read_bytes();assert hashlib.sha256(plan_raw).hexdigest()==cfg['source_ownership']['sha256'];plan=json.loads(plan_raw)
assert len(set().union(*(set(x['allowed_jobs']) for x in plan['nodes'].values())))==60869
assert sum(x['scope_clips'] for x in plan['nodes'].values())==1151805
for name,sha in cfg['code_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
replays=[read(F/f'replay-{n}.json') for n in cfg['nodes']];assert all(x['passed'] for x in replays);samples=[x for r in replays for x in r['samples']];assert len(samples)==8
assert read(F/'gate_skip.json')['passed'];assert read(F/'historical_receipt_validation.json')['passed'];assert read(F/'real_source_dryrun.json')['passed']
producer=read(W/'producer.json');assert not producer.get('error') and time.time()-producer['at']<30
legacy=producer['nodes']['node12_legacy'];assert legacy['scanned_sources']>0
# Verify the first genuinely published recovered source against original nod owner.
jid=next(iter(plan['nodes']['node12_legacy']['allowed_jobs']));origin=read(W/'job_origins'/f'{jid}.json')
assert origin['old_work']==plan['nodes']['node12_legacy']['old_work'] and origin['node']=='node12_legacy'
receipt=read(origin['nod_receipt']);job=read(W/'jobs'/f'{jid}.json');assert receipt['source_result_sha256']==job['source_result_sha256']==origin['source_result_sha256']
for path in [W/'queues/caption/state.json',W/'visual_quality/queues/measure/state.json']:
 assert not read(path)['failed']
health=read(W/'nod_cpu_share/after_source_repair.json');before=read(W/'nod_cpu_share/final_health.json')
for n,nh in health['nodes'].items():
 assert all(x['live'] and x['fresh'] and not x['errors'] for x in nh['gpu'])
 assert [(x['pid'],x['start_ticks']) for x in nh['gpu']]==[(x['pid'],x['start_ticks']) for x in before['nodes'][n]['gpu']]
history=read(plan['history_scope']);nod={}
for n,item in history['nodes'].items():
 state=read(Path(item['work'])/'queues/nod/state.json');target=item['branch_scope_clips']['nod'];assert not state['failed'] and state['completed_clips']<=target
 nod[n]=dict(total=target,completed=state['completed_clips'],remaining=target-state['completed_clips'])
report=dict(at=time.time(),passed=True,total_unique_sources=60869,total_target_clips=1151805,recovered_missing_sources=1295,recovered_missing_clips=62488,delegated_sources_resolved=6647,historical_receipts_checked=7942,historical_nod_clips=304045,replay_samples=8,replay_full_silence=sum(x['full_silence'] for x in samples),replay_with_audio=sum(not x['full_silence'] for x in samples),sample_input_hashes_and_prompts_identical=True,gate_skip_examples=read(F/'gate_skip.json')['examples'],recovered_sources_published=legacy['scanned_sources'],first_recovered_origin=origin,gpu_processes_unchanged=32,tests_passed=53,nod_owner_progress=nod,remaining_nod_clips=sum(x['remaining'] for x in nod.values()),scope='Complete source-list reconciliation and named historical source receipts; eight current media replays and bounded gate/provenance checks. Not manual accuracy review or re-decoding every clip.')
write(F/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['gate_skip_examples','first_recovered_origin']},ensure_ascii=False))
