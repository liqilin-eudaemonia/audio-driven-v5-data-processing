import hashlib,json,time,sys
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
from listen_visual_quality.io import canonical_hash
c=read(W/'config.json');cfg=read(W/'nod_cpu_share/config.json');old=Path(cfg['source_work']);h=read(W/'nod_cpu_share/final_health.json');base=read(W/'nod_cpu_share/canary_start.json')
checks=[];samples=[]
previous=read(W/'nod_cpu_share/audit.json').get('production_samples',[]) if (W/'nod_cpu_share/audit.json').exists() else []
assert len(h['nodes']['node17']['original_nod'])==48
assert all(x['live'] and x['fresh'] and not x['errors'] for x in h['nodes']['node17']['original_nod'])
for n,nh in h['nodes'].items():
 assert all(x['live'] and x['fresh'] and not x['errors'] for x in nh['gpu'])
 assert [(x['pid'],x['start_ticks']) for x in nh['gpu']]==[(x['pid'],x['start_ticks']) for x in base['nodes'][n]['gpu']]
 assert nh['quality']['live'] and nh['quality']['age']<60 and nh['quality']['stage']=='running' and not nh['quality']['errors']
 assert nh['quality']['workers']==c['nodes'][n]['visual_quality_workers']
 if n not in cfg['nodes']:continue
 assert nh['supervisor']['live'] and nh['supervisor']['age']<30 and not nh['supervisor']['blocked']
 assert len(nh['workers'])==cfg['nodes'][n]['workers']
 assert all(x['live'] and x['fresh'] and not x['errors'] for x in nh['workers'])
 assert sum(x['completed'] for x in nh['workers'])>0
 assert read(W/'nod_cpu_share'/f'parity-{n}.json')['passed']
 # Bounded production provenance audit: at most four sources / ten rows per node.
 used=set();checked=0
 # Pin previously observed production samples: a worker may already have moved
 # to a new source whose first ten rows have not been written yet.
 for pinned in [x for x in previous if x['node']==n][:4]:
  p=Path(pinned['path']);assert p==result_path(old,'nod',pinned['sample_id'])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==pinned['sha256']
  job=read(old/'jobs'/f"{pinned['job_id']}.json")
  row=next(x for x in job['rows'] if x['sample_id']==pinned['sample_id']);r=read(p)
  assert r['source']==row['source'] and r['fps']==25 and r['profile']=='recall' and r['status']=='ok'
  assert r['completed_at']>=cfg['created_at']
  receipt=read(W/'visual_quality/early/completed'/f"{pinned['job_id']}.json")
  decision=next(x for x in receipt['decisions'] if x['sample_id']==pinned['sample_id'])
  assert decision['decision']=='retain' and decision['source_sha256']==canonical_hash(row['source'])
  assert all(0<=e['start']<e['end']<=r['duration_s'] for e in r['events'])
  samples.append(pinned);checked+=1
 for worker in sorted(nh['workers'],key=lambda x:-x['completed']):
  jid=worker.get('job_id')
  if not jid or jid in used:continue
  used.add(jid);job=read(old/'jobs'/f'{jid}.json')
  for row in job['rows'][:10]:
   p=result_path(old,'nod',row['sample_id'])
   if not p.exists():continue
   r=read(p)
   if r.get('completed_at',0)<cfg['created_at'] or r.get('status')!='ok':continue
   assert r['source']==row['source'] and r['fps']==25 and r['profile']=='recall'
   receipt=read(W/'visual_quality/early/completed'/f'{jid}.json')
   d=next(x for x in receipt['decisions'] if x['sample_id']==row['sample_id'])
   assert d['decision']=='retain' and d['source_sha256']==canonical_hash(row['source'])
   assert all(0<=e['start']<e['end']<=r['duration_s'] for e in r['events'])
   samples.append(dict(node=n,sample_id=row['sample_id'],job_id=jid,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   checked+=1;break
  if len(used)>=4:break
 assert checked>0,(n,'No bounded production sample found')
 checks.append(dict(node=n,workers=len(nh['workers']),completed=sum(x['completed'] for x in nh['workers']),production_samples=checked))
assert not read(old/'queues/nod/state.json')['failed']
assert not read(W/'queues/caption/state.json')['failed']
assert not read(W/'visual_quality/queues/measure/state.json')['failed']
out=dict(at=time.time(),passed=True,checks=checks,production_samples=samples,gpu_identities_unchanged=32,scope='Three-host exact replay, actual runtime identities, fresh production progress, bounded provenance and quality admission; not a full output scan.')
write(W/'nod_cpu_share/audit.json',out);print(json.dumps({k:v for k,v in out.items() if k!='production_samples'}))
