import concurrent.futures,hashlib,json,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];F=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write
T=R/'experiments/listen_j_annotation_three_nodes_20261030';scope=read(T/'nodes.json');original_path=R/'experiments/listen_j_annotation_two_nodes_20261030/original_ready.jsonl'
def entries(path):
 raw=path.read_bytes();seen={};dups=[];offset=0
 for line in raw.splitlines(keepends=True):
  x=json.loads(line)
  if x['job_id'] in seen:
   assert seen[x['job_id']]==x['count'];dups.append(offset)
  seen[x['job_id']]=x['count'];offset+=len(line)
 return seen,dups,hashlib.sha256(raw).hexdigest(),len(raw.splitlines())
original,_,orig_sha,_=entries(original_path);nodes={};all_present={};stats={}
for n in ['node17','node8']:
 old=Path(scope['nodes'][n]['work']);items,dups,sha,count=entries(old/'ready.jsonl');assert not set(items)&set(all_present);all_present.update(items)
 nodes[n]=dict(old_work=str(old),scope_sources=count,unique_sources=len(items),scope_clips=sum(items.values()),journal_sha256=sha,skip_duplicate_offsets=dups,allowed_jobs=items)
 stats[n]=dict(journal_entries=count,unique_sources=len(items),clips=sum(items.values()),duplicate_offsets=dups)
local=Path(scope['nodes']['node12']['work']);local_items,dups,sha,count=entries(local/'ready.jsonl');assert not dups
missing={j:count for j,count in original.items() if j not in all_present};assert set(missing)<=set(local_items)
assert set(original)==set(all_present)|set(local_items) and all(original[j]==count for j,count in all_present.items()) and all(original[j]==count for j,count in local_items.items())
q=read(local/'queues/nod/state.json');assert not q['inflight'] and not q['failed'] and q['offset']==(local/'ready.jsonl').stat().st_size
# Read only named historical receipts/jobs, at concurrency two; no recursive scan.
def check(entry):
 jid,count=entry;rp=local/'nod/completed'/f'{jid}.json';raw=rp.read_bytes();receipt=json.loads(raw);job=read(local/'jobs'/f'{jid}.json')
 assert receipt['job_id']==jid and receipt['clips']==count==len(job['rows']) and receipt['source_result_sha256']==job['source_result_sha256']
 return jid,dict(clips=count,receipt_sha256=hashlib.sha256(raw).hexdigest(),source_result_sha256=job['source_result_sha256'])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:verified=dict(pool.map(check,local_items.items()))
filtered=F/'node12_completed_missing.jsonl';filtered.write_text(''.join(json.dumps(dict(job_id=j,count=n))+'\n' for j,n in missing.items()))
nodes['node12_legacy']=dict(old_work=str(local),scope_sources=len(missing),unique_sources=len(missing),scope_clips=sum(missing.values()),imported_completed_journal=str(filtered),journal_sha256=hashlib.sha256(filtered.read_bytes()).hexdigest(),allowed_jobs=missing)
owners={j:x['nod'] for j,x in scope['branch_ownership_overrides'].items()};assert all(Path(x)==local and j in local_items for j,x in owners.items())
assert sum(n['unique_sources'] for n in nodes.values())==len(original)==60869
assert sum(n['scope_clips'] for n in nodes.values())==sum(original.values())==1151805
plan=dict(schema='j-omni-source-ownership-v1',created_at=time.time(),total_clips=1151805,total_unique_sources=60869,nodes=nodes,delegated_nod_owners=owners,original_journal=str(original_path),original_journal_sha256=orig_sha,history_scope=str(T/'nodes.json'),history_scope_sha256=hashlib.sha256((T/'nodes.json').read_bytes()).hexdigest(),reason='Recover completed third-node scope and resolve historical delegated nod ownership without reannotation.')
write(F/'source_ownership.json',plan);write(F/'historical_receipt_validation.json',dict(at=time.time(),passed=True,sources=len(verified),clips=sum(x['clips'] for x in verified.values()),receipts=verified))
report=dict(at=time.time(),passed=True,total_sources=60869,total_clips=1151805,missing_sources=len(missing),missing_clips=sum(missing.values()),delegated_sources=len(owners),source_nodes=stats,plan_path=str(F/'source_ownership.json'),plan_sha256=hashlib.sha256((F/'source_ownership.json').read_bytes()).hexdigest())
write(F/'scope_reconciliation.json',report);print(json.dumps(report))
