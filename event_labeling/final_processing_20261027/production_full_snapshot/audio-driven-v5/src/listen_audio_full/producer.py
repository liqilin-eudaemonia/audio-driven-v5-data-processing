"""CPU archive producer with append-only bounded task streams and an explicit final barrier."""
import collections,concurrent.futures,json,os,time
from pathlib import Path
from p2_listen.contracts import read,sha,digest,atomic_json
from listen_benchmark.full_fusion import lock
from .indexing import connect
from .preparation import init_view,prepare_archive


def verify_code(run):
 manifest=read(Path(run)/'code_manifest.json')
 for path,expected in manifest.items():
  if sha(path)!=expected:raise ValueError('Full pipeline source changed: '+path)


def publish_ready(run,published,shards):
 run=Path(run);added=0
 for p in sorted((run/'ready').glob('*.json')):
  if p.stem in published:continue
  task=read(p);key=task['task_id']
  if key in published:
   if published[key]!=digest(task):raise ValueError('Previously queued task changed')
   continue
  if not 0<=task['rank']<shards:raise ValueError('Task worker rank out of range')
  queue=run/'queues'/f'{task["rank"]:03d}.jsonl';queue.parent.mkdir(parents=True,exist_ok=True)
  with queue.open('a') as f:f.write(json.dumps(task,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
  published[key]=digest(task);added+=1
 return added


def queued_tasks(run,shards,repair_tail=False):
 published={};queues={};samples=0
 for rank in range(shards):
  p=Path(run)/'queues'/f'{rank:03d}.jsonl';n=0;total=0
  if p.exists():
   data=p.read_bytes()
   if data and not data.endswith(b'\n'):
    if not repair_tail:raise ValueError('Torn task queue')
    end=data.rfind(b'\n')+1
    with p.open('r+b') as f:f.truncate(end);f.flush();os.fsync(f.fileno())
    data=data[:end]
   for line in data.splitlines():
    task=json.loads(line)
    if task['task_id'] in published:raise ValueError('Duplicate task assignment')
    if task['rank']!=rank:raise ValueError('Task in wrong worker stream')
    published[task['task_id']]=digest(task);n+=1;total+=task['samples']
  queues[str(rank)]=dict(samples=total,tasks=n,sha256=sha(p) if p.exists() else None);samples+=total
 return published,queues,samples


def produce(run):
 run=Path(run);verify_code(run);launch=read(run/'launch.json');shards=launch['shards'];started=time.time()
 with lock(run/'producer.lock') as acquired:
  if not acquired:raise RuntimeError('Producer already active')
  init_view(run);published,_,_=queued_tasks(run,shards,repair_tail=True)
  db=connect(run);all_archives=[x[0] for x in db.execute('SELECT key FROM archives ORDER BY samples DESC')];db.close()
  repairs=[p.stem for p in (run/'preparation_repairs').glob('*.json')]
  if any(k not in all_archives for k in repairs):raise ValueError('Unknown archive repair request')
  archives=repairs if (run/'producer_complete.json').exists() and repairs else repairs+[k for k in all_archives if k not in repairs]
  # At most one bounded archive per thread; no full event corpus is materialized.
  errors=[];done=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=launch['prepare_workers']) as pool:
   pending={};iterator=iter(archives)
   while True:
    while len(pending)<launch['prepare_workers']:
     try:key=next(iterator)
     except StopIteration:break
     pending[pool.submit(prepare_archive,run,key)]=key
    if not pending:break
    completed,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
    publish_ready(run,published,shards)
    for future in completed:
     key=pending.pop(future)
     try:
      future.result();done+=1
      request=run/'preparation_repairs'/f'{key}.json'
      if request.exists():
       repaired=run/'repair_history'/f'{key}-{time.time_ns()}.json';repaired.parent.mkdir(parents=True,exist_ok=True);request.replace(repaired)
      prior_error=run/'archive_errors'/f'{key}.json'
      if prior_error.exists():
       recovered=run/'resolved_errors'/f'{key}-{time.time_ns()}.json';recovered.parent.mkdir(parents=True,exist_ok=True);prior_error.replace(recovered)
     except Exception as error:
      failure=dict(archive_key=key,error=str(error),time=time.time());errors.append(failure);atomic_json(run/'archive_errors'/f'{key}.json',failure)
    atomic_json(run/'producer_status.json',dict(phase='preparing',archives_done=done,archives_total=len(archives),ready_tasks=len(published),errors=len(errors),updated=time.time(),started=started))
    if (run/'STOP').exists():raise RuntimeError('Producer STOP requested')
  publish_ready(run,published,shards)
  if errors:raise RuntimeError(f'{len(errors)} archive preparations failed; tasks retained, producer will retry')
  published,queues,total=queued_tasks(run,shards)
  if total!=read(run/'scope.json')['samples']:raise ValueError('Ready task scope count differs')
  result=dict(samples=total,tasks=len(published),queues=queues,scope_sha256=sha(run/'scope.json'),code_manifest_sha256=sha(run/'code_manifest.json'))
  atomic_json(run/'producer_complete.json',result,immutable=True)
  atomic_json(run/'producer_status.json',dict(phase='complete',archives_done=done,ready_tasks=len(published),samples=total,errors=0,updated=time.time(),started=started))
  return result
