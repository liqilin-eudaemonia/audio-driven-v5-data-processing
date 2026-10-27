"""Native audio-only workers and per-task exhaustive verification with bounded memory."""
import collections,json,time,uuid,os
from pathlib import Path
import torch
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import load,activate
from p2_listen.cache import load_part,save_part
from p2_listen import features
from p2_listen.annotations import rasterize,validate,validate_contract
from listen_audio_postprocess.overlay import build_one,sample_contract
from listen_audio_postprocess.recipes import masks
from listen_benchmark.full_fusion import lock
from .indexing import connect
from .preparation import verified_label


def task_rows(task):
 if sha(task['path'])!=task['sha256'] or sha(task['integrated'])!=task['integrated_sha256']:raise ValueError('Published task/integrated source changed')
 with Path(task['path']).open() as f:jobs=[json.loads(line) for line in f]
 with Path(task['integrated']).open() as f:opened=[json.loads(line) for line in f]
 if len(jobs)!=task['samples'] or len(opened)!=len(jobs):raise ValueError('Task row count differs')
 return jobs,opened


def quarantine_corrupt(run,folder,identity):
 folder=Path(folder);marker=folder/'complete.json'
 if not marker.exists():return
 m=read(marker)
 if m['identity']!=digest(identity) or digest(m['contract'])!=m['identity']:raise ValueError('Cache identity mismatch; explicit version repair required')
 if not (folder/'payload.pt').exists() or sha(folder/'payload.pt')!=m['payload_sha256']:
  dest=Path(run)/'quarantine'/f'{folder.parent.name}-{folder.name}-{uuid.uuid4().hex}'
  dest.parent.mkdir(parents=True,exist_ok=True);folder.replace(dest)


def cache_one(run,c,job,encoder):
 o,t=job['observation'],job['target'];sid=o['sample_id']
 _,_,_,_,label,identity=sample_contract(c,o,t,require_admitted=False)
 quarantine_corrupt(run,Path(c['output'])/'cache/audio_postprocess'/sid,identity)
 result=build_one(c,o,t,encoder,require_admitted=False)
 if result.get('busy'):raise RuntimeError('Unexpected sample writer collision')
 record=read(Path(run)/'records'/f'{sid}.json')
 if digest(record)!=job['record_sha256'] or digest(label)!=job['canonical_sha256']:raise ValueError('Prepared record/label changed')
 validate(label,o,require_admitted=False);validate_contract(label,c)
 event_identity=features.identity(c,'events',o,label=label);folder=features.folder(c,'events',sid)
 quarantine_corrupt(run,folder,event_identity)
 existing=load_part(folder,event_identity,missing_ok=True)
 if existing is None:
  _,layout=features.load_input(c,o,'cpu');save_part(folder,event_identity,rasterize(label,layout,c['events']['K_train']))
 return result


def audit_task(run,c,task,final=False):
 run=Path(run);base=Path(c['data']['audio_postprocess']['base_output']);jobs,opened=task_rows(task)
 db=connect(run);db.row_factory=__import__('sqlite3').Row;counts=collections.Counter();classes=collections.Counter();ids=[]
 for job,open_label in zip(jobs,opened):
  o,t=job['observation'],job['target'];sid=o['sample_id'];ids.append(sid)
  source=db.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone()
  if source is None or json.loads(source['obs'])!=o or json.loads(source['target'])!=t:raise ValueError('Task scope/media changed')
  original_open=verified_label(source)
  if source['label_sha']!=job['source_label_sha256']:raise ValueError('Integrated label source mismatch')
  _,_,_,recipe,label,identity=sample_contract(c,o,t,require_admitted=False)
  original=read(base/'labels/canonical'/f'{sid}.json');record=read(run/'records'/f'{sid}.json');audit=read(run/'audits'/f'{sid}.json')
  if digest(recipe)!=job['recipe_sha256'] or digest(audit)!=job['audit_sha256'] or digest(record)!=job['record_sha256'] or digest(label)!=job['canonical_sha256'] or digest(original)!=label['provenance']['original_canonical_label_sha256']:raise ValueError('Label provenance changed')
  if label['training_admitted']!=original['training_admitted'] or label['provenance'].get('formal_training_admitted')!=original['provenance'].get('formal_training_admitted'):raise ValueError('Training admission modified')
  if [e for e in original['events'] if e['type']!='listener_vocalization']!=[e for e in label['events'] if e['type']!='listener_vocalization']:raise ValueError('Visual canonical modified')
  if [e for e in original_open['events'] if e.get('modality')!='audio']!=[e for e in open_label['events'] if e.get('modality')!='audio']:raise ValueError('Visual open records modified')
  originals={e['event_id']:e for e in original_open['events']}
  original_audio={e['event_id'] for e in original_open['events'] if e.get('modality')=='audio'}
  kept={e['event_id'] for e in open_label['events'] if e.get('modality')=='audio'}
  excluded={e['original_event']['event_id'] for e in audit['excluded_audio_events']}
  unresolved={e['original_event']['event_id'] for e in audit['unresolved_audio_events']}
  if kept & excluded or kept & unresolved or excluded & unresolved or kept|excluded|unresolved!=original_audio:raise ValueError('Audio event accounting incomplete or overlapping')
  for item in [*audit['excluded_audio_events'],*audit['unresolved_audio_events']]:
   e=item['original_event'];source_event=originals[e['event_id']]
   if e['description']!=source_event['description'] or e.get('text')!=source_event.get('text'):raise ValueError('Excluded original text changed')
  for e in open_label['events']:
   if e['description']!=originals[e['event_id']]['description'] or e.get('text')!=originals[e['event_id']].get('text'):raise ValueError('Original text modified')
  if sha(recipe['source']['path'])!=recipe['source']['sha256']:raise ValueError('Source WAV changed')
  patch=load_part(Path(c['output'])/'cache/audio_postprocess'/sid,identity)
  events=load_part(features.folder(c,'events',sid),features.identity(c,'events',o,label=label))
  if 'av' in patch or 'video' in patch:raise ValueError('Unexpected duplicated video payload')
  if not patch.get('reuse_base_audio') and not torch.isfinite(patch['audio_latents']['latents']).all():raise ValueError('Nonfinite audio latent')
  for value in events.values():
   if isinstance(value,dict):
    if any(isinstance(v,torch.Tensor) and not torch.isfinite(v).all() for v in value.values()):raise ValueError('Nonfinite event payload')
  source_events={e['event_id']:e for e in record['audio_events']};actual=collections.defaultdict(list)
  for e in label['events']:
   if e['type']!='listener_vocalization':continue
   eid=e['source']['audio_postprocess']['source_event_id'];ref=source_events[eid]
   if e['description']!=ref['description'] or e.get('text')!=ref.get('text'):raise ValueError('Canonical text changed')
   actual[eid].append(dict(start=e['start'],end=e['end']))
  if set(actual)!=set(source_events) or any(actual[k]!=v['intervals'] for k,v in source_events.items()):raise ValueError('Audio intervals changed')
  gate,extra,_=masks(recipe);support=gate*extra;sr=recipe['source']['sample_rate']
  for spans in actual.values():
   for x in spans:
    a,b=round(x['start']*sr),round(x['end']*sr)
    if b<=a or not (support[a:b]>0).all():raise ValueError('Audio label lies outside postprocessed support')
  if not label['training_admitted'] and any(v['known'].any() for v in events.values()):raise ValueError('Unadmitted events must remain entirely unknown')
  checked=job['ordinal']%7000==0
  if checked:
   from p2_listen.training_data import load_training_sample
   if label['training_admitted']:load_training_sample(c,o,t,'cpu')
   else:
    try:load_training_sample(c,o,t,'cpu')
    except ValueError as error:
     if 'not been admitted' not in str(error):raise
    else:raise ValueError('Unadmitted sample accepted for training')
  classes.update({e['type'] for e in label['events']})
  counts.update(dict(samples=1,training_admitted=int(label["training_admitted"]),training_unadmitted=int(not label["training_admitted"]),retained_audio_events=len(actual),open_retained_audio_events=audit['open_retained_audio_events'],
   unresolved_audio_events=len(audit['unresolved_audio_events']),canonical_excluded_unmapped=len(audit['canonical_excluded_unmapped']),
   raw_issue_count=audit['raw_issue_count'],semantic_issue_count=audit['semantic_issue_count'],removed_chinese_events=audit['removed_chinese_events'],
   with_audio_labels=int(bool(actual)),without_audio_labels=int(not actual),full_silence_support=int(not bool(support.any())),
   unchanged_audio=int(bool(patch.get('reuse_base_audio'))),changed_audio=int(not patch.get('reuse_base_audio')),
   audio_payload_bytes=(Path(c['output'])/'cache/audio_postprocess'/sid/'payload.pt').stat().st_size,training_read_checked=int(checked)))
 db.close()
 result=dict(task_id=task['task_id'],task_sha256=task['sha256'],integrated_sha256=task['integrated_sha256'],code_manifest_sha256=sha(run/'code_manifest.json'),
             final=final,counts=dict(counts),classes=dict(classes),ids=ids,verified_at=time.time())
 atomic_json(run/('final_audits' if final else 'task_done')/f'{task["task_id"]}.json',result)
 return result


def run_worker(run,rank):
 import traceback
 run=Path(run);c=load(run/'view/config.json');activate(c);torch.set_num_threads(2);encoder=None
 def get_encoder():
  nonlocal encoder
  if encoder is None:
   from listen_audio_postprocess.encoder import NativeAudioEncoder
   encoder=NativeAudioEncoder(c,'cuda:0')
  return encoder
 queue=run/'queues'/f'{rank:03d}.jsonl';position=0;tasks=[];completed=0;started=time.time();failed={};last_retry=0.
 def process(task):
  with lock(run/'task_locks'/f'{task["task_id"]}.lock') as owns:
   if not owns:raise RuntimeError('Task writer collision')
   jobs,_=task_rows(task)
   for job in jobs:cache_one(run,c,job,get_encoder)
   audit_task(run,c,task)
 def attempt(task):
  try:
   process(task)
   error=run/'task_errors'/f'{task["task_id"]}.json'
   if error.exists():
    dest=run/'resolved_errors'/f'{task["task_id"]}-{time.time_ns()}.json';dest.parent.mkdir(parents=True,exist_ok=True);error.replace(dest)
   failed.pop(task['task_id'],None);return True
  except Exception:
   failed[task['task_id']]=task
   # Missing/corrupt prepared artifacts need the CPU producer, not endless cache-only retries.
   repair=False
   try:
    jobs,_=task_rows(task)
    for job in jobs:
     sid=job['observation']['sample_id']
     for folder in ('records','audits','view/audio_postprocess/recipes','view/labels/canonical'):
      value=read(run/folder/f'{sid}.json')
      if folder=='records' and digest(value)!=job['record_sha256']:repair=True
      if folder=='view/labels/canonical' and digest(value)!=job['canonical_sha256']:repair=True
      if folder=='view/audio_postprocess/recipes' and digest(value)!=job['recipe_sha256']:repair=True
      if folder=='audits' and digest(value)!=job['audit_sha256']:repair=True
   except (OSError,ValueError,KeyError):repair=True
   if repair:
    db=connect(run);keys=[r[0] for r in db.execute('SELECT key FROM archives WHERE key LIKE ?',(task['task_id'].split('-')[0]+'%',))];db.close()
    if len(keys)!=1:raise ValueError('Ambiguous archive repair target')
    atomic_json(run/'preparation_repairs'/f'{keys[0]}.json',dict(task_id=task['task_id'],reason='missing_or_changed_prepared_artifact',time=time.time()))
   atomic_json(run/'task_errors'/f'{task["task_id"]}.json',dict(rank=rank,task=task,error=traceback.format_exc(),time=time.time()))
   return False
 def status(phase,**extra):
  atomic_json(run/'workers'/f'{rank:03d}.json',dict(rank=rank,generation=os.environ.get('FULL_AUDIO_GENERATION'),completed=completed,finished=False,phase=phase,failed_tasks=len(failed),updated=time.time(),started=started,**extra))
 with lock(run/'worker_locks'/f'{rank:03d}.lock') as acquired:
  if not acquired:raise RuntimeError('Worker already active')
  while True:
   if (run/'STOP').exists():raise RuntimeError('Full processing STOP requested')
   if queue.exists():
    with queue.open() as stream:
     stream.seek(position)
     while True:
      at=stream.tell();line=stream.readline()
      if not line or not line.endswith('\n'):position=at;break
      task=json.loads(line);tasks.append(task);position=stream.tell()
      if attempt(task):completed+=task['samples']
      status('cache')
   if failed and time.time()-last_retry>30:
    for task in list(failed.values()):
     if attempt(task):completed+=task['samples']
    last_retry=time.time()
   if (run/'producer_complete.json').exists():
    producer=read(run/'producer_complete.json');expected=producer['queues'][str(rank)]
    if not queue.exists():
     if expected['samples'] or expected['tasks']:raise ValueError('Missing nonempty task queue')
    elif sha(queue)!=expected['sha256']:raise ValueError('Final task queue identity differs')
    if sum(t['samples'] for t in tasks)!=expected['samples'] or len(tasks)!=expected['tasks']:raise ValueError('Producer/worker scope differs')
    if not failed:break
   status('retry_failed' if failed else 'await_ready');time.sleep(3)
  # Revalidate actual payloads after the complete producer barrier, not just task receipts.
  for i,task in enumerate(tasks):
   while True:
    if (run/'STOP').exists():raise RuntimeError('Full processing STOP requested')
    try:
     audit_task(run,c,task,final=True);break
    except Exception:
     attempt(task);status('repair_final_audit',audited_tasks=i,total_tasks=len(tasks));time.sleep(3)
   status('final_audit',audited_tasks=i+1,total_tasks=len(tasks))
  result=dict(rank=rank,generation=os.environ.get('FULL_AUDIO_GENERATION'),completed=completed,tasks=len(tasks),finished=True,phase='finished',updated=time.time(),started=started)
  atomic_json(run/'workers'/f'{rank:03d}.json',result);return result
