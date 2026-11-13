"""Bounded archive preparation; preserves unresolved originals and training admission."""
import copy,hashlib,itertools,json,math,os,tarfile,time,traceback,functools
from pathlib import Path
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import load
from listen_audio_postprocess.bulk import derive
from listen_audio_postprocess.recipes import make_recipe,SCHEMA
from listen_audio_postprocess.prepare import adapt_canonical
from listen_audio_postprocess.overlay import VERSION
from .indexing import connect


def verified_label(row):
 with Path(row['label_file']).open('rb') as f:f.seek(row['label_offset']);line=f.read(row['label_size'])
 if hashlib.sha256(line).hexdigest()!=row['label_sha']:raise ValueError('Indexed source label changed')
 label=json.loads(line)
 if label['sample_id']!=row['sid']:raise ValueError('Indexed label ID differs')
 return label


def init_view(run):
 run=Path(run);launch=read(run/'launch.json');scope=read(run/'scope.json');base=Path(launch['base']);view=run/'view'
 cfg=load(base/'p2_config.json');cfg['training']['feature_loss_enabled']=False
 atomic_json(run/'base_config.json',cfg,immutable=True)
 db=connect(run);archives={key:dict(path=path,size=Path(path).stat().st_size,mtime_ns=Path(path).stat().st_mtime_ns) for key,path in db.execute('SELECT key,path FROM archives')};db.close()
 if (run/'archive_sources.json').exists():
  if read(run/'archive_sources.json')!=archives:raise ValueError('Source archive fingerprint changed')
 else:atomic_json(run/'archive_sources.json',archives,immutable=True)
 policy=dict(schema=VERSION,recipe_schema=SCHEMA,scope_sha256=sha(run/'scope.json'),source_shards_sha256=sha(run/'source_shards.json'),
  archive_sources_sha256=sha(run/'archive_sources.json'),code_manifest_sha256=sha(run/'code_manifest.json'),samples=scope['samples'],
  base_output=str(base),base_config_sha256=sha(run/'base_config.json'),training_started=False,
  unresolved_policy='preserve_original_excluded_no_invented_timing',canonical_policy='retain_existing_admission_no_new_events')
 atomic_json(view/'audio_postprocess/policy_manifest.json',policy,immutable=True)
 for dest,source in [(view/'manifests'/n,base/'manifests'/n) for n in ('observations.json','targets.json')]+[(view/'cache'/n,base/'cache'/n) for n in ('input','text')]:
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.is_symlink():
   if dest.resolve()!=source.resolve():raise ValueError('Unexpected view link')
  elif dest.exists():raise FileExistsError(dest)
  else:dest.symlink_to(source,target_is_directory=source.is_dir())
 cfg['output']=str(view);cfg['data']['audio_postprocess']=dict(schema=VERSION,base_output=str(base),policy_manifest_sha256=sha(view/'audio_postprocess/policy_manifest.json'))
 atomic_json(view/'config.json',cfg,immutable=True)
 return cfg


def derive_sample(label,original,obs,target,start,raw,provenance,line_sha,code_sha):
 audio=[e for e in label['events'] if e.get('modality')=='audio'];valid=[];unresolved=[]
 for event in audio:
  if all(type(event.get(k)) in (int,float) and math.isfinite(event[k]) for k in ('start','end')) and event['end']>event['start']:valid.append(event)
  else:unresolved.append(dict(original_event=copy.deepcopy(event),reason='source_audio_event_has_no_valid_interval',action='retained_for_review_no_invented_mute_or_training_interval'))
 job=dict(sample_id=obs['sample_id'],target_audio=target['target_audio'],source_audio_sha256=target['media_hashes']['target_audio'],duration_s=obs['requested_duration_s'],start=start,integrated_sha256=line_sha,events=valid)
 record,audit=derive(job,raw,provenance)
 opened=copy.deepcopy(label);final=record['audio_events']
 opened['events']=[copy.deepcopy(e) for e in label['events'] if e.get('modality')!='audio']+copy.deepcopy(final)
 allowed={e['source'].get('event_id') for e in original['events'] if e['type']=='listener_vocalization'}
 unmapped=[copy.deepcopy(e) for e in final if e['event_id'] not in allowed]
 record['audio_events']=[e for e in final if e['event_id'] in allowed]
 record['provenance']['full_pipeline_code_sha256']=code_sha
 audit.update(original_audio_events=len(audio),open_retained_audio_events=len(final),retained_audio_events=len(record['audio_events']),
  unresolved_audio_events=unresolved,canonical_excluded_unmapped=[dict(original_event=e,reason='not_admitted_in_original_canonical') for e in unmapped],
  raw_issue_count=label.get('raw_issue_count',0),semantic_issue_count=label.get('semantic_issue_count',0))
 opened['audio_postprocess']=dict(policy=VERSION,original_integrated_sha256=line_sha,text_rewritten=False,visual_events_unchanged=True,
  excluded_audio_events=audit['excluded_audio_events'],unresolved_audio_events=unresolved,canonical_unmapped_event_ids=[e['event_id'] for e in unmapped],
  canonical_policy='keep_original_admission',full_pipeline_code_sha256=code_sha)
 return record,audit,opened


def atomic_jsonl(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
 if path.exists():
  if sha(path)!=sha(temp):temp.unlink();raise ValueError('Immutable task differs')
  temp.unlink()
 else:os.replace(temp,path)


def prepare_archive(run,key):
 from listen_benchmark.full_fusion import lock
 with lock(Path(run)/'prepare_locks'/f'{key}.lock') as acquired:
  if not acquired:raise RuntimeError('Archive preparation already active')
  return _prepare_archive(run,key)

@functools.lru_cache(maxsize=4)
def archive_sources(run):return read(Path(run)/'archive_sources.json')

def _prepare_archive(run,key):
 run=Path(run);cfg=load(run/'view/config.json');base=Path(cfg['data']['audio_postprocess']['base_output']);view=run/'view'
 source=archive_sources(str(run))[key];archive=Path(source['path']);before=archive.stat()
 if (before.st_size,before.st_mtime_ns)!=(source['size'],source['mtime_ns']):raise ValueError('Official archive changed')
 marker=run/'prepared_archives'/f'{key}.json'
 repair=(run/'preparation_repairs'/f'{key}.json').exists()
 if marker.exists() and not repair:
  try:report=read(marker)
  except (ValueError,OSError):report={'tasks':[]};repair=True
  intact=not repair
  for task in report['tasks']:
   if not Path(task['path']).exists() or sha(task['path'])!=task['sha256']:
    intact=False;continue
   ready=run/'ready'/f'{task["task_id"]}.json'
   try:intact &= Path(task['integrated']).exists() and sha(task['integrated'])==task['integrated_sha256'] and read(ready)==task
   except (ValueError,OSError):intact=False
   with Path(task['path']).open() as stream:
    for line in stream:
     job=json.loads(line);sid=job['observation']['sample_id']
     for folder,field in [('records','record_sha256'),('audits','audit_sha256'),('view/audio_postprocess/recipes','recipe_sha256'),('view/labels/canonical','canonical_sha256')]:
      try:intact &= digest(read(run/folder/f'{sid}.json'))==job[field]
      except (OSError,ValueError,KeyError):intact=False
  if intact:return report
  repair=True
 db=connect(run);db.row_factory=__import__('sqlite3').Row
 members={x[0] for x in db.execute('SELECT DISTINCT member FROM samples WHERE archive=?',(str(archive),))};found={}
 with tarfile.open(archive,'r:') as tar:
  for member in tar:
   if member.name not in members:continue
   if member.name in found:raise ValueError('Duplicate official metadata member')
   body=tar.extractfile(member).read();raw=json.loads(body);provenance=dict(archive=str(archive),member=member.name,sha256=hashlib.sha256(body).hexdigest(),archive_size=before.st_size,archive_mtime_ns=before.st_mtime_ns)
   dest=run/'official_metadata'/f'{digest([str(archive),member.name])}.json'
   atomic_json(dest,dict(raw=raw,provenance=provenance),immutable=True);found[member.name]=dest
   if len(found)==len(members):break
 if set(found)!=members:raise FileNotFoundError(f'Official metadata missing: {str(archive)} {sorted(members-set(found))[:8]}')
 after=archive.stat()
 if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Official archive changed during read')
 launch=read(run/'launch.json');code_sha=sha(run/'code_manifest.json');tasks=[];errors=[];count=0
 cursor=db.execute('SELECT * FROM samples WHERE archive=? ORDER BY member,ordinal',(str(archive),))
 for chunk_index in itertools.count():
  batch=cursor.fetchmany(launch['chunk_size'])
  if not batch:break
  jobs=[];chunk_errors=[];integrated=[]
  for row in batch:
   sid=row['sid']
   try:
    o,t=json.loads(row['obs']),json.loads(row['target']);label=verified_label(row);original=read(base/'labels/canonical'/f'{sid}.json');meta=read(found[row['member']])
    record,audit,opened=derive_sample(label,original,o,t,row['start'],meta['raw'],meta['provenance'],row['label_sha'],code_sha)
    opened['audio_postprocess'].update(source_integrated_file=row['label_file'],source_integrated_offset=row['label_offset'],source_evidence_root=str(Path(row['label_file']).parent.parent))
    recipe=make_recipe(sid,t['target_audio'],t['media_hashes']['target_audio'],record['official_listener'],record['extra_mute_intervals'],record['provenance'])
    canonical=adapt_canonical(original,record,recipe,o,cfg,require_admitted=False)
    for folder,value in [('records',record),('audits',audit),('view/audio_postprocess/recipes',recipe),('view/labels/canonical',canonical)]:
     dest=run/folder/f'{sid}.json'
     if repair and dest.exists():
      try:equal=read(dest)==value
      except (ValueError,OSError):equal=False
      if not equal:
       quarantine=run/'quarantine'/f'prepared-{folder.replace(chr(47),chr(95))}-{sid}-{time.time_ns()}.json';quarantine.parent.mkdir(parents=True,exist_ok=True);dest.replace(quarantine)
     atomic_json(dest,value,immutable=True)
    jobs.append(dict(observation=o,target=t,ordinal=row['ordinal'],source_label_sha256=row['label_sha'],record_sha256=digest(record),canonical_sha256=digest(canonical),recipe_sha256=digest(recipe),audit_sha256=digest(audit)))
    integrated.append(opened);count+=1
   except Exception:
    error=dict(sample_id=sid,archive_key=key,error=traceback.format_exc(),time=time.time());atomic_json(run/'preparation_errors'/f'{sid}.json',error);chunk_errors.append(error)
  if chunk_errors:errors.extend(chunk_errors);continue
  task_id=f'{key[:24]}-{chunk_index:05d}';path=run/'tasks'/f'{task_id}.jsonl';open_path=run/'integrated'/f'{task_id}.jsonl'
  if repair:
   for damaged in (open_path,path):
    if damaged.exists():
     preserved=run/'quarantine'/f'task-{damaged.parent.name}-{damaged.name}-{time.time_ns()}';preserved.parent.mkdir(parents=True,exist_ok=True);damaged.replace(preserved)
  atomic_jsonl(open_path,integrated);atomic_jsonl(path,jobs)
  task=dict(task_id=task_id,path=str(path),sha256=sha(path),samples=len(jobs),integrated=str(open_path),integrated_sha256=sha(open_path),rank=int(digest(task_id)[:12],16)%launch['shards'])
  # The immutable ready marker is published only after labels/recipes/tasks/integrated records exist.
  ready=run/'ready'/f'{task_id}.json'
  if repair and ready.exists():
   preserved=run/'quarantine'/f'ready-{task_id}-{time.time_ns()}.json';preserved.parent.mkdir(parents=True,exist_ok=True);ready.replace(preserved)
  atomic_json(ready,task,immutable=True);tasks.append(task)
  for job in jobs:(run/'preparation_errors'/f'{job["observation"]["sample_id"]}.json').unlink(missing_ok=True)
 db.close()
 report=dict(archive_key=key,samples=count,tasks=tasks,errors=len(errors))
 if errors:raise RuntimeError(f'{len(errors)} preparation errors in archive {key}; valid tasks preserved')
 if repair and marker.exists():
  preserved=run/'quarantine'/f'archive-{key}-{time.time_ns()}.json';preserved.parent.mkdir(parents=True,exist_ok=True);marker.replace(preserved)
 atomic_json(marker,report,immutable=True);return report
