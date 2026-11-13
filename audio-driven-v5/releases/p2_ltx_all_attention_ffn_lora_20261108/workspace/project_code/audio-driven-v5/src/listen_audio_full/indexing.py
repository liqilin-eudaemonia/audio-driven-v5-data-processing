"""Build an immutable RO scope index without materializing full manifests or text."""
import hashlib,json,os,sqlite3,time
from pathlib import Path
from p2_listen.contracts import read,sha,atomic_json,digest


def array_rows(path,chunk_size=1024*1024):
 decoder=json.JSONDecoder();buffer='';ended=False;first=True
 with Path(path).open() as stream:
  def more():
   nonlocal buffer,ended
   data=stream.read(chunk_size);buffer+=data;ended=not data
  more();buffer=buffer.lstrip()
  if not buffer.startswith('['):raise ValueError('Expected JSON array')
  buffer=buffer[1:]
  while True:
   buffer=buffer.lstrip()
   while not buffer and not ended:more();buffer=buffer.lstrip()
   if buffer.startswith(']'):
    if buffer[1:].strip() or stream.read().strip():raise ValueError('Trailing JSON data')
    return
   if not first:
    if not buffer.startswith(','):raise ValueError('Expected array separator')
    buffer=buffer[1:].lstrip()
   while True:
    try:
     buffer=buffer.lstrip();value,end=decoder.raw_decode(buffer);break
    except json.JSONDecodeError:
     if ended:raise
     more()
   if not isinstance(value,dict):raise ValueError('Expected object rows')
   yield value;buffer=buffer[end:];first=False


def connect(run):return sqlite3.connect(f'file:{Path(run)/"scope.sqlite"}?mode=ro&immutable=1',uri=True)


def freeze_scope(run,expected=722706):
 from listen_benchmark.backchannel_face_stats import SourceReader
 run=Path(run);launch=read(run/'launch.json');base=Path(launch['base']);fusion=Path(launch['fusion'])
 sources={str(p):sha(p) for p in [base/'p2_config.json',base/'manifests/observations.json',base/'manifests/targets.json',fusion/'index.json',base/'source_index/COMPLETE.json']}
 if (run/'scope.json').exists():
  saved=read(run/'scope.json')
  if saved['samples']!=expected or saved['source_files']!=sources or sha(run/'scope.sqlite')!=saved['sqlite_sha256']:raise ValueError('Frozen full scope changed')
  return saved
 temp=run/'scope.sqlite.building';temp.unlink(missing_ok=True);db=sqlite3.connect(temp)
 db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
 db.executescript('CREATE TABLE samples(sid TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,obs TEXT NOT NULL,target TEXT,label_file TEXT,label_offset INTEGER,label_size INTEGER,label_sha TEXT,archive TEXT,member TEXT,start REAL); CREATE TABLE archives(key TEXT PRIMARY KEY,path TEXT UNIQUE,samples INTEGER);')
 duration=0.;count=0;started=time.monotonic()
 for ordinal,o in enumerate(array_rows(base/'manifests/observations.json')):
  if o['official_split']!='train' or o['experiment_split']!='train' or not 2-1e-7<=o['requested_duration_s']<=11.95+1e-6:raise ValueError('Unexpected full-scope split/duration')
  db.execute('INSERT INTO samples(sid,ordinal,obs) VALUES(?,?,?)',(o['sample_id'],ordinal,json.dumps(o,ensure_ascii=False)));duration+=o['requested_duration_s'];count+=1
  if count%10000==0:atomic_json(run/'index_progress.json',dict(stage='observations',completed=count,expected=expected))
 if count!=expected:raise ValueError(f'Scope count {count} != {expected}')
 seen=0
 for t in array_rows(base/'manifests/targets.json'):
  cursor=db.execute('UPDATE samples SET target=? WHERE sid=? AND target IS NULL',(json.dumps(t,ensure_ascii=False),t['sample_id']))
  if cursor.rowcount!=1:raise ValueError('Duplicate/out-of-scope target '+t['sample_id'])
  seen+=1
 if seen!=expected:raise ValueError('Target scope differs')
 db.commit();index=read(fusion/'index.json');reader=SourceReader(base);processed=0;shard_bindings=[]
 if index['samples']!=expected:raise ValueError('Fusion scope differs')
 try:
  for shard in index['shards']:
   if not shard['done']:raise ValueError('Unfinished source fusion shard')
   for relative,info in shard['files'].items():
    p=fusion/relative
    if p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:raise ValueError('Source shard changed: '+str(p))
   path=fusion/shard['labels'];n=0
   with path.open('rb') as f:
    while True:
     offset=f.tell();line=f.readline()
     if not line:break
     label=json.loads(line);sid=label['sample_id'];row=db.execute('SELECT obs,target,label_file FROM samples WHERE sid=?',(sid,)).fetchone()
     if row is None or row[2] is not None:raise ValueError('Duplicate/out-of-scope integrated row '+sid)
     obs,target=json.loads(row[0]),json.loads(row[1]);full=reader.get(label['source'])
     if full['split']!='train' or abs(full['end']-full['start']-obs['requested_duration_s'])>1/48000+1e-6:raise ValueError('Source timing mismatch')
     source=full['sources']['listener'];member=str(Path(source['member']).with_suffix('.json'))
     if Path(member).stem!=full['listener_id']:raise ValueError('Official member/listener mismatch')
     db.execute('UPDATE samples SET label_file=?,label_offset=?,label_size=?,label_sha=?,archive=?,member=?,start=? WHERE sid=?',
      (str(path),offset,len(line),hashlib.sha256(line).hexdigest(),source['archive'],member,full['start'],sid));n+=1;processed+=1
   if n!=shard['counts']['samples']:raise ValueError('Fusion shard count mismatch')
   db.commit();shard_bindings.append(dict(shard=shard['shard'],files=shard['files'],counts=shard['counts']))
   atomic_json(run/'index_progress.json',dict(stage='fusion_and_sources',completed=processed,total=expected,shards=len(shard_bindings),seconds=time.monotonic()-started))
 finally:reader.close()
 if processed!=expected or db.execute('SELECT count(*) FROM samples WHERE label_file IS NULL OR target IS NULL').fetchone()[0]:raise ValueError('Incomplete scope association')
 db.execute('CREATE INDEX by_archive ON samples(archive,member,ordinal)')
 for path,n in db.execute('SELECT archive,count(*) FROM samples GROUP BY archive').fetchall():db.execute('INSERT INTO archives VALUES(?,?,?)',(digest(path),path,n))
 db.commit();ids=hashlib.sha256()
 for (sid,) in db.execute('SELECT sid FROM samples ORDER BY ordinal'):ids.update((sid+'\n').encode())
 archives=db.execute('SELECT count(*) FROM archives').fetchone()[0];db.close()
 with temp.open('rb') as f:os.fsync(f.fileno())
 os.replace(temp,run/'scope.sqlite')
 atomic_json(run/'source_shards.json',shard_bindings,immutable=True)
 result=dict(samples=expected,duration_s=duration,all_train=True,archives=archives,source_files=sources,source_shards_sha256=sha(run/'source_shards.json'),
  ordered_ids_sha256=ids.hexdigest(),ids_hash_format='utf8 sample_id plus newline in original observation order',sqlite_sha256=sha(run/'scope.sqlite'),
  source_completion_status=index['completion_status'],training_admitted=False,seconds=time.monotonic()-started)
 atomic_json(run/'scope.json',result,immutable=True);return result
