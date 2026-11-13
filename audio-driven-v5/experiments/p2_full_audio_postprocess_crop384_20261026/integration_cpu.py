"""Real-source CPU producer->worker->final-audit integration including post-completion corruption."""
import os,sys,json,sqlite3,concurrent.futures,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from p2_listen.contracts import read,sha,digest,atomic_json
from listen_audio_full.indexing import connect
from listen_audio_full.producer import produce
from listen_audio_full.worker import run_worker,task_rows
from listen_audio_full.cluster import finalize
from prepare_validation import manifest
import listen_audio_postprocess.encoder as enc
OUT=Path(__file__).resolve().parent;RUN=OUT/'integration_cpu';BASE=Path(read(OUT/'launch.json')['base'])
ids=[r['sample_id'] for r in read(OUT/'native_cpu_validation.json')['records']]
classes=set()
for sid in ids:classes.update(e['type'] for e in read(BASE/'labels/canonical'/f'{sid}.json')['events'])
if len(classes)<7:
 with (Path(read(OUT/'launch.json')['fusion'])/'labels/0000.jsonl').open() as f:
  for line in f:
   label=json.loads(line);sid=label['sample_id'];types={e['type'] for e in read(BASE/'labels/canonical'/f'{sid}.json')['events']}
   if types-classes:ids.append(sid);classes|=types
   if len(classes)==7:break
assert len(classes)==7
unadmitted=[]
for p in (OUT/'before_candidate_cache_fix/preparation_errors').glob('*.json'):
 sid=read(p)['sample_id']
 if sid not in ids:ids.append(sid);unadmitted.append(sid)
assert unadmitted
RUN.mkdir(exist_ok=True);dbpath=RUN/'scope.sqlite';dbpath.unlink(missing_ok=True);src=connect(OUT);db=sqlite3.connect(dbpath)
for table in ('samples','archives'):db.execute(src.execute('SELECT sql FROM sqlite_master WHERE type="table" AND name=?',(table,)).fetchone()[0])
for sid in ids:
 row=src.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone();db.execute('INSERT INTO samples VALUES('+','.join('?'*len(row))+')',row)
for archive,count in db.execute('SELECT archive,count(*) FROM samples GROUP BY archive').fetchall():db.execute('INSERT INTO archives VALUES(?,?,?)',(digest(archive),archive,count))
db.execute('CREATE INDEX by_archive ON samples(archive,member,ordinal)');db.commit();db.close();src.close()
launch=read(OUT/'launch.json');launch.update(expected_samples=len(ids),chunk_size=16,prepare_workers=3,shards=3,validation_only=True)
atomic_json(RUN/'launch.json',launch);atomic_json(RUN/'scope.json',dict(samples=len(ids),source_full_scope_sha256=sha(OUT/'scope.json'),validation_only=True,sqlite_sha256=sha(dbpath)))
atomic_json(RUN/'source_shards.json',dict(source_full_index_sha256=sha(OUT/'source_shards.json'),validation_only=True));atomic_json(RUN/'code_manifest.json',manifest())
started=time.time();produce(RUN)
original=enc.NativeAudioEncoder;encoder=None
def cpu_encoder(c,device):
 global encoder
 if encoder is None:encoder=original(c,'cpu')
 return encoder
enc.NativeAudioEncoder=cpu_encoder;os.environ['FULL_AUDIO_GENERATION']='cpu-integration'
for rank in range(3):run_worker(RUN,rank)
first=finalize(RUN,'cpu-integration')
from p2_listen.config import load
from p2_listen.training_data import load_training_sample
c=load(RUN/'view/config.json')
for p in (RUN/'ready').glob('*.json'):
 for job in task_rows(read(p))[0]:
  if job['observation']['sample_id'] not in unadmitted:continue
  try:load_training_sample(c,job['observation'],job['target'],'cpu')
  except ValueError as error:assert 'not been admitted' in str(error)
  else:raise AssertionError('Unadmitted sample accepted')
# Existing COMPLETE must not conceal a subsequently missing audio payload.
task=read(next((RUN/'ready').glob('*.json')));job=task_rows(task)[0][0];sid=job['observation']['sample_id']
(RUN/'view/cache/audio_postprocess'/sid/'payload.pt').unlink()
try:finalize(RUN,'must-detect-missing')
except FileNotFoundError:pass
else:raise AssertionError('Stale COMPLETE incorrectly bypassed real audit')
# Missing recipe requires CPU producer repair while the worker keeps other tasks moving.
(RUN/'view/audio_postprocess/recipes'/f'{sid}.json').unlink()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 future=pool.submit(run_worker,RUN,task['rank']);deadline=time.monotonic()+90
 while not any((RUN/'preparation_repairs').glob('*.json')):
  if future.done():future.result();raise AssertionError('Expected a preparation repair request')
  if time.monotonic()>deadline:raise TimeoutError('Repair request not observed')
  time.sleep(.1)
 produce(RUN);future.result(timeout=120)
second=finalize(RUN,'cpu-repaired')
report=dict(status='passed',unadmitted_samples=len(unadmitted),unadmitted_training_rejected=True,samples=len(ids),classes=sorted(classes),virtual_workers=3,producer_worker_audit_passed=True,
 stale_complete_does_not_hide_missing_payload=True,producer_reopened_after_completion=True,real_native_audio_vae=True,
 gpu_used=False,training_started=False,seconds=time.time()-started,completion=second)
atomic_json(OUT/'integration_cpu_validation.json',report);print(json.dumps(report,indent=2))
