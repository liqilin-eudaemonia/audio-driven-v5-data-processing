"""Bounded status inspection: fixed state files and an incremental receipt cursor."""
from pathlib import Path
import json,time,sqlite3
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def main():
 cfg=read(WORK/'config.json');q=read(WORK/'queues/caption/state.json')
 db=sqlite3.connect(WORK/'receipt_index.sqlite',timeout=30)
 db.execute('CREATE TABLE IF NOT EXISTS sources(jid TEXT PRIMARY KEY,done INTEGER DEFAULT 0,caption INTEGER DEFAULT 0,held INTEGER DEFAULT 0,failed INTEGER DEFAULT 0)')
 db.execute('CREATE TABLE IF NOT EXISTS cursor(id INTEGER PRIMARY KEY,offset INTEGER)')
 db.commit();db.execute('BEGIN IMMEDIATE')
 row=db.execute('SELECT offset FROM cursor WHERE id=1').fetchone();offset=row[0] if row else 0
 with (WORK/'ready.jsonl').open() as f:
  f.seek(offset)
  for _ in range(1000):
   if f.tell()>=q['offset']:break
   line=f.readline()
   if not line or not line.endswith('\n'):break
   db.execute('INSERT OR IGNORE INTO sources(jid) VALUES(?)',(json.loads(line)['job_id'],));offset=f.tell()
 for (jid,) in db.execute('SELECT jid FROM sources WHERE done=0').fetchall():
  p=WORK/'caption/completed'/f'{jid}.json'
  if p.exists():
   r=read(p);db.execute('UPDATE sources SET done=1,caption=?,held=?,failed=? WHERE jid=?',(r['caption_clips'],r['held_clips'],r['failed_generation_clips'],jid))
 db.execute('INSERT OR REPLACE INTO cursor VALUES(1,?)',(offset,));db.commit()
 done,caption,held,failed=db.execute('SELECT COALESCE(SUM(done),0),COALESCE(SUM(caption),0),COALESCE(SUM(held),0),COALESCE(SUM(failed),0) FROM sources').fetchone()
 index=dict(offset=offset,completed_sources=done,caption_clips=caption,held_clips=held,failed_generation_clips=failed,valid_generation_clips=caption-failed)
 db.close()
 report=dict(at=time.time(),target_clips=cfg['total_clips'],resolved_clips=q['completed_clips'],failed_sources=len(q['failed']),inflight_sources=len(q['inflight']),receipt_counts=index,nodes={})
 for node,nc in cfg['nodes'].items():
  p=WORK/'nodes'/node/'supervisor.json';s=read(p) if p.exists() else {}
  rows=[]
  for gpu in nc['gpu_ids']:
   p=WORK/'workers'/f'{node}-gpu-{gpu}.json';h=read(p) if p.exists() else {}
   rows.append(dict(gpu=gpu,stage=h.get('stage'),completed_this_process=h.get('completed_clips',0),held_this_process=h.get('held_clips',0),generation_failures_this_process=h.get('failed_clips',0),last_error=(h.get('last_error') or '')[:250],execution_revision=h.get('execution_revision'),heartbeat_age=time.time()-h.get('heartbeat',time.time())))
  old=Path(nc['old_work']) if nc.get('old_work') else None
  oq=read(old/'queues/nod/state.json') if old else dict(completed_clips=0,failed={})
  os=read(old/'supervisor.json') if old else {}
  report['nodes'][node]=dict(stage=s.get('stage'),error=s.get('error'),blocked_gpus=s.get('blocked_gpus',{}),devices=s.get('devices'),workers=rows,nod_completed=oq['completed_clips'],nod_failed_sources=len(oq['failed']),legacy_gpu_ids=os.get('gpu_ids'),nod_supervisor_error=os.get('error'))
 (WORK/'status.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
