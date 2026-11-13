"""Bounded cross-node lock and real J checkpoint regression checks."""
import json,sys,time,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from listen_pair_j_queue import read,write,lock,sha
from run_listen_pair_j_cluster import remote,launch,terminate,verify
cfg=read(ROOT/'configs/listen_pair_j_cpu_cluster.json')
code='''import json,fcntl,time,sys
p=json.load(sys.stdin);f=open(p['path'],'a+')
try:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
except BlockingIOError:held=False
if held:time.sleep(p.get('delay',0))
print(json.dumps(dict(acquired=held)))
'''
probe=EXP/'lock_probe.lock';handle=lock(probe)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 blocked=list(pool.map(lambda n:remote(n,cfg,code,dict(path=str(probe))),cfg['nodes'][:2]))
handle.close();assert all(not r['acquired'] for r in blocked)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 competed=list(pool.map(lambda n:remote(n,cfg,code,dict(path=str(probe),delay=2)),cfg['nodes'][:2]))
assert sum(r['acquired'] for r in competed)==1
write(EXP/'lock_validation.json',dict(remote_exclusion=blocked,remote_competition=competed,passed=True))
print('CROSS NODE LOCK PASS',flush=True)
work=EXP/'regression';m=verify(work);node=cfg['nodes'][0];session='regression-'+str(int(time.time()));started=time.time()
write(work/'control.json',dict(session=session,enabled=['node8'],stop=False,heartbeat=time.time(),deadline=time.time()+180,started=started))
launch(work,m,node,session);interrupted=False;resumed=False;checkpoint_pair=None
try:
 while time.time()-started<180:
  write(work/'control.json',dict(session=session,enabled=['node8'],stop=False,heartbeat=time.time(),deadline=started+180,started=started))
  if not interrupted:
   for item in m['items']:
    checkpoint=work/'results'/item['pair']/'checkpoint.json'
    if not checkpoint.exists():continue
    state=read(checkpoint)
    if len(state['leaves'])+len(state['parents'])>=2 and state['pending']:
     terminate(work,m,node,session);interrupted=True;checkpoint_pair=item['pair'];break
  if interrupted and not resumed:
   status=work/'nodes/node8.status.json'
   if status.exists() and read(status).get('stage')=='paused':
    # Wait for the old slot to be released, not merely its last status write.
    slot=lock(work/'nodes/node8.slot.lock',False)
    if slot:
     slot.close();launch(work,m,node,session);resumed=True
  q=read(work/'queue.json')
  if q['failed']:raise RuntimeError(str(q['failed']))
  if q['completed']==len(m['items']):break
  time.sleep(.15)
 assert read(work/'queue.json')['completed']==len(m['items'])
 assert interrupted and resumed
 results=[]
 known={}
 for folder in ['pair_semantic_trial2_20261029','pair_semantic_remaining5_20261029','pair_semantic_random5_20261029']:
  for item in m['items']:
   p=ROOT/'experiments'/folder/'results'/item['pair']/'result.json'
   if p.exists():known[item['pair']]=p
 def signature(d):
  keys=['sample_id','start','end','speaker_index','listener_index','text','semantic_start','semantic_end','recovery_depth','recovery_parent_id','semantic_recovery_hold','sentence_end_protection','head_attachment','head_attachment_fallback']
  def rows(name):return [dict({k:r.get(k) for k in keys},pool=r['audio_final']['pool'],reasons=r['audio_final']['reasons']) for r in d[name]]
  return dict(clips=rows('clips'),parents=rows('superseded_attempts'),initial_ids=d['initial_ids'],diagnostic=d['diagnostic'])
 for item in m['items']:
  pair=item['pair'];old=read(known[pair]);new=read(work/'results'/pair/'result.json')
  assert signature(old)==signature(new),pair
  assert new['counts'].get('fresh',0)==0
  results.append(dict(pair=pair,matched=True,checkpoint_resumes=new['checkpoint_resumes'],counts=new['counts']))
 assert read(work/'results'/checkpoint_pair/'result.json')['checkpoint_resumes']>0
 write(EXP/'regression_validation.json',dict(passed=True,interrupted_pair=checkpoint_pair,real_sigterm_resume=True,sources=results))
 print('REAL CHECKPOINT AND GOLDEN RESULTS PASS',flush=True)
finally:
 write(work/'control.json',dict(session=session,enabled=[],stop=True,heartbeat=time.time(),deadline=started+180,started=started))
 terminate(work,m,node,session)
