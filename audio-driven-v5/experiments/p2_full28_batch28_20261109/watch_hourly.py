"""Read-only hourly monitoring of the current full28 batch28 pipeline."""
from pathlib import Path
import time,json,fcntl,statistics,traceback
import manager as X
C=X.C;W=X.W;OUT=X.OUT
C.WORK=W;old_path=C.path;C.path=lambda arm,target=False:OUT if arm=='full'and target else old_path(arm,target);C.ARMS=dict(full=28)
def sample():
 status=C.read(OUT/'status.json');base=C.read(OUT/'MIGRATION.json')['step'];rows=C.legacy.rows_after(OUT,base);recent=rows[-50:];flags=[]
 try:C.legacy.check_rows(recent,True)
 except Exception as e:flags.append(repr(e))
 placement=C.probe('full');selected={n['name']:set(n['gpus'])for n in C.read(OUT/'cluster.json')['nodes']};errors=[]
 for n in placement['nodes']:
  errors.extend(e for e in n['errors']if 'rank'in e)
  uuids={g['uuid']for g in n['gpus']if g['gpu']in selected[n['node']]}
  errors.extend(dict(foreign_pid=a['pid'],node=n['node'])for a in n['foreign']if a['uuid']in uuids and not a['allowed_vllm'])
 healthy=not errors and sum(len(n['workers'])for n in placement['nodes'])==28
 if not healthy:flags.append('selected_worker_or_resource_error')
 age=time.time()-(OUT/'metrics.jsonl').stat().st_mtime;mean=statistics.mean(r['seconds']for r in recent)if recent else None
 saving=status['step']%500==0 and not(OUT/f"step_{status['step']:06d}.complete.json").exists()
 if not status['complete']and age>max(1800 if saving else 600,10*(mean or 0)):flags.append('progress_stale')
 return dict(at=time.time(),interval_seconds=3600,output=str(OUT),status=status,mean_recent_seconds=mean,metrics_age_seconds=age,saving=saving,flags=flags,placement=placement,selected_healthy=healthy,resource_errors=errors)
def main():
 with(W/'hourly.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);C.write(W/'HOURLY_CONTROLLER.json',dict(**C.record(),interval_seconds=3600,read_only=True))
  while not(W/'STOP_MONITOR').exists():
   try:r=sample()
   except Exception as e:r=dict(at=time.time(),observation_error=repr(e),traceback=traceback.format_exc(),training_not_modified=True)
   C.write(W/'HOURLY_LATEST.json',r)
   with(W/'hourly.jsonl').open('a')as f:f.write(json.dumps(r)+'\n')
   if r.get('status',{}).get('complete'):return
   time.sleep(3600)
if __name__=='__main__':main()
