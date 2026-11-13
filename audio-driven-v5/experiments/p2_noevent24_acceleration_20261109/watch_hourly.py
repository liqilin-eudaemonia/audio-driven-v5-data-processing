"""Read-only hourly observation of the adopted24 and unchanged8 GPU jobs."""
from pathlib import Path
import sys,time,json,statistics,fcntl,traceback
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path.insert(0,str(S))
import control as C
_original_path=C.path
OUT=ROOT/'experiments/p2_ltx_all_lora_noevent24_fast_20261109'
C.path=lambda arm,target=False:OUT if arm=='noevent'and target else _original_path(arm,target)

def sample():
 report=dict(at=time.time(),interval_seconds=3600,read_only=True,arms={})
 for arm in C.ARMS:
  out=C.path(arm,True);status=C.read(out/'status.json');base=C.read(out/'MIGRATION.json')['step']
  rows=C.legacy.rows_after(out,base);recent=rows[-20:];flags=[]
  try:C.legacy.check_rows(recent,arm=='full')
  except Exception as exc:flags.append(repr(exc))
  saving=status['step']%500==0 and not(out/f"step_{status['step']:06d}.complete.json").exists()
  age=time.time()-(out/'metrics.jsonl').stat().st_mtime
  mean=statistics.mean(r['seconds']for r in recent)if recent else None
  if not status['complete']and age>max(1800 if saving else 600,10*(mean or 0)):flags.append('progress_stale')
  placement=None
  if not status['complete']:
   try:
    C.WORK=W if arm=='noevent'else S
    placement=C.probe(arm)
   finally:C.WORK=S
   if not placement['healthy']:flags.append('placement_observation_not_healthy')
  report['arms'][arm]=dict(output=str(out),status=status,mean_recent_seconds=mean,recent_updates=len(recent),metrics_age_seconds=age,saving=saving,flags=flags,placement=placement)
 return report

def main():
 with(W/'watch_hourly.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  C.write(W/'HOURLY_CONTROLLER.json',dict(**C.record(),interval_seconds=3600,read_only=True))
  while not(W/'STOP_WATCH_HOURLY').exists():
   try:report=sample()
   except Exception as exc:report=dict(at=time.time(),observation_error=repr(exc),traceback=traceback.format_exc(),training_not_modified=True)
   C.write(W/'HOURLY_LATEST.json',report)
   with(W/'hourly.jsonl').open('a')as f:f.write(json.dumps(report)+'\n')
   if report.get('arms')and all(a['status']['complete']for a in report['arms'].values()):return
   time.sleep(3600)
if __name__=='__main__':main()
