"""Rescore baseline and100 with one latest revision; then resume the trial."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import eval28 as E
from trial import command,state

def main():
 c=E.read(E.R.RUN/'config.json')
 for step in(0,100):
  report=E.folder(step)/'REPORT.json'
  if not report.exists():
   for attempt in range(5):
    try:
     command([str(W/'evaluate_overlap.py'),'--step',str(step)],f'latest_rescore_{step}')
     break
    except RuntimeError:
     launches=sorted((E.folder(step)/'events').glob('parallel_launch_*'))
     transient=bool(launches)and any('kex_exchange_identification'in p.read_text(errors='replace')for p in launches[-1].glob('*.log'))
     if not transient or attempt==4:raise
     state('retry_latest_scoring_transport',step=step,attempt=attempt+1)
     time.sleep(10)
  E.E.verify_report(c,step)
 E.atomic_json(W/'LATEST_RESCORING_ACCEPTED.json',dict(success=True,at=time.time(),revision=E.read(W/'LATEST_EVALUATION_ACTIVE.json'),
  reports={str(s):E.sha(E.folder(s)/'REPORT.json')for s in(0,100)},generated_media_reused=True,training_repeated=False))
 command([str(W/'trial.py')],'trial_after_latest_rescore')

if __name__=='__main__':
 with(W/'latest_rescore.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except BaseException as error:
   state('latest_rescore_failed',error=repr(error),complete=False);raise
