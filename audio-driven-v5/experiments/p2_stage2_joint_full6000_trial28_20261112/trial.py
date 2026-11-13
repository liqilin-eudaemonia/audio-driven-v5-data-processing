"""Calibrated28-GPU stage2 short trial: baseline,100, and at most300 updates."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,traceback
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import isolate_events
isolate_events.activate()
import runtime28 as R
R.activate()
from p2_listen.contracts import read,atomic_json,sha
from p2_stage2_joint_v2 import evaluation as E
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'

def state(stage,**extra):
 atomic_json(W/'TRIAL_STATUS.json',dict(stage=stage,at=time.time(),**extra))

def command(args,name):
 state(name)
 with(W/(name+'.log')).open('ab',buffering=0)as f:
  code=subprocess.run([PY,'-u',str(W/'reliable_entry.py'),*args],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')).returncode
 if code:raise RuntimeError(name+' failed; inspect '+str(W/(name+'.log')))

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def wait_released():
 import p2_whisper_event_feature_train31 as native
 cl=read(R.RUN/'cluster.json');began=time.monotonic()
 while True:
  resources=native.selected_resources(cl)
  if resources['selected_all_idle']:return
  from release_own import reap
  reap()
  if time.monotonic()-began>300:raise RuntimeError('Selected28 GPUs remain occupied; preserve other tasks')
  time.sleep(5)

def comparison(c,steps,decision):
 summaries={}
 for s in steps:
  out=R.RUN/'evaluation'/f'step_{s:06d}'
  summaries[str(s)]=dict(E.verify_report(c,s)['summary'],video_metrics=read(out/'metrics/video/summary.json')['metrics'],
   audio_activity=read(out/'metrics/audio/summary.json')['audio']['activity']['acoustic'])
 result=dict(success=True,at=time.time(),world=28,batch=28,summaries=summaries,decision=decision,
  calibration=read(R.RUN/'CALIBRATION.json')['aggregate'],baseline_is_evaluated_not_retrained=True,automatic_1000_training=False,
  limitation='One frozen64-dev set and generation seed; before/after stage2 comparison, not a frozen-Qwen continuation control or statistical proof')
 atomic_json(W/'FINAL_COMPARISON.json',result)
 return result

def main():
 c=R.validate(read(R.RUN/'config.json'))
 if not read(R.RUN/'CALIBRATION.json')['passed']or not read(R.RUN/'LONGEST_SAMPLE_CHECK.json')['success']:raise ValueError('Real GPU calibration/debug required')
 base=R.RUN/'evaluation/step_000000/REPORT.json'
 pid=read(W/'STEP0_EVALUATION_LAUNCHER.json')['pid']
 state('baseline_evaluation_running',pid=pid)
 while not base.exists():
  if not alive(pid):raise RuntimeError('Baseline evaluator exited without report; preserve saved0')
  time.sleep(10)
 while alive(pid):time.sleep(2)
 E.verify_report(c,0)
 for start,finish in ((0,100),(100,300)):
  if start==100:
   decision=E.decision(c,100)
   if decision['decision']!='continue_to_300':
    comparison(c,(0,100),decision)
    state('finished_at_100',decision=decision,complete=True)
    return
  saved=R.RUN/f'step_{finish:06d}.complete.json'
  if not saved.exists():
   wait_released()
   command([str(W/'control.py'),'launch','--resume',str(R.RUN/f'step_{start:06d}.pt')],f'train_{start}_to_{finish}')
  status=read(R.RUN/'status.json')
  if status['step']!=finish:raise RuntimeError('Training stopped before milestone; preserve exact resume state')
  report=R.RUN/'evaluation'/f'step_{finish:06d}'/'REPORT.json'
  if not report.exists():command([str(W/'evaluate_overlap.py'),'--step',str(finish)],f'evaluate_{finish}')
  E.verify_report(c,finish)
 result=E.decision(c,300)
 comparison(c,(0,100,300),result)
 state('finished',complete=True,decision=result)

if __name__=='__main__':
 with(W/'trial.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except BaseException as error:
   state('failed',error=repr(error),complete=False);traceback.print_exc();raise
