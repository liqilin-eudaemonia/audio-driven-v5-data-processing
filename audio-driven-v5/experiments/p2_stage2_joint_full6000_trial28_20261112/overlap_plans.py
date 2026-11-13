"""One-shot overlap of baseline text judging with already running CPU cues."""
from pathlib import Path
import json,os,signal,sys,time,traceback
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import eval28 as E
pid=E.read(W/'STEP0_EVALUATION_LAUNCHER.json')['pid']
try:
 c=E.read(E.R.RUN/'config.json')
 E.atomic_json(W/'OVERLAP_PLANS.json',dict(stage='running',at=time.time(),paused_evaluator=pid))
 E.cluster_phase(c,0,'plans')
 E.atomic_json(W/'OVERLAP_PLANS.json',dict(stage='complete',at=time.time(),paused_evaluator=pid))
except BaseException as error:
 E.atomic_json(W/'OVERLAP_PLANS.json',dict(stage='failed',error=repr(error),at=time.time()));traceback.print_exc();raise
finally:
 os.kill(pid,signal.SIGCONT)
