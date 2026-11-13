"""Overlap independent plan judging with event CPU preparation, preserving gates."""
from pathlib import Path
import argparse,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import reliable_entry
import eval28 as E

def link_completed_checkpoint(step):
 """Mmap the unchanged complete checkpoint; do not rescan/copy its Adam.

 The native writer already SHA256-checks the entire file before committing
 its complete marker. Native inference only accesses the model tensors;
 torch.load(mmap=True) leaves optimizer storage unmapped until accessed.
 """
 out=E.folder(step);out.mkdir(parents=True,exist_ok=True)
 receipt=out/'MODEL_EXPORT.json';target=out/'model.pt'
 if receipt.exists():return
 source=E.R.RUN/f'step_{step:06d}.pt';marker=E.read(source.with_suffix('.complete.json'))
 if marker['step']!=step:raise ValueError('Wrong native complete checkpoint')
 before=E.stat(source)
 if target.exists():raise ValueError('Uncommitted model export exists; preserve it for review')
 os.link(source,target)
 if E.stat(source)!=before or source.stat().st_ino!=target.stat().st_ino:raise ValueError('Checkpoint changed during hardlink')
 E.atomic_json(receipt,dict(checkpoint_sha256=marker['sha256'],model_sha256=marker['sha256'],model_stat=before,
  method='zero-copy hardlink of native SHA-complete checkpoint; mmap optimizer tensors are not accessed',checkpoint_stat=before,
  native_complete_marker_sha256=E.sha(source.with_suffix('.complete.json')),scheduler_sha256=E.sha(__file__)))

def run(step):
 c=E.read(E.R.RUN/'config.json');out=E.folder(step)
 from trial import wait_released
 wait_released()
 link_completed_checkpoint(step)
 child=subprocess.Popen([E.PY,'-u',str(E.R.PARENT/'reliable_entry.py'),str(W/'eval28.py'),'evaluate','--step',str(step)],stdin=subprocess.DEVNULL)
 overlap=False
 while child.poll()is None:
  events=out/'events'
  status=events/'CPU_STAGE_STATUS.json'
  cue_launches=list(events.glob('cpu_candidates_launch_*'))if events.exists()else[]
  all_cpu_nodes_started=bool(cue_launches)and all((cue_launches[-1]/(n['name']+'.log')).exists()for n in E.read(E.R.RUN/'cluster.json')['nodes'])
  if not overlap and status.exists()and E.read(status)['stage']=='candidates'and all_cpu_nodes_started and not(out/'stages/plans.json').exists():
   os.kill(child.pid,signal.SIGSTOP)
   try:
    # A completed CPU stage may have raced this observation. Never overlap
    # two GPU judge models on the same allocation.
    if not(events/'inference_shards').exists():
     E.atomic_json(out/'OVERLAP.json',dict(stage='running',at=time.time(),evaluator_pid=child.pid,script_sha256=E.sha(__file__)))
     E.cluster_phase(c,step,'plans');overlap=True
     E.atomic_json(out/'OVERLAP.json',dict(stage='complete',at=time.time(),script_sha256=E.sha(__file__)))
    else:overlap=True
   finally:
    from trial import wait_released
    try:
     if overlap and not(events/'inference_shards').exists():wait_released()
    finally:os.kill(child.pid,signal.SIGCONT)
  time.sleep(3)
 if child.returncode:raise RuntimeError('Stage evaluation failed: '+str(step))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True,choices=[1000]);a=p.parse_args();run(a.step)
