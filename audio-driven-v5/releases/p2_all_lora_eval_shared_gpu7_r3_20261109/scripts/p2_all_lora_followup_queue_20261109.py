"""Sequential GPU7-only checkpoint evaluations after the active5000 evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON

WORK=ROOT/'experiments/p2_ltx_all_lora_followups_gpu7_20261109'
CLI=ROOT/'scripts/p2_all_lora_followups_gpu7_overlay_20261109.py'
HELPER=ROOT/'src/p2_all_lora_followups_eval_20261109.py'
UPSTREAM=ROOT/'experiments/p2_ltx_all_lora5000_seamless200_sft2752_eval4_20261109'


def completed(work,step):
    if (work/'STOP').exists():return False
    status=work/'status.json'
    if not status.exists() or read(status).get('stage')!='complete':return False
    p=read(work/'protocol.json');a=read(work/'FINAL_AUDIT.json');h=read(work/'HANDOFF_COMPLETE.json')
    assert a['success'] and h['success'] and a['step']==step and a['outputs']==h['outputs']==200
    assert p['seed']==137 and p['arms'][0]['step']==step and list(p['datasets'])==['seamless']
    result=read(work/'RESULTS.json')[p['arms'][0]['name']+'/seamless']
    assert result['requested_pairs']==200 and result['audio']['pairs_with_errors']==0
    assert all(result['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    # Wait for the upstream controller to release ownership before reusing GPU7.
    with (work/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def run_child(command,work,logname,step,phase):
    guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
    cmd=['env','PYTHONPATH=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=',
         str(PYTHON),str(guard),'--guard',*command]
    with (WORK/logname).open('ab',buffering=0) as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
    save(WORK/'active.json',dict(pid=proc.pid,step=step,phase=phase,at=time.time(),command=cmd))
    next_status=0
    try:
        while proc.poll() is None:
            if (WORK/'STOP').exists():
                if phase=='evaluation':(work/'STOP').write_text('Followup queue stopped by user\n')
                return False
            if time.time()>=next_status:
                state=read(work/'status.json') if (work/'status.json').exists() else None
                save(WORK/'status.json',dict(stage=phase,step=step,at=time.time(),child=state,physical_gpus=[7]))
                next_status=time.time()+20
            time.sleep(1)
        if proc.returncode:raise RuntimeError(f'{phase} step{step} failed with {proc.returncode}; see {logname}')
        return True
    finally:
        if proc.stdin:proc.stdin.close()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # This is our local guard only, never an unrelated training process.
            proc.terminate();proc.wait(timeout=10)


def main(work):
    global WORK
    WORK=work.resolve();WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(WORK/'PLAN.json')
        assert plan['steps']==[5500,7000,7500] and plan['physical_gpus']==[7] and plan['node']=='node11'
        for path,digest in plan['code'].items():assert sha(path)==digest,'Queue source changed: '+path
        anchor=read(UPSTREAM/'protocol.json')
        assert anchor['arms'][0]['step']==5000 and anchor['arms'][0]['kind']=='ltx_all_lora_full'
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),node=os.uname().nodename,physical_gpus=[7]))
        # CPU work runs ahead while the current5000 process keeps using GPU7.
        for row in plan['evaluations']:
            cp=Path(row['checkpoint']);assert read(cp.with_suffix('.complete.json'))['sha256']==row['checkpoint_sha256']
            child=Path(row['work']);child.mkdir(exist_ok=True)
            if not run_child([str(PYTHON),'-u',str(CLI),'prepare','--step',str(row['step'])],child,
                             f'prepare_{row["step"]}.log',row['step'],'cpu_preparation'):return
            p=read(child/'protocol.json')
            for key in ['datasets','seed','generation','planner_adapter_identity','whisper_encoder','feature_code','sampling','shared_items_sha256']:
                assert p[key]==anchor[key],'Different evaluation conditions: '+key
            assert p['arms'][0]['original_sha256']==row['checkpoint_sha256']
            assert read(child/'cluster.json')['nodes'][0]['gpus']==[7]
        save(WORK/'PREPARED_ALL.json',dict(success=True,at=time.time(),steps=plan['steps'],gpu_used=False))
        previous=UPSTREAM;previous_step=5000
        for row in plan['evaluations']:
            step=row['step'];child=Path(row['work'])
            while not completed(previous,previous_step):
                if (WORK/'STOP').exists():return
                save(WORK/'status.json',dict(stage='waiting_upstream',at=time.time(),upstream_step=previous_step,
                     next_step=step,upstream_status=read(previous/'status.json') if (previous/'status.json').exists() else None,
                     physical_gpus=[7],upstream_stopped=(previous/'STOP').exists()))
                time.sleep(10)
            if (WORK/'STOP').exists():return
            if not completed(child,step):
                if (child/'STOP').exists():raise RuntimeError('Followup explicitly stopped: '+str(child))
                p=read(child/'protocol.json');assert p['arms'][0]['original_sha256']==row['checkpoint_sha256']
                save(child/'RUNTIME_GPU7.json',dict(at=time.time(),node='node11',physical_gpus=[7],
                     logical_metric_shards=4,script=str(CLI),script_sha256=sha(CLI),
                     protocol_sha256=sha(child/'protocol.json'),parent_queue=str(WORK)))
                if not run_child([str(PYTHON),'-u',str(CLI),'queue','--step',str(step)],child,
                                 f'evaluate_{step}.log',step,'evaluation'):return
                assert completed(child,step)
            save(WORK/f'COMPLETE_{step}.json',dict(success=True,at=time.time(),step=step,outputs=200,work=str(child)))
            previous=child;previous_step=step
        save(WORK/'status.json',dict(stage='complete',at=time.time(),steps=plan['steps'],outputs=600,physical_gpus=[7]))
        save(WORK/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),steps=plan['steps'],outputs=600))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,default=WORK);a=parser.parse_args()
    try:main(a.work)
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
    finally:
        if (a.work/'STOP').exists():save(a.work/'status.json',dict(stage='stopped',at=time.time(),physical_gpus=[7]))
