"""Continue the authorized evaluation after a particular live generator exits.

Never restarts generation, training, or an unrelated process. Every stage fails
closed and records its failure; checkpoints and old results are read-only.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from vico_avatarforcing import ROOT,WORK,read,save,sha


def run(generation_pid,resume_scoring=False):
    path=Path(f'/proc/{generation_pid}/cmdline')
    if not (WORK/'avatarforcing/GENERATION_COMPLETE.json').exists():
        assert path.exists() and b'vico_avatarforcing.py' in path.read_bytes()
    with (WORK/'finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'FINISH_RUNTIME.json',dict(pid=os.getpid(),observed_generation_pid=generation_pid,
            hostname=subprocess.check_output(['hostname'],text=True).strip(),started_at=time.time()))
        while path.exists() and b'vico_avatarforcing.py' in path.read_bytes():
            time.sleep(5)
        assert (WORK/'avatarforcing/GENERATION_COMPLETE.json').exists(),'Generator exited without all43 complete'
        assert read(WORK/'avatarforcing/GENERATION_COMPLETE.json')['clips']==43
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        evaluator=ROOT/'compare/deployment/vico_unified_evaluate.py'
        def command(args,name,python=None):
            save(WORK/'pipeline_status.json',dict(stage=name,pid=os.getpid(),at=time.time()))
            with (WORK/(name+'.log')).open('w') as f:
                subprocess.run([python or sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        command([evaluator,'freeze'],'freeze_evaluation')
        handles=[];children=[]
        save(WORK/'pipeline_status.json',dict(stage='extracting_bfm_deca',workers=2,at=time.time()))
        try:
            if resume_scoring:
                samples=read(WORK/'avatarforcing/protocol.json')['samples']
                for s in samples:
                    r=read(WORK/'avatarforcing/features'/(s['id']+'.json'))
                    assert sha(WORK/'avatarforcing/scoring/avatarforcing'/(s['id']+'.listener.mp4'))==r['source_sha256']
                    for path,h in r['outputs'].items():assert sha(path)==h
                assert len(samples)==43
            else:
                for i in range(2):
                    f=(WORK/f'extract_{i}.log').open('w');handles.append(f)
                    children.append(subprocess.Popen([sys.executable,str(evaluator),'extract','--shard',str(i),'--shards','2'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT))
                codes=[c.wait() for c in children]
                assert codes==[0,0],f'Extractor failed: {codes}'
        finally:
            for c in children:
                if c.poll() is None:c.terminate();c.wait()
            for f in handles:f.close()
        command([evaluator,'images'],'image_scoring',str(ROOT/'compare/.envs/vico_eval/bin/python'))
        command([evaluator,'motions'],'motion_scoring')
        command([evaluator,'summarize'],'summary')
        command([ROOT/'scripts/build_vico43_unified_review.py'],'review_build')
        command([ROOT/'scripts/prepare_vico43_blind_review.py'],'blind_review_build')
        command([ROOT/'scripts/check_vico43_unified.py'],'numerical_and_media_audit')
        save(WORK/'PIPELINE_COMPLETE.json',dict(passed=True,at=time.time(),
            results_sha256=sha(WORK/'RESULTS.json'),browser_validation_pending=True))
        save(WORK/'pipeline_status.json',dict(stage='numerical_complete_browser_pending',at=time.time()))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--generation-pid',type=int,required=True);a.add_argument('--resume-scoring',action='store_true');args=a.parse_args()
    try:run(args.generation_pid,args.resume_scoring)
    except BaseException as e:
        save(WORK/'PIPELINE_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),at=time.time()))
        raise
