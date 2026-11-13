"""Four isolated text-only 30B workers; never touches GPUs 4..7 or labeling jobs."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import acquire,write

PYTHON='/private/software/conda/envs/vllm/bin/python'
ARMS=['all65']


def main():
    owner=acquire(WORK/'controller.lock',False)
    if owner is None:raise SystemExit('Already running')
    if (WORK/'run_manifest.json').exists():raise SystemExit('Existing experiment; inspect/recover explicit workers, do not relaunch all')
    gpu_info=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(line.split(',')[1])>1500 for line in gpu_info.splitlines()):raise RuntimeError('Requested GPU occupied; no automatic killing')
    env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    started=time.time();manifest={'started_at':started,'controller_pid':os.getpid(),'model':'/private/ckpts/Qwen3-30B-A3B-Instruct-2507',
        'settings':{'batch':64,'memory':.8,'max_batched_tokens':32768,'tensor_parallel_size':1,'attempts':3},
        'arms':{},'raw_annotations_modified':False,'training_cache_updated':False}
    with ExitStack() as stack:
        processes={}
        for gpu,arm in enumerate(ARMS,start=1):
            log=stack.enter_context((WORK/f'{arm}.log').open('a'))
            cmd=[PYTHON,str(ROOT/'scripts/run_listener_semantic_constrained_groups.py'),'--inputs',str(WORK/f'{arm}.jsonl'),
                 '--output',str(WORK/'runs'/arm),'--model',manifest['model'],'--batch','64','--memory','.8','--attempts','3','--decode-graph']
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
            processes[arm]=proc
            manifest['arms'][arm]={'gpu':gpu,'pid':proc.pid,'start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                                   'command':cmd,'input_sha256':hashlib.sha256((WORK/f'{arm}.jsonl').read_bytes()).hexdigest()}
        write(WORK/'run_manifest.json',manifest)
        while True:
            states={arm:{'pid':proc.pid,'exit_code':proc.poll()} for arm,proc in processes.items()}
            for arm,s in states.items():
                p=WORK/'runs'/arm/'progress.json'
                if p.exists():s['progress']=json.loads(p.read_text())
            write(WORK/'status.json',{'at':time.time(),'elapsed_s':time.time()-started,'workers':states})
            try:
                telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
                with (WORK/'telemetry.jsonl').open('a') as f:f.write(json.dumps({'at':time.time(),'gpu_csv':telemetry})+'\n')
            except subprocess.SubprocessError:pass
            if all(s['exit_code'] is not None for s in states.values()):break
            time.sleep(20)
        summary={}
        for arm,proc in processes.items():
            run=WORK/'runs'/arm;entry={'exit_code':proc.returncode}
            if (run/'summary.json').exists():entry['model_summary']=json.loads((run/'summary.json').read_text())
            if proc.returncode==0:
                cmd=[PYTHON,str(ROOT/'scripts/audit_listener_semantic_prompt.py'),'--inputs',str(WORK/f'{arm}.jsonl'),'--run',str(run)]
                with (WORK/f'{arm}.audit.log').open('w') as f:
                    audit=subprocess.run(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
                entry['audit_exit_code']=audit.returncode
            summary[arm]=entry
        write(WORK/'finished.json',{'at':time.time(),'elapsed_s':time.time()-started,'arms':summary,
                                    'scope':'Model/program acceptance only; not human ground truth'})
    owner.close()


if __name__=='__main__':main()
