"""Launch only named experiment workers on node8 GPUs with ample measured memory headroom."""
import json, os, socket, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert socket.gethostname()=='dedicated-developjob-dev-machine-hi3wi'
jobs=json.loads(sys.argv[1]);launched=[]
gpus={int(x.split(',')[0]):float(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
assert all(gpus[j['gpu']]<16384 for j in jobs),gpus
for job in jobs:
    gpu=job['gpu'];worker=job['worker'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4')
    if job.get('vllm'):env['LD_LIBRARY_PATH']='/private/software/conda/envs/vllm/lib:'+env.get('LD_LIBRARY_PATH','')
    cmd=job['command'];logfile=ROOT/(worker+'.log')
    with logfile.open('x') as log:p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=dict(pid=p.pid,gpu=gpu,worker=worker,command=cmd,log=str(logfile));launched.append(row)
    (ROOT/(worker+'.process.json')).write_text(json.dumps(row,indent=2))
print(json.dumps(launched))
