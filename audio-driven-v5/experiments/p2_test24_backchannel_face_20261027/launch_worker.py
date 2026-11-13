"""Explicit bounded workers; does not stop or supervise unrelated GPU jobs."""
import os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
branch=sys.argv[1];gpu=int(sys.argv[2]);rank=int(sys.argv[3]);memory=sys.argv[4]
node=sys.argv[5] if len(sys.argv)>5 else 'node17'
worker=f'{node}-g{gpu}-{branch}'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
python='/private/software/conda/envs/'+('ovi' if branch=='asr' else 'vllm')+'/bin/python'
if branch!='asr':env['LD_LIBRARY_PATH']='/private/software/conda/envs/vllm/lib:'+env.get('LD_LIBRARY_PATH','')
if branch=='semantic':
 command=[python,str(ROOT/'scripts/run_listen_full_fusion_semantic.py'),'--run',str(RUN/'fusion'),'--worker',worker,'--rank',str(rank),'--batch','128','--memory',memory]
else:
 command=[python,str(ROOT/'scripts/run_listen_full_raw.py'),'--run',str(RUN/'raw_run'),'--branch',branch,'--worker',worker,'--rank',str(rank),'--batch','16' if branch!='asr' else '32','--prepare-workers','4','--memory',memory]
log=RUN/'raw_run/logs'/f'{worker}.log'
with log.open('a') as f:
 p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(RUN/f'launch-{worker}.json').write_text(json.dumps(dict(pid=p.pid,command=command,gpu=gpu,memory_fraction=float(memory),log=str(log),utc=time.time()),indent=2))
print(json.dumps(dict(worker=worker,pid=p.pid,log=str(log))))
