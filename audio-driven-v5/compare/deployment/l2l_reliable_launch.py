"""Launch on node11 only, after GPU7 empties and preflight checks pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_l2l_reliable_retrain_20261107'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--share',action='store_true');args=parser.parse_args()
    for name in ['preflight/COMPLETE.json','boundary_checks/COMPLETE.json']:
        assert json.loads((WORK/name).read_text())['passed']
    if (WORK/'LAUNCH.json').exists():
        old=json.loads((WORK/'LAUNCH.json').read_text());pid=old['pid']
        command=Path('/proc')/str(pid)/'cmdline'
        assert not command.exists() or b'l2l_reliable.py' not in command.read_bytes(),'Training already running'
    raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    used,free=[int(v.strip()) for v in raw.split(',')]
    if args.share:assert free>=16*1024,f'GPU7 free memory insufficient: {free}MiB'
    else:assert used<128,f'GPU7 not idle: {used}MiB; no existing job touched'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',L2L_PROJECT_ROOT=str(ROOT))
    if args.share:env['L2L_GPU_MEMORY_LIMIT_GIB']='12'
    command=[str(ROOT/'compare/.envs/l2l/bin/python'),str(WORK/'code_snapshot/l2l_reliable.py'),'run','--work',str(WORK)]
    with (WORK/'train.log').open('a') as log:
        process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(pid=process.pid,host='172.29.189.119',gpu=7,command=command,sharing_authorized=args.share,
                allocator_limit_gib=12 if args.share else None,existing_memory_used_mib=used,
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),authorization='User allowed temporaryGPU7 retraining, return to inference afterwards')
    (WORK/'LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
