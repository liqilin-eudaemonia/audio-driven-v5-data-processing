"""Second measured concurrency step; unchanged generation in four extra workers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/p2_compare_responsenet_random200_20261106/dystream'
SCRIPT=ROOT/'compare/deployment/responsenet_dystream_capacity.py'


def save(name,value):
    path=WORK/name;temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


with (WORK/'capacity12.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((WORK/'CAPACITY_CONTROL.json').read_text());assert config['enabled']
    processes=[];handles=[];start=time.time()
    try:
        for rank in range(4):
            log=(WORK/f'capacity12_worker_{rank:02d}.log').open('a');handles.append(log)
            processes.append(subprocess.Popen([sys.executable,str(SCRIPT),'worker','--work',str(WORK)],
                stdout=log,stderr=subprocess.STDOUT))
        save('CAPACITY12_LAUNCH.json',dict(pid=os.getpid(),workers=[p.pid for p in processes],
            target_total_workers=12,at=start))
        while any(p.poll() is None for p in processes):
            codes=[p.poll() for p in processes]
            memory=float(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],text=True).strip())
            if memory>130000 or any(c not in (None,0) for c in codes):
                config['enabled']=False;save('CAPACITY_CONTROL.json',config)
                save('CAPACITY12_GUARD.json',dict(memory_mib=memory,codes=codes,at=time.time()))
            save('capacity12_status.json',dict(active=[p.pid for p in processes if p.poll() is None],
                codes=codes,memory_mib=memory,at=time.time()))
            time.sleep(3)
        save('CAPACITY12_COMPLETE.json',dict(codes=[p.returncode for p in processes],seconds=time.time()-start,at=time.time()))
    finally:
        for f in handles:f.close()
