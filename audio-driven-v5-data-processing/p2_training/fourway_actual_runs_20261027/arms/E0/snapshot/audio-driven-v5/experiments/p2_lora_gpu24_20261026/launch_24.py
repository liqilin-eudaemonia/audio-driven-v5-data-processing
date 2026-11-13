"""Launch finite 24-rank tests after this experiment's prerequisite jobs release GPUs."""
from pathlib import Path
import json,subprocess,sys,shlex,time
p=Path(__file__).resolve().parent;r=p.parents[1];py='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
wait_names=['bench_local_4_batch8','bench_node17_8_batch8','bench_node4_maxlength']
start=time.time()
while True:
 if time.time()-start>1800:raise TimeoutError('Prerequisites did not release GPUs')
 f=p/'bench_local_1/results/profile_baseline_rank0.json'
 if f.exists():(p/'bench_local_1/STOP').touch()
 names=wait_names+['bench_local_1']
 states=[json.loads((p/n/'status.json').read_text())['state'] for n in names if (p/n/'status.json').exists()]
 if len(states)==len(names) and all(s!='running' for s in states):break
 time.sleep(2)
base=json.loads((p/'bench_node17_8/spec.json').read_text())
for rank,host in enumerate(['172.29.252.11','172.29.15.91','172.29.225.51']):
 d=p/f'bench_24_node{rank}';d.mkdir(exist_ok=True)
 argv=[py,'-m','torch.distributed.run','--nnodes=3','--nproc-per-node=8',f'--node-rank={rank}',
       '--master-addr=172.29.252.11','--master-port=29626',str(p/'benchmark_scaling.py'),
       '--config',str(p/'benchmark/config.json'),'--output',str(p/'bench_24_results'),
       '--batch','24','--steps','6','--profiles','baseline,bucket,half']
 spec={**base,'argv':argv,'timeout':1800,'env':{**base['env'],'NCCL_SOCKET_IFNAME':'eth0','NCCL_DEBUG':'INFO',
       'NCCL_DEBUG_FILE':str(d/'nccl.%h.%p.log')}}
 (d/'spec.json').write_text(json.dumps(spec,indent=2))
 cmd=[sys.executable,str(p/'run_job.py'),str(d/'spec.json')]
 if rank==0:
  q=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);print(rank,q.pid,flush=True)
 else:
  code='import subprocess,sys; p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); print(p.pid)'
  subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes',f'root@{host}','python -c '+shlex.quote(code)],check=True)
(p/'launch_24_completed.json').write_text(json.dumps(dict(launched=time.time())))
