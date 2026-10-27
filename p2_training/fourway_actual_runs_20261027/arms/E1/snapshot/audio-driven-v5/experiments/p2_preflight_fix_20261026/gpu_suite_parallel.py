"""Finite, three-host real-model comparison. Stops only this suite's jobs on failure."""
import json,os,shlex,subprocess,sys,time,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
HOSTS=['172.29.252.11','172.29.15.91','172.29.225.51'];KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def admission(rank):
 cmd=['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits']
 if rank:cmd=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+HOSTS[rank],shlex.join(cmd)]
 rows=subprocess.check_output(cmd,text=True,timeout=20).strip().splitlines()
 memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows}
 if any(memory[i]>=1024 for i in (0,1)):raise RuntimeError(f'GPU busy on node {rank}: {memory}')
 return memory
cases=[('baseline','E0',None,None),('baseline','E1',None,None),('fast','E0',None,None),('fast','E1',None,None),
       ('resume','E1',3,None),('resume','E1',None,OUT/'resume/runs/s0_20261023/E1/step_000003.pt')]
completed=[]
for index,(kind,arm,stop,resume) in enumerate(cases):
 if index<2:continue
 with ThreadPoolExecutor(max_workers=3) as pool:gpus=list(pool.map(admission,range(3)))
 name=f'{index:02d}_{kind}_{arm}';jobs=[];children=[];session=uuid.uuid4().hex
 cli=ROOT/'scripts/p2_listen.py' if kind=='baseline' else ROOT/'releases/p2_preflight_v1/scripts/p2_listen.py'
 try:
  for rank in range(3):
   dest=OUT/'jobs_parallel'/name/f'node{rank}';dest.mkdir(parents=True,exist_ok=False);jobs.append(dest)
   cmd=[str(PY),'-u','-m','torch.distributed.run','--nnodes=3','--nproc-per-node=2',f'--node-rank={rank}',
        '--master-addr='+HOSTS[0],f'--master-port={29862+index}','--max-restarts=0',str(cli),'train','--config',str(OUT/kind/'config.json'),
        '--arm',arm,'--train-seed','20261023','--s0','--steps','5']
   if stop:cmd+=['--stop-after',str(stop)]
   if resume:cmd+=['--resume',str(resume)]
   shell='source '+shlex.quote(str(ROOT/'configs/p2_listen_h200_multinode.env'))+'\nexec '+shlex.join(cmd)
   spec=dict(argv=['/bin/bash','-c',shell],cwd=str(ROOT),timeout=1200,
             env=dict(CUDA_VISIBLE_DEVICES='0,1',P2_PREFLIGHT_SESSION=session),admission=gpus[rank])
   write(dest/'spec.json',spec)
   launcher=[str(PY),str(OUT/'run_job.py'),str(dest/'spec.json')]
   if rank:launcher=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+HOSTS[rank],shlex.join(launcher)]
   children.append(subprocess.Popen(launcher,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT))
  write(OUT/'gpu_suite_status.json',dict(state='running',case=name,completed=completed,started=time.time()))
  while True:
   states=[json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {} for p in jobs]
   if any(s.get('state')=='failed' for s in states):raise RuntimeError('GPU case failed: '+name)
   if all(s.get('state')=='completed' for s in states):break
   if any(child.poll() not in (None,0) for child in children):raise RuntimeError('GPU SSH wrapper failed: '+name)
   time.sleep(2)
  for child in children:child.wait(timeout=20)
  completed.append(dict(case=name,states=states))
 except BaseException as error:
  for p in jobs:(p/'STOP').touch()
  for child in children:
   try:child.wait(timeout=40)
   except subprocess.TimeoutExpired:child.terminate()
  write(OUT/'gpu_suite_status.json',dict(state='failed',case=name,completed=completed,error=str(error)))
  raise
write(OUT/'gpu_suite_status.json',dict(state='complete',completed=completed,finished=time.time()))
