from pathlib import Path
import json,subprocess,sys,shlex,time
p=Path(__file__).resolve().parent;r=p.parents[1];py='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
base=json.loads((p/'bench_node17_8/spec.json').read_text())
def wait(names):
 start=time.time()
 while True:
  states=[json.loads((p/n/'status.json').read_text())['state'] for n in names if (p/n/'status.json').exists()]
  if len(states)==len(names) and all(s!='running' for s in states):
   if not all(s=='completed' for s in states):raise RuntimeError(f'Prerequisite failed: {names} {states}')
   return
  if time.time()-start>2400:raise TimeoutError(str(names))
  time.sleep(2)
def launch(name,spec,host=None):
 d=p/name;d.mkdir(exist_ok=False);(d/'spec.json').write_text(json.dumps(spec,indent=2));cmd=[sys.executable,str(p/'run_job.py'),str(d/'spec.json')]
 if host:
  code='import subprocess; p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); print(p.pid)'
  subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes',f'root@{host}','python -c '+shlex.quote(code)],check=True)
 else:
  q=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);print(name,q.pid,flush=True)
wait([f'bench_24_fixed_node{i}' for i in range(3)])
for rank,host in enumerate(['172.29.252.11','172.29.15.91','172.29.225.51']):
 template=json.loads((p/f'bench_24_fixed_node{rank}/spec.json').read_text());d=p/f'optimized_24v2_node{rank}'
 argv=[py,'-m','torch.distributed.run','--nnodes=3','--nproc-per-node=8',f'--node-rank={rank}','--master-addr=172.29.252.11','--master-port=29632',str(r/'scripts/p2_listen.py'),'train','--config',str(p/'optimized_24gpu/config.json'),'--arm','E1','--train-seed','20261026','--s0','--steps','5']
 env={**template['env'],'NCCL_DEBUG_FILE':str(d/'nccl.%h.%p.log')}
 launch(f'optimized_24v2_node{rank}',{**template,'argv':argv,'env':env},None if rank==0 else host)
wait([f'optimized_24v2_node{i}' for i in range(3)])
for world,arm,gpus in [(1,'E0','0'),(1,'E1','1'),(2,'E0','2,3'),(2,'E1','4,5')]:
 argv=[py]+(['-m','torch.distributed.run','--standalone','--nproc-per-node=2'] if world==2 else [])
 argv += [str(r/'scripts/p2_listen.py'),'train','--config',str(p/f'optimized_{world}gpu/config.json'),'--arm',arm,'--train-seed','20261026','--s0','--steps','7']
 if world==1 and arm=='E1':argv+=['--stop-after','6']
 launch(f'optimized_{world}gpu_{arm}',{**base,'argv':argv,'env':{**base['env'],'CUDA_VISIBLE_DEVICES':gpus}})
wait(['optimized_1gpu_E1'])
s=json.loads((p/'optimized_1gpu_E1/spec.json').read_text());argv=s['argv'][:-2]+['--resume',str(p/'optimized_1gpu/runs/s0_20261026/E1/step_000006.pt')]
launch('optimized_1gpu_E1_resume',{**s,'argv':argv})
wait(['optimized_1gpu_E0','optimized_1gpu_E1_resume','optimized_2gpu_E0','optimized_2gpu_E1'])
(p/'validation_queue_completed.json').write_text(json.dumps(dict(time=time.time())))
