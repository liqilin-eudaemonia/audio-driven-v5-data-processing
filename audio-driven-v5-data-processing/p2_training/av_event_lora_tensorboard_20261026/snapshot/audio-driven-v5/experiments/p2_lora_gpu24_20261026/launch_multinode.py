"""Read-only/finite test launcher, invocation stored per node."""
import argparse,json,subprocess,sys,shlex
from pathlib import Path
p=Path(__file__).resolve().parent
args=argparse.ArgumentParser();args.add_argument('--name',required=True);args.add_argument('--probe',action='store_true');args.add_argument('--port',type=int,required=True);args.add_argument('--env',default='{}');a=args.parse_args()
base=json.loads((p/'bench_node17_8/spec.json').read_text());py=base['argv'][0]
for rank,host in enumerate(['172.29.252.11','172.29.15.91','172.29.225.51']):
 d=p/f'{a.name}_node{rank}';d.mkdir(exist_ok=False)
 argv=[py,'-m','torch.distributed.run','--nnodes=3','--nproc-per-node=8',f'--node-rank={rank}','--master-addr=172.29.252.11',f'--master-port={a.port}']
 if a.probe:argv += [str(p/'network_probe.py')]
 else:argv += [str(p/'benchmark_scaling.py'),'--config',str(p/'benchmark/config.json'),'--output',str(p/f'{a.name}_results'),'--batch','24','--steps','6','--profiles','baseline,bucket,half']
 env={**base['env'],'NCCL_SOCKET_IFNAME':'eth0','NCCL_DEBUG':'INFO','NCCL_DEBUG_FILE':str(d/'nccl.%h.%p.log'),'PROBE_OUTPUT':str(p/f'{a.name}_results'),**json.loads(a.env)}
 spec={**base,'argv':argv,'env':env,'timeout':300 if a.probe else 1800};(d/'spec.json').write_text(json.dumps(spec,indent=2))
 cmd=[sys.executable,str(p/'run_job.py'),str(d/'spec.json')]
 if rank==0:
  q=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);print(rank,q.pid,flush=True)
 else:
  code='import subprocess; p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); print(p.pid)'
  subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes',f'root@{host}','python -c '+shlex.quote(code)],check=True)
