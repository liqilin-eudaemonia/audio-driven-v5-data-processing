import sys,json,time,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent
sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote
node=sys.argv[1];gpus=list(map(int,sys.argv[2:]))
code='''import os,json,time,signal
from pathlib import Path
W=Path(WORK);out=[]
for gpu in GPUS:
 p=W/'workers'/f'{NODE}-gpu-{gpu}.json';s=json.loads(p.read_text());pid=s['pid']
 if '+bounded-crop-v1' in s.get('execution_revision',''):
  out.append(dict(gpu=gpu,action='already_updated',pid=pid));continue
 proc=Path('/proc')/str(pid)
 if not proc.exists():out.append(dict(gpu=gpu,action='already_exited'));continue
 cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert 'listen_j_omni.py worker' in cmd and str(W) in cmd and ('--node '+NODE) in cmd and ('--gpu '+str(gpu)) in cmd,cmd
 ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
 os.kill(pid,signal.SIGTERM);out.append(dict(gpu=gpu,pid=pid,start_ticks=ticks,action='graceful_roll_requested'))
print(json.dumps(dict(node=NODE,at=time.time(),actions=out)))
'''
code='WORK='+repr(str(W))+'; NODE='+repr(node)+'; GPUS='+repr(gpus)+'\n'+code
r=remote(node,code);p=D/f'roll-{node}-{time.time_ns()}.json';p.write_text(json.dumps(r,indent=2));print(json.dumps(r))
