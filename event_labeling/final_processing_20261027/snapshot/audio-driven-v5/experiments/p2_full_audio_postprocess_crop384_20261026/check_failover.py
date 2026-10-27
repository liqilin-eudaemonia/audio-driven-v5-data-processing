"""Terminate only the verified full-pipeline leader; require successful standby tick."""
import sys,json,time,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_audio_full.cluster import call,HOSTS,KEY,PYTHON,TRAIN_QUEUE,SCRIPT
from p2_listen.contracts import read,atomic_json
RUN=Path(__file__).resolve().parent
state=read(RUN/'status.json');leader=state['leader'];rank=leader['rank'];pid=leader['pid']
code=f'''import pathlib,os,signal
p={pid}
a=[x.decode() for x in pathlib.Path(f'/proc/{{p}}/cmdline').read_bytes().split(b'\\0') if x]
assert {str(SCRIPT)!r} in a and 'daemon' in a and a[a.index('--run')+1]=={str(RUN)!r}
assert a[a.index('--node-rank')+1]=={str(rank)!r}
os.kill(p,signal.SIGTERM)
print('verified leader terminated')'''
cmd=['ssh','-i',KEY,'-o','UserKnownHostsFile='+str(TRAIN_QUEUE/'queue_known_hosts'),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','root@'+HOSTS[rank],shlex.join([str(PYTHON),'-c',code])]
started=time.time();subprocess.run(cmd,check=True,timeout=30)
while time.time()-started<120:
 state=read(RUN/'status.json')
 if state.get('leader',{}).get('pid')!=pid and state.get('updated',0)>started and state.get('last_error') is None and state.get('phase')=='cpu_prepare_waiting_E1':break
 time.sleep(2)
else:raise TimeoutError('No successful standby takeover')
result={'status':'passed','terminated_leader':leader,'takeover_leader':state['leader'],'seconds':time.time()-started,'verified_successful_cluster_tick':True,'gpu_authorized':state.get('gpu_authorized',False)}
result['restored_standby']=call(rank,['daemon','--run',RUN,'--node-rank',rank,'--detach'])
atomic_json(RUN/'failover_validation.json',result);print(json.dumps(result),flush=True)
