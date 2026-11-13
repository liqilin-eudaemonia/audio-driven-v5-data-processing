"""One-shot two-node inspection. Only recorded PIDs and fixed state files.

This does not start jobs, optimize parameters, or enable periodic messages.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

WORK=Path(__file__).resolve().parent


def inspect_node(item):
    name,node=item
    code='WORK='+repr(node['work'])+'\n'+r'''
from pathlib import Path
import json,time,subprocess
w=Path(WORK)
def read(name):return json.loads((w/name).read_text())
def live(d):
    if not d or not d.get('pid'):return False
    try:
        f=Path('/proc/%s/stat'%d['pid']).read_text().rsplit(')',1)[1].split()
        return f[0]!='Z' and f[19]==str(d.get('start_ticks'))
    except FileNotFoundError:return False
s=read('supervisor.json');watch=read('watchdog.json')
workers={}
for n,d in s.get('children',{}).items():
    if not n.startswith(('gpu-','nod-')):continue
    p=w/'workers'/f'{n}.json';h=json.loads(p.read_text()) if p.exists() else {}
    workers[n]={'pid':d['pid'],'live':live(d),'heartbeat_matches':h.get('pid')==d['pid'],
                'stage':h.get('stage'),'completed_clips':h.get('completed_clips',0),
                'heartbeat_age_s':time.time()-h.get('heartbeat',time.time()),
                'errors':h.get('errors',h.get('failed_clips',0))}
q={}
for b in ['vlm','nod']:
    p=w/'queues'/b/'state.json';d=json.loads(p.read_text()) if p.exists() else {}
    q[b]={k:d.get(k) for k in ['completed_clips','completed','failed','offset']}
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
print(json.dumps({'at':time.time(),'host':__import__('socket').gethostname(),'supervisor_live':live(s),
    'watchdog_live':live(watch),'supervisor_error':s.get('error'),'watchdog_error':watch.get('error'),
    'queues':q,'workers':workers,'gpus_csv':gpus}))
'''
    socket='/tmp/listen-node17-speed-ssh.sock' if name=='node17' else '/tmp/listen-node8-annotation-ssh.sock'
    command=['ssh','-S',socket,'-F','/dev/null','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519',
             '-o','BatchMode=yes','-o','ConnectTimeout=10',f"root@{node['host']}",'python3 -']
    try:
        r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=50,check=True)
        return name,json.loads(r.stdout)
    except Exception as exc:return name,{'inspection_error':str(exc)}


def main():
    nodes=json.loads((WORK/'nodes.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:states=dict(pool.map(inspect_node,nodes['nodes'].items()))
    report={'at':time.time(),'target_clips':nodes['total_clips'],'nodes':states}
    complete=all('queues' in n for n in states.values())
    report['combined_completed_clips']={b:sum((n['queues'][b].get('completed_clips') or 0) for n in states.values())
                                        for b in ['vlm','nod']} if complete else None
    report['scope']='Queue receipt counts per branch; not intersection count or annotation quality.'
    p=WORK/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
    print(json.dumps({'combined_completed_clips':report['combined_completed_clips'],
        'nodes':{n:({'supervisor_live':s['supervisor_live'],'watchdog_live':s['watchdog_live'],
                    'live_workers':sum(v['live'] for v in s['workers'].values()),'queues':s['queues'],
                    'gpus_csv':s['gpus_csv']} if 'queues' in s else s) for n,s in states.items()}},indent=2))


if __name__=='__main__':main()
