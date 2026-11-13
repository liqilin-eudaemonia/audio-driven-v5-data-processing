"""Read-only three-node/19-GPU status and fixed acceleration checks."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parent


def inspect(item):
    name,cfg=item
    code='WORK='+repr(cfg['work'])+'\nGPU_IDS='+repr(cfg['gpu_ids'])+'\n'+r'''
from pathlib import Path
import json,time,subprocess
w=Path(WORK)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def live(c):
 try:
  a=Path('/proc/%s/stat'%c['pid']).read_text().rsplit(')',1)[1].split()
  return a[0]!='Z' and a[19]==str(c.get('start_ticks'))
 except (FileNotFoundError,KeyError):return False
s=read(w/'supervisor.json');watch=read(w/'watchdog.json');gpus={}
for i in GPU_IDS:
 name='gpu-'+str(i);c=s.get('children',{}).get(name,{});h=read(w/'workers'/f'{name}.json');e=h.get('execution',{})
 gpus[name]={'pid':c.get('pid'),'live':live(c),'heartbeat_matches':h.get('pid')==c.get('pid'),
    'stage':h.get('stage'),'completed_clips':h.get('completed_clips',0),'errors':h.get('failed_clips',0),
    'last_progress_at':h.get('last_progress_at'),'execution':{k:e.get(k) for k in
     ('batch_size','tokens','frames','decode_backend','renderer_workers','rolling_annotation_groups','memory')}}
queues={}
for b in ('vlm','nod'):
 q=read(w/'queues'/b/'state.json')
 queues[b]={k:q.get(k,0) for k in ('completed','completed_clips','offset')}
 queues[b].update(inflight=len(q.get('inflight',{})),failed=len(q.get('failed',{})))
raw=(subprocess.check_output(['nvidia-smi','-i',','.join(map(str,GPU_IDS)),
 '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20) if GPU_IDS else '')
print(json.dumps({'at':time.time(),'supervisor_live':live(s),'watchdog_live':live(watch),
 'supervisor_error':s.get('error'),'gpu_workers':gpus,'gpu_csv':raw,'queues':queues,
 'nod_workers_live':sum(live(c) for n,c in s.get('children',{}).items() if n.startswith('nod-'))}))
'''
    if name=='node12':cmd=[sys.executable,'-']
    else:
        sock='/tmp/listen-node17-speed-ssh.sock' if name=='node17' else '/tmp/listen-node8-annotation-ssh.sock'
        cmd=['ssh','-S',sock,'-F','/dev/null','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519',
             '-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+cfg['host'],'python3 -']
    try:
        r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=40,check=True)
        return name,json.loads(r.stdout)
    except Exception as e:return name,{'inspection_error':str(e)}


if __name__=='__main__':
    scope=json.loads((BASE/'nodes.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(3) as pool:nodes=dict(pool.map(inspect,scope['nodes'].items()))
    report={'at':time.time(),'total_clips':scope['total_clips'],'configured_gpus':sum(len(n['gpu_ids']) for n in scope['nodes'].values()),'nodes':nodes,
        'live_gpu_workers':sum(sum(w['live'] for w in n.get('gpu_workers',{}).values()) for n in nodes.values()),
        'completed_clips_by_branch':{b:sum(n.get('queues',{}).get(b,{}).get('completed_clips',0) for n in nodes.values()) for b in ('vlm','nod')},
        'note':'Source receipt counts per branch; not their intersection or label quality.'}
    temp=BASE/'status.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(BASE/'status.json')
    print(json.dumps({'live_gpu_workers':report['live_gpu_workers'],'completed_clips_by_branch':report['completed_clips_by_branch'],
        'nodes':{n:{k:s.get(k) for k in ('supervisor_live','watchdog_live','nod_workers_live','queues','gpu_csv')} for n,s in nodes.items()}},indent=2))
