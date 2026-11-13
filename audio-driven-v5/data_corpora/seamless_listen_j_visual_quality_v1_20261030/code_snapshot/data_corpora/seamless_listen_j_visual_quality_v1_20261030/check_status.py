"""One-shot bounded inspection of both quality/annotation stages; no task mutations."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent

def inspect(item):
    node,cfg=item
    code='WORK='+repr(cfg['quality_work'])+'\n'+r'''
import json,os,time
from pathlib import Path
w=Path(WORK)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def cpu(c):
 try:
  fields=Path('/proc/%s/stat'%c['pid']).read_text().rsplit(')',1)[1].split()
  if fields[0]=='Z' or fields[19]!=str(c.get('start_ticks')):return None
  return (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
 except (FileNotFoundError,KeyError):return None
s=read(w/'controller.json');q=read(w/'queues/quality/state.json');cfg=read(w/'pipeline_config.json')
children=s.get('children',{})
before={n:cpu(c) for n,c in children.items()};tick=time.monotonic();time.sleep(3)
after={n:cpu(c) for n,c in children.items()};elapsed=time.monotonic()-tick
cores=sum(max(0,after[n]-v) for n,v in before.items() if v is not None and after[n] is not None)/elapsed
out={'at':time.time(),'phase':s.get('phase'),'controller_live':cpu(s) is not None,
     'quality_workers_live':sum(v is not None for v in after.values()),'quality_cpu_cores_used':cores,
     'cpu_quota':cfg.get('cpu_quota'),'quality_cpu_quota_percent':100*cores/cfg.get('cpu_quota',1),
     'quality_completed_sources':q.get('completed',0),'quality_completed_clips':q.get('completed_clips',0),
     'quality_failed_sources':len(q.get('failed',{})),'inflight_sources':len(q.get('inflight',{})),
     'scope_clips':cfg.get('scope_clips'),'quality_complete':read(w/'quality_complete.json')}
if s.get('phase')=='annotation_resumed':
 a=Path(cfg['annotation_work']);out['annotation']={}
 for b in ('vlm','nod'):
  state=read(a/'queues'/b/'state.json');out['annotation'][b]={k:state.get(k) for k in ('completed','completed_clips')}
print(json.dumps(out))
'''
    sock='/tmp/listen-node17-speed-ssh.sock' if node=='node17' else '/tmp/listen-node8-annotation-ssh.sock'
    command=['ssh','-S',sock,'-F','/dev/null','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519',
             '-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+cfg['host'],'python3 -']
    try:
        r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=35,check=True)
        return node,json.loads(r.stdout)
    except Exception as e:return node,{'inspection_error':str(e)}

if __name__=='__main__':
    scope=json.loads((BASE/'nodes.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(2) as pool:nodes=dict(pool.map(inspect,scope['nodes'].items()))
    out={'at':time.time(),'scope_clips':scope['total_clips'],'nodes':nodes,
         'completed_quality_clips':sum(v.get('quality_completed_clips',0) for v in nodes.values()),
         'note':'Completed source receipts; inflight clips are not counted. CPU sample uses only recorded local quality workers.'}
    temp=BASE/'status.tmp';temp.write_text(json.dumps(out,indent=2)+'\n');temp.replace(BASE/'status.json')
    print(json.dumps(out,indent=2))
