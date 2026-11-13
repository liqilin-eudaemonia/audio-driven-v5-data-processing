"""Read-only identity checks and scoped operations for the current full31 run."""
from pathlib import Path
import sys,os,json,time,importlib.util,concurrent.futures,subprocess,math,statistics,csv
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');WORK=Path(__file__).resolve().parent
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from listen_cluster import transport
spec=importlib.util.spec_from_file_location('frozen_observation_helpers',WORK/'observations_frozen.py');OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
OLD.PROBE=OLD.PROBE.replace("if len(a)==3 and a[0].isdigit():", "if len(a)==3 and a[0].isdigit() and a[1] in {c['uuid']for c in cards}:")
OLD.PROBE=OLD.PROBE.replace("gpu_context_present=any(a['pid']==pid and a['uuid']==uu for a in apps)","memory_cap_gib=env.get('P2_TRAIN_MEMORY_GIB'),hca=env.get('NCCL_IB_HCA'),gpu_context_present=any(a['pid']==pid and a['uuid']==uu for a in apps)")

def alive(record):return OLD.alive(record)
def tail():return OLD.tail(OUT)
def event(kind,**kw):
 v=dict(at=time.time(),kind=kind,**kw)
 with(WORK/'events.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)

def probe():
 v=OLD.probe(OUT);v['reserved_gpu']='node11:0'
 for n in v['nodes']:
  for p in n.get('foreign',[]):p['allowed_background']=n['node']=='node11'and any('VLLM::'in a for a in p.get('argv',[]))
 return v

def observed(snap):
 c=read(OUT/'cluster.json');names={n['name']:n['gpus']for n in c['nodes']}
 return len(snap['nodes'])==4 and {n['node']for n in snap['nodes']}==set(names)and all(not n.get('error')and not n.get('errors')and sorted(g['gpu']for g in n.get('cards',[]))==names[n['node']]for n in snap['nodes'])

def ready(snap):
 if not observed(snap):return False
 for n in snap['nodes']:
  if n['workers']or any(not f.get('allowed_background')for f in n.get('foreign',[])):return False
  for g in n['cards']:
   if n['node']=='node11':
    if g['used_mib']>=34000:return False
    if g['used_mib']>=2048 and not n.get('foreign'):return False
   elif g['used_mib']>=2048 or g['util']>=10:return False
 return True

def contract(check_data=False):
 p=read(WORK/'protocol.json');c=read(OUT/'config.json');run=read(OUT/'run.json')
 if sha(OUT/'config.json')!=p['config_sha256']or sha(OUT/'cluster.json')!=p['cluster_sha256']or run['experiment']!=c or run['world']!=31:raise ValueError('Frozen training config/cluster/world changed')
 if any(sha(f)!=h for f,h in run['source'].items()):raise ValueError('Frozen training source changed')
 if c['training']['gradient_checkpointing']!='all'or c['training']['save_every']!=500 or c['training']['global_batch']!=31:raise ValueError('Stable execution policy changed')
 if check_data:
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  from p2_ltx_all_lora.config import load
  from p2_ltx_all_lora.data import data_identity
  load(OUT/'config.json')
  if data_identity(c)!=run['data_identity']:raise ValueError('Frozen training data/cache identity changed')
 return dict(success=True,source_files=len(run['source']),config_sha256=p['config_sha256'],cluster_sha256=p['cluster_sha256'],data_checked=check_data)

def health(now=None):
 now=now or time.time();rows=tail();status=read(OUT/'status.json');step=status['step'];flags=[]
 age=now-(OUT/'metrics.jsonl').stat().st_mtime
 saving=step%500==0 and not(OUT/f'step_{step:06d}.complete.json').exists();limit=600 if saving else 180
 if age>limit:flags.append('stalled')
 for row in rows[-30:]:
  if not all(math.isfinite(row[k])for k in ['av','av_video','av_audio','event','feature_video','feature_audio','weighted_total']):flags.append('nonfinite');break
  expected=row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
  if abs(row['weighted_total']-expected)>1e-8:flags.append('wrong_loss');break
  if any(not math.isfinite(row['grad_norm'].get(k,float('nan')))for k in ['connector','ltx_lora','whisper','events','features']):flags.append('nonfinite_gradient');break
 return dict(at=now,step=step,complete=status.get('complete',False),metrics_age_seconds=age,stall_limit=limit,saving=saving,mean_last50_seconds=statistics.mean(r['seconds']for r in rows[-50:])if rows else None,flags=sorted(set(flags)))

def stop_intent():return(OUT/'STOP').exists()

def spawn_record(p,**kw):
 st=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=p.pid,start_ticks=st[19],at=time.time(),**kw)

KILL=__import__('ast').literal_eval(next(n.value for n in __import__('ast').parse((WORK/'recovery_reference_frozen.py').read_text()).body if isinstance(n,__import__('ast').Assign)and any(isinstance(t,__import__('ast').Name)and t.id=='KILL'for t in n.targets)))

def terminate(snap):
 if not observed(snap):raise RuntimeError('Incomplete observations; refuse termination')
 cluster=read(OUT/'cluster.json')
 def one(n):
  obs=next(v for v in snap['nodes']if v['node']==n['name'])
  data=dict(workers=obs['workers'],script=cluster['training_script'],config=str(OUT/'config.json'),mask=','.join(map(str,n['gpus'])))
  code=KILL.replace('PAYLOAD',repr(json.dumps(data)))
  p=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=35,check=True)
  return dict(node=n['name'],results=json.loads(p.stdout))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:return list(pool.map(one,cluster['nodes']))
