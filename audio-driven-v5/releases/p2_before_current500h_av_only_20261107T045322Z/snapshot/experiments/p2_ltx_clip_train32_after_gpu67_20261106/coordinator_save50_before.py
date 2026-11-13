"""Scheduled, guarded 2->32 GPU handoff; never kills unrelated GPU jobs."""
from pathlib import Path
import sys,os,json,time,copy,datetime,subprocess,concurrent.futures,fcntl,runpy,traceback
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from listen_cluster import transport,launch
OUT=Path(__file__).resolve().parent;OLD=ROOT/'experiments/p2_ltx_clip_node11_gpu67_20261106'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(name,**extra):
 r=dict(state=name,time=now(),controller_pid=os.getpid(),**extra);atomic_json(OUT/'QUEUE_STATE.json',r);print(json.dumps(r),flush=True)

PROBE="""import subprocess,json
from pathlib import Path
def query(kind,fields):
 r=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr)
 return [list(map(str.strip,line.split(','))) for line in r.stdout.splitlines() if line.strip()]
cards={}
for index,uuid,mem,util in query('gpu','index,uuid,memory.used,utilization.gpu'):
 cards[index]={'uuid':uuid,'memory_used_mib':int(mem),'utilization':int(util),'processes':[]}
for uuid,pid,name in query('compute-apps','gpu_uuid,pid,process_name'):
 try:args=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\\0');args=[x.decode(errors='replace') for x in args if x]
 except OSError:args=[]
 own=('--config' in args and args[args.index('--config')+1]=='/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_clip_node11_gpu67_20261106/config.json' and any(x.endswith('/scripts/p2_ltx_events.py') for x in args))
 for card in cards.values():
  if card['uuid']==uuid:card['processes'].append({'pid':int(pid),'own_current_run':own,'name':name})
print(json.dumps({'cards':cards}))
"""

def blockers(observations,allow_owned):
 result=[]
 for obs in observations:
  node=obs['node']
  if 'error' in obs:result.append(dict(node=node,error=obs['error']));continue
  cards=obs['cards']
  if set(cards)!=set(map(str,range(8))):result.append(dict(node=node,error='Expected eight GPUs'));continue
  for index,card in cards.items():
   procs=card['processes'];owned=allow_owned and node=='node11' and index in ('6','7') and procs and all(p['own_current_run'] for p in procs)
   if owned:continue
   if procs or card['memory_used_mib']>=256 or card['utilization']!=0:
    result.append(dict(node=node,gpu=int(index),memory_used_mib=card['memory_used_mib'],utilization=card['utilization'],processes=procs))
 return result

def probe_all(cluster):
 def probe(node):
  try:
   r=subprocess.run(transport(cluster,node,[cluster['python'],'-c',PROBE]),capture_output=True,text=True,timeout=45)
   if r.returncode:return dict(node=node['name'],error=r.stderr[-1000:])
   return dict(node=node['name'],**json.loads(r.stdout))
  except Exception as exc:return dict(node=node['name'],error=str(exc))
 with concurrent.futures.ThreadPoolExecutor(4) as pool:return list(pool.map(probe,cluster['nodes']))

def cancelled():return (OUT/'CANCEL').exists()
def main():
 with (OUT/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  import torch
  from p2_ltx_events.config import source_identity,validate_baseline
  from p2_ltx_events.training import preflight,check_preflight
  from p2_ltx_events.data import data_identity
  from p2_ltx_events.checkpoint import checked
  from p2_ltx_events.schedule import Schedule
  from p2_listen.cache import atomic_torch
  from p2_whisper_event_feature.execution import settings
  torch.set_num_threads(2)
  c=read(OUT/'config.json');cluster=read(OUT/'cluster.json');schedule=read(OUT/'SCHEDULE.json')
  validate_baseline(c,read(c['hidden_event_parent_config']))
  assert c['training']['global_batch']==32 and c['training']['save_every']==50
  assert [n['name'] for n in cluster['nodes']]==['node12','node8','node4','node11']
  assert all(n['gpus']==list(range(8)) for n in cluster['nodes'])
  state('preflight_while_dual_gpu_continues',not_before_utc=schedule['not_before_utc'])
  pre=preflight(c,32)
  while not cancelled():
   observations=probe_all(cluster);busy=blockers(observations,True)
   atomic_json(OUT/'RESOURCE_PROBE.json',dict(time=now(),observations=observations,blockers=busy))
   remaining=max(0,schedule['not_before_timestamp']-time.time())
   if remaining==0 and not busy:break
   state('waiting_scheduled_time' if remaining else 'waiting_other_jobs',not_before_utc=schedule['not_before_utc'],seconds_until_not_before=round(remaining),blockers=busy,current_dual_step=read(OLD/'status.json')['step'])
   time.sleep(30)
  if cancelled():state('cancelled_before_handoff');return
  # Fail before stopping the source run if immutable code/data changed meanwhile.
  check_preflight(c,pre,source_identity(),data_identity(c))
  state('requesting_dual_gpu_checkpoint',source_step=read(OLD/'status.json')['step'])
  (OLD/'STOP').touch()
  while not (OLD/'launch/completed.json').exists():
   if cancelled():state('cancelled_during_graceful_stop');return
   time.sleep(5)
  assert read(OLD/'launch/completed.json')['returncodes']==[0]
  step=read(OLD/'status.json')['step'];origin=OLD/f'step_{step:06d}.pt'
  state('migrating_latest_checkpoint',saved_step=step)
  saved=checked(origin);before=saved['identity'];old_c=read(OLD/'config.json')
  assert before['world']==2 and len(saved['rank_rng'])==2 and before['experiment']==old_c
  assert {k:v for k,v in c.items() if k!='output'}=={k:v for k,v in old_c.items() if k!='output'}
  assert before['source']==source_identity()==pre['source']
  assert before['data_identity']==pre['data_identity'] and before['statistics']==pre['statistics'] and before['execution']==settings(c)
  plan=Schedule(c)
  for s in range(step+1,min(step+9,c['training']['steps']+1)):
   old=[plan.indices(plan.size,s,32,c['training']['seed'],rank,2) for rank in range(2)]
   old_global=[old[r][m] for m in range(16) for r in range(2)]
   new=[plan.indices(plan.size,s,32,c['training']['seed'],rank,32)[0] for rank in range(32)]
   assert old_global==new
  identity=dict(before,world=32,experiment=c,source=pre['source'],data_identity=pre['data_identity'],statistics=pre['statistics'])
  migrated=dict(saved,identity=identity,rank_rng=[copy.deepcopy(saved['rank_rng'][r%2]) for r in range(32)])
  lineage=dict(origin=str(origin),origin_sha256=read(origin.with_suffix('.complete.json'))['sha256'],step=step,old_world=2,new_world=32,
   model_preserved=True,optimizer_preserved=True,global_batch=32,save_every=50,source_unchanged=True,
   sample_order_and_ordinals_preserved=True,rng_policy='rank r uses prior rank r%2 state; each sample reseeded from global ordinal')
  migrated['world_migration']=lineage;checkpoint=OUT/f'step_{step:06d}.pt';atomic_torch(checkpoint,migrated)
  atomic_json(checkpoint.with_suffix('.complete.json'),dict(sha256=sha(checkpoint),step=step))
  readback=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
  assert readback['identity']==identity and len(readback['rank_rng'])==32
  assert readback['optimizer']['param_groups']==saved['optimizer']['param_groups']
  for prefix in ['connector.','event_heads.','decoders.','text_lora.','whisper_adapter.','whisper_branches.']:
   key=next(k for k in saved['model'] if k.startswith(prefix));assert torch.equal(saved['model'][key],readback['model'][key])
  atomic_json(OUT/'MIGRATION.json',lineage);del saved,migrated,readback
  stable=0
  while not cancelled():
   observations=probe_all(cluster);busy=blockers(observations,False);stable=stable+1 if not busy else 0
   state('waiting_all32_idle',stable_checks=stable,blockers=busy,saved_step=step)
   if stable>=2:break
   time.sleep(15)
  if cancelled():state('cancelled_with_checkpoint_saved',saved_step=step);return
  check_preflight(c,pre,source_identity(),data_identity(c))
  cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='scheduled_train32')
  commands=cli['commands'](cluster,OUT/'config.json',resume=checkpoint)
  assert len(commands)==4
  atomic_json(OUT/'COMMANDS.json',commands)
  state('launching_four_nodes',saved_step=step,next_step=step+1,world_size=32,global_batch=32,save_every=50)
  launch(cluster,commands,OUT/'launch')
  state('finished',training_status=read(OUT/'status.json'))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  state('failed',error=str(exc));print(traceback.format_exc(),flush=True);raise
