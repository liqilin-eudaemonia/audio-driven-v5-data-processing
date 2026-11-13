"""Owned31->24 continuation; no training command includes node11."""
from pathlib import Path
import os,sys,time,json,copy,shutil,subprocess,concurrent.futures,traceback,signal,statistics
from collections import Counter
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_ltx_events.config import source_identity,validate_baseline
from p2_ltx_events.data import data_identity
from p2_ltx_events.training import check_preflight
from p2_ltx_events.schedule import Schedule,CurrentSchedule
from p2_ltx_events.resume24 import continuation_plan
from p2_ltx_events.resume31 import checkpoint_due
from p2_listen.cache import atomic_torch
from listen_cluster import launch,transport
import p2_ltx_resume24 as runtime
OLD=ROOT/'experiments/p2_ltx_clip_train31_cadence_20261107'
OUT=Path(__file__).resolve().parent
PARENT=read(OUT/'PARENT_IDENTITY.json')


def state(stage,**fields):
 value=dict(stage=stage,at=time.time(),controller_pid=os.getpid(),**fields)
 atomic_json(OUT/'switch_state.json',value);print(json.dumps(value),flush=True)


def equal(a,b):
 import torch
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b


def probes(cluster):
 code='''import subprocess,json,pathlib
r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
a=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
workers=[]
for line in a.stdout.splitlines():
 pid=line.split(',')[0].strip()
 if not pid.isdigit():continue
 p=pathlib.Path('/proc')/pid
 try:
  args=p.joinpath('cmdline').read_bytes().split(b'\\0')
  if CONFIG.encode() in args:workers.append(dict(pid=int(pid),args=[x.decode() for x in args if x]))
 except OSError:pass
print(json.dumps(dict(free_mib=[float(s.split(',')[1]) for s in r.stdout.splitlines()],workers=workers,gpu_processes=a.stdout.splitlines())))
'''.replace('CONFIG',repr(str(OLD/'config.json')))
 def one(node):
  for attempt in range(4):
   try:
    r=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=40)
    if r.returncode==0:return dict(node=node['name'],**json.loads(r.stdout))
    error=r.stderr[-500:]
   except Exception as e:error=repr(e)
   time.sleep(1)
  return dict(node=node['name'],error=error)
 with concurrent.futures.ThreadPoolExecutor(4)as pool:return list(pool.map(one,cluster['nodes']))


def configuration(step):
 parent=read(OLD/'config.json')
 if parent!=PARENT['experiment']:raise ValueError('Parent configuration changed')
 changes={'src/p2_ltx_events/config.py','src/p2_ltx_events/schedule.py','src/p2_ltx_events/resume31.py'}
 for path,expected in PARENT['source'].items():
  relative=str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)+'/') else None
  actual=OUT/'source_before'/relative if relative in changes else Path(path)
  if sha(actual)!=expected:raise ValueError('Parent source changed: '+path)
 original=Schedule(parent)
 original.durations=CurrentSchedule(read(parent['resume_execution']['parent_config'])).durations
 plan=continuation_plan(original,step);manifest=OUT/'resume.schedule.json';atomic_json(manifest,plan,immutable=True)
 c=copy.deepcopy(parent);c['output']=str(OUT)
 c['training'].update(global_batch=24,steps=step+len(plan['pairs'])//24)
 c['world24_execution']=dict(kind='resume24-preserve-remaining-v1',world=24,reserved_node='node11',reserved_gpus=list(range(8)))
 c['resume_execution']=dict(kind='preserve-parent-suffix-v1',parent_config=str(OLD/'config.json'),parent_config_sha256=sha(OLD/'config.json'),
  resume_step=step,manifest=str(manifest),manifest_sha256=sha(manifest),world=24,parent_world=31)
 validate_baseline(c,read(c['hidden_event_parent_config']));did=data_identity(c)
 if did!=PARENT['data_identity']:raise ValueError('Data identity changed')
 source=source_identity();pre=copy.deepcopy(read(OLD/'preflight.json'));pre.update(configuration_sha256=digest(c),source=source)
 check_preflight(c,pre,source,did);schedule=Schedule(c)
 actual=[]
 for s in range(step+1,c['training']['steps']+1):
  for rank in range(24):actual.extend(schedule.indices(schedule.size,s,24,schedule.seed,rank,24))
 remaining=original.pairs[(step-original.base_step)*31:];ordinals={o for _,o in remaining}
 retained=[p for p in actual if p[1]in ordinals]
 assert Counter(retained)==Counter(remaining) and len({o for _,o in actual})==len(actual)
 def average(pairs,batch):
  return statistics.mean(max(original.durations[i] for i,_ in pairs[j:j+batch]) for j in range(0,len(pairs),batch))
 durations=dict(parent_mean_batch_max_seconds=average(remaining,31),new_mean_batch_max_seconds=average(actual,24))
 cluster=copy.deepcopy(read(OLD/'cluster.json'));cluster['nodes']=cluster['nodes'][:3];cluster['master_port']=29844
 cluster.update(training_script=str(ROOT/'scripts/p2_ltx_resume24.py'),training_config=str(OUT/'config.json'));runtime.layout(cluster)
 identity=copy.deepcopy(PARENT);identity.update(experiment=c,source=source,world=24,data_identity=did)
 atomic_json(OUT/'CPU_VALIDATION.json',dict(success=True,at=time.time(),resume_step=step,world=24,global_batch=24,
  source_files=len(source),data_identity_unchanged=True,unconsumed_sample_noise_pairs_exact=True,
  original_remaining=plan['original_remaining'],padding=plan['padding'],new_budget=c['training']['steps'],
  duration_grouping=durations,next_save=next(s for s in range(step+1,c['training']['steps']+1) if checkpoint_due(c,s))))
 return c,pre,cluster,identity,plan


def tensorboard(cluster):
 old=read(OLD/'TENSORBOARD_SERVER.json');p=Path('/proc',str(old['pid']))
 if p.exists():
  stat=p.joinpath('stat').read_text().split();args=p.joinpath('cmdline').read_bytes().split(b'\0')
  if stat[2]!='Z' and stat[21]==old['start_ticks'] and b'tensorboard.main' in args and str(OLD/'tensorboard').encode()in args:
   os.kill(old['pid'],signal.SIGTERM)
   until=time.time()+10
   while p.exists() and time.time()<until:
    if p.joinpath('stat').read_text().split()[2]=='Z':break
    time.sleep(.2)
 command=[cluster['python'],'-m','tensorboard.main','--logdir',str(OUT/'tensorboard'),'--host','0.0.0.0','--port','6006',
  '--reload_interval','10','--load_fast','false','--samples_per_plugin','scalars=14000']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (OUT/'tensorboard_server.log').open('ab',buffering=0)as f:
  p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 atomic_json(OUT/'TENSORBOARD_SERVER.json',dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],at=time.time(),
  command=command,logdir=str(OUT/'tensorboard'),port=6006,private_url='http://172.29.252.37:6006',gpu_devices=''))


def monitor(cluster):
 text=(OLD/'monitor.py').read_text().replace('31-GPU','24-GPU').replace('31/max(mean','24/max(mean').replace('31*(step-baseline[1])','24*(step-baseline[1])')
 compile(text,str(OUT/'monitor.py'),'exec');(OUT/'monitor.py').write_text(text)
 with (OUT/'monitor.log').open('ab',buffering=0)as f:
  p=subprocess.Popen([cluster['python'],'-u',str(OUT/'monitor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 atomic_json(OUT/'MONITOR.json',dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],at=time.time()))


def main():
 import torch
 torch.set_num_threads(4)
 cluster_old=read(OLD/'cluster.json')
 if not (OLD/'STOP').exists():raise ValueError('Explicit parent graceful STOP required')
 if (OUT/'STOP').exists():raise ValueError('Preserve explicit child STOP')
 state('waiting_parent_exit')
 until=time.time()+600
 while True:
  step=read(OLD/'status.json')['step'];marker=OLD/f'step_{step:06d}.complete.json'
  observations=probes(cluster_old);atomic_json(OUT/'PARENT_EXIT_PROBE.json',observations)
  if marker.exists() and all('error' not in n and not n['workers'] for n in observations):break
  if time.time()>until:raise TimeoutError('Parent checkpoint/exit not verified; no processes killed')
  time.sleep(3)
 atomic_json(OUT/'NODE11_RELEASED.json',dict(node='node11',training_workers=[],observed=next(n for n in observations if n['node']=='node11'),at=time.time()))
 (OLD/'MONITOR_STOP').touch()
 state('building_validated24_configuration',step=step)
 c,pre,cluster,identity,plan=configuration(step)
 state('migrating_model_adam_rng',step=step)
 from p2_ltx_events.checkpoint import checked
 origin=OLD/f'step_{step:06d}.pt';saved=checked(origin)
 assert saved['identity']==PARENT and saved['step']==step and len(saved['rank_rng'])==31
 target=OUT/origin.name;value=dict(saved,identity=identity,rank_rng=saved['rank_rng'][:24])
 atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(step=step,sha256=sha(target)))
 loaded=checked(target)
 assert equal(saved['model'],loaded['model']) and equal(saved['optimizer'],loaded['optimizer'])
 assert equal(saved['rank_rng'][:24],loaded['rank_rng'])
 optimizer_tensors=[v for s in loaded['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v)]
 assert all(torch.isfinite(v).all() for v in optimizer_tensors)
 optimizer_steps={float(s['step']) for s in loaded['optimizer']['state'].values() if 'step' in s}
 assert optimizer_steps=={float(step)}
 atomic_json(OUT/'MIGRATION.json',dict(step=step,origin=str(origin),origin_sha256=read(marker)['sha256'],world=24,global_batch=24,
  model_optimizer_bitwise_equal=True,all24_rng_bitwise_equal=True,parent_all31_rng_preserved=True,
  optimizer_all_finite=True,optimizer_tensors=len(optimizer_tensors),optimizer_steps=sorted(optimizer_steps),
  data_loss_unchanged=True,unconsumed_sample_noise_pairs_exact=True,original_remaining=plan['original_remaining'],
  padding=plan['padding'],old_budget=9041,new_budget=c['training']['steps'],at=time.time()))
 del saved,value,loaded,optimizer_tensors
 for name,content in [('config.json',c),('preflight.json',pre),('cluster.json',cluster),('run.json',identity)]:atomic_json(OUT/name,content,immutable=True)
 shutil.copy2(OLD/'metrics.jsonl',OUT/'metrics.jsonl');shutil.copytree(OLD/'tensorboard',OUT/'tensorboard',dirs_exist_ok=True)
 atomic_json(OUT/'status.json',dict(step=step,budget=c['training']['steps'],complete=False))
 observations=probes(cluster);atomic_json(OUT/'RESOURCES_BEFORE_LAUNCH.json',observations)
 for n in observations:
  if 'error'in n or n['workers'] or len(n['free_mib'])!=8 or min(n['free_mib'])<108*1024:
   raise ValueError('Selected training GPU still occupied or query failed: '+str(n))
 atomic_json(OUT/'RESERVED_NODE.json',dict(node='node11',host='172.29.189.119',reserved_gpus=list(range(8)),training_visible_gpus=[]))
 atomic_json(OLD/'SUPERSEDED_BY.json',dict(output=str(OUT),step=step,world=24,at=time.time()))
 atomic_json(OUT/'QUEUE_STATE.json',dict(state='training',phase='startup',controller_pid=os.getpid(),resume=str(target),at=time.time()))
 cmds=runtime.commands(cluster,OUT/'config.json',target,OUT/'rank_logs');atomic_json(OUT/'COMMANDS.json',cmds)
 tensorboard(cluster);monitor(cluster);state('starting24',step=step,budget=c['training']['steps'],resume=str(target))
 launch(cluster,cmds,OUT/'launch')
 atomic_json(OUT/'QUEUE_STATE.json',dict(state='finished',controller_pid=os.getpid(),at=time.time()))
 state('finished')


if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(OUT/'SWITCH_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
  state('failed',error=traceback.format_exc());raise
