"""Preserve both optimizer histories while exchanging their1/31-GPU allocation."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib,importlib.util,json,math,os,shutil,subprocess,sys,time,traceback
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]
from gate import read,write,record,request_stop
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
SELF=Path(__file__).resolve();WORKER=WORK/'worker.py'
PACKAGES={'full':'p2_ltx_all_lora','noevent':'p2_ltx_all_lora_noevent'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def activate():
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
def module(arm,name):return importlib.import_module(PACKAGES[arm]+'.'+name)
def protocol():return read(WORK/'PROTOCOL.json')
def path(arm,target=False):return Path(protocol()[arm+('_target'if target else'_parent')])
def emit(stage,**kw):
 v=dict(at=time.time(),stage=stage,**kw);write(WORK/'STATE.json',v)
 with(WORK/'events.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)
def cancelled():
 if(WORK/'STOP').exists():raise RuntimeError('Resource swap cancelled by user')
def alive(r):
 try:
  s=Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return s[0]!='Z'and s[19]==r['start_ticks']
 except OSError:return False
def observation_module():
 old=ROOT/'experiments/p2_full_lora_guard9h_20261108'
 sys.path.insert(0,str(old));import common
 return common
def transport(cluster,node,cmd):
 from listen_cluster import transport as native
 return native(cluster,node,cmd)
def remote(cluster,node,code,timeout=35):
 p=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code]),text=True,capture_output=True,timeout=timeout,check=True)
 return json.loads(p.stdout)

def single_probe(out):
 cl=read(out/'cluster.json');node=cl['nodes'][0]
 payload=dict(out=str(out),script=cl['training_script'],config=str(out/'config.json'))
 code='x='+repr(payload)+'\n'+r'''
from pathlib import Path
import json,subprocess
g=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
uuid=g[0].strip();apps=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
 a=[s.strip()for s in line.split(',')]
 if len(a)==2 and a[0].isdigit()and a[1]==uuid:apps.append(int(a[0]))
r=Path(x['out'])/'CONTROLLER.json';workers=[]
if r.exists():
 v=json.loads(r.read_text());p=Path('/proc',str(v['pid']))
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().decode().split(chr(0));env=dict(a.split('=',1)for a in(p/'environ').read_bytes().decode().split(chr(0))if'='in a)
  if st[0]!='Z'and st[19]==v['start_ticks']:
   if x['script']not in argv or x['config']not in argv or env.get('CUDA_VISIBLE_DEVICES')!='0'or env.get('WORLD_SIZE')!='1':raise RuntimeError('Single process identity mismatch')
   workers.append(dict(pid=v['pid'],start_ticks=st[19],gpu_context=v['pid']in apps))
 except FileNotFoundError:pass
print(json.dumps(dict(workers=workers,apps=apps,used_mib=float(g[1]),free_mib=float(g[2]),healthy=bool(workers and workers[0]['gpu_context']))))
'''
 return remote(cl,node,code)

def target_identity(parent,c,pre,world):
 if (parent['world'],world)not in[(1,31),(31,1)]:raise ValueError('Require an explicit1-to31 or31-to1 transition')
 before=copy.deepcopy(c);before['output']=parent['experiment']['output']
 if before!=parent['experiment']or c['output']==before['output']or c['training']['global_batch']!=31:raise ValueError('Model, loss, data, schedule or effective batch changed')
 if pre['source']!=parent['source']or pre['data_identity']!=parent['data_identity']or pre['statistics']!=parent['statistics']:raise ValueError('Frozen source/data/statistics drift')
 result=copy.deepcopy(parent);result.update(experiment=copy.deepcopy(c),world=world)
 return result

def equal_tree(a,b):
 import torch
 if torch.is_tensor(a):return torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(equal_tree(a[k],b[k])for k in a)
 if type(a)!=type(b):return False
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal_tree(x,y)for x,y in zip(a,b))
 return a==b

def rewrite(arm,source,destination,identity,parent_identity=None):
 import torch
 from p2_listen.cache import atomic_torch
 torch.set_num_threads(2);cp=module(arm,'checkpoint');v=cp.checked(source);old=v['identity']
 if source.resolve()==destination.resolve()or source.parent.resolve()!=Path(old['experiment']['output']).resolve()or destination.parent.resolve()!=Path(identity['experiment']['output']).resolve()or source.name!=destination.name:raise ValueError('Preserve original checkpoint and exact step in independent output')
 if parent_identity is not None and old!=parent_identity:raise ValueError('Wrong parent checkpoint/run identity')
 if len(v['rank_rng'])!=old['world']:raise ValueError('Parent RNG/world count differs')
 normalized=copy.deepcopy(identity);normalized['world']=old['world'];normalized['experiment']['output']=old['experiment']['output']
 if normalized!=old or identity['world']not in(1,31)or old['world']==identity['world']:raise ValueError('Unapproved identity change')
 for state in v['optimizer']['state'].values():
  for t in state.values():
   if torch.is_tensor(t)and not torch.isfinite(t).all():raise ValueError('Nonfinite parent Adam state')
 states=[copy.deepcopy(v['rank_rng'][0])for _ in range(identity['world'])]
 payload=dict(v,identity=identity,rank_rng=states);atomic_torch(destination,payload)
 write(destination.with_suffix('.complete.json'),dict(step=v['step'],sha256=sha(destination)))
 actual=cp.checked(destination)
 if not equal_tree(v['model'],actual['model'])or not equal_tree(v['optimizer'],actual['optimizer']):raise ValueError('Model or Adam tensor changed during migration')
 if actual['step']!=v['step']or actual['identity']!=identity or len(actual['rank_rng'])!=identity['world']:raise ValueError('Migration metadata differs')
 return dict(success=True,at=time.time(),source=str(source),source_sha256=sha(source),destination=str(destination),destination_sha256=sha(destination),step=v['step'],parent_world=old['world'],world=identity['world'],model_and_adam_equal=True,rng_policy='Parent rank0 startup RNG copied to target ranks; unchanged native per-sample seed/noise ordinal reseeding determines subsequent examples. Original checkpoint retains all original RNG streams.',original_checkpoint_preserved=True)

def prepare():
 activate();p=protocol();records={};plans=[]
 for arm,world in [('full',1),('noevent',31)]:
  parent=path(arm);out=path(arm,True);out.mkdir(exist_ok=True)
  c=copy.deepcopy(read(parent/'config.json'));c['output']=str(out);module(arm,'config').validate(c)
  write(out/'config.json',c)
  cl=copy.deepcopy(read(path('full')/'cluster.json'));cl.update(training_script=str(WORKER),training_config=str(out/'config.json'),master_port=29891)
  if world==1:
   node=copy.deepcopy(next(n for n in cl['nodes']if n['name']=='node11'));node['gpus']=[0];cl['nodes']=[node]
  write(out/'cluster.json',cl)
  pre=module(arm,'training').preflight(c);r=read(parent/'run.json');target_identity(r,c,pre,world)
  for f,h in r['source'].items():
   if sha(f)!=h:raise ValueError('Frozen parent source changed: '+f)
  source=Path(module(arm,'training').__file__).read_text()
  if "torch.manual_seed(t['seed']+ordinal)"not in source or "manual_seed(t['seed']+1000000+ordinal)"not in source:raise ValueError('World-independent per-example RNG contract absent')
  schedule=module(arm,'schedule').Schedule(c);plans.append(schedule.pairs)
  for step in range(1,c['training']['steps']+1):
   one=schedule.indices(schedule.size,step,31,c['training']['seed'],0,1)
   many=[q for rank in range(31)for q in schedule.indices(schedule.size,step,31,c['training']['seed'],rank,31)]
   if one!=many:raise ValueError('Sample/noise schedule changes across worlds')
  records[arm]=dict(output=str(out),world=world,config_sha256=sha(out/'config.json'),cluster_sha256=sha(out/'cluster.json'),parent_source_files=len(r['source']),canary_reads=len(pre['canary_reads']),all9169_steps_schedule_equal=True)
 if plans[0]!=plans[1]:raise ValueError('Two experiment sample/noise plans differ')
 ready=dict(success=True,at=time.time(),targets=records,source={p.name:sha(p)for p in [SELF,WORKER,WORK/'gate.py']},effective_batch=31,same_final500h_and_latest_caption=True)
 write(WORK/'CODE_READY.json',ready);return ready

def migrate(arm):
 activate();parent=path(arm);out=path(arm,True);step=read(parent/'status.json')['step'];source=parent/f'step_{step:06d}.pt'
 if arm=='full'and step!=6000:raise ValueError('Full run must stop exactly at6000')
 if not source.with_suffix('.complete.json').exists():raise ValueError('Latest completed optimizer update has not been saved')
 c=module(arm,'config').load(out/'config.json');pre=read(out/'preflight.json');parent_run=read(parent/'run.json')
 identity=target_identity(parent_run,c,pre,1 if arm=='full'else 31)
 destination=out/source.name;proof=rewrite(arm,source,destination,identity,parent_run)
 write(out/'run.json',identity);write(out/'status.json',dict(step=step,budget=c['training']['steps'],complete=False))
 with(out/'metrics.jsonl').open('w')as f:
  for line in(parent/'metrics.jsonl').read_text().splitlines():
   if json.loads(line)['step']<=step:f.write(line+'\n')
 write(out/'MIGRATION.json',proof);write(WORK/(arm+'_MIGRATED.json'),proof)
 return proof

def source_ready():
 r=read(WORK/'CODE_READY.json')
 if not r['success']or any(sha(WORK/k)!=v for k,v in r['source'].items()):raise ValueError('Reviewed swap source changed')
 for arm,v in r['targets'].items():
  if sha(path(arm,True)/'config.json')!=v['config_sha256']or sha(path(arm,True)/'cluster.json')!=v['cluster_sha256']:raise ValueError('Target placement/config changed')

def launch_node(index):
 out=path('noevent',True);cl=read(out/'cluster.json');node=cl['nodes'][index];offset=sum(len(n['gpus'])for n in cl['nodes'][:index]);logs=out/'rank_logs';logs.mkdir(exist_ok=True)
 procs=[];streams=[]
 try:
  for local,gpu in enumerate(node['gpus']):
   rank=offset+local;env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,node['gpus'])),RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='31',LOCAL_WORLD_SIZE=str(len(node['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']),P2_TRAIN_MEMORY_GIB='108'if node['name']=='node11'else'134')
   cmd=[PY,'-u',str(WORKER),'--arm','noevent','--config',str(out/'config.json'),'--resume',read(out/'MIGRATION.json')['destination']]
   stream=(logs/f'rank{rank:02d}.log').open('a');streams.append(stream);procs.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT))
  write(logs/(node['name']+'_workers.json'),dict(node=node['name'],pids=[p.pid for p in procs],ranks=list(range(offset,offset+len(procs))),gpus=node['gpus']))
  while any(p.poll()is None for p in procs):
   if any(p.poll()not in(None,0)for p in procs):raise RuntimeError('Own local training worker failed')
   time.sleep(2)
 finally:
  for p in procs:
   if p.poll()is None:p.terminate()
  for p in procs:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for s in streams:s.close()

def launch31():
 from listen_cluster import launch
 source_ready();out=path('noevent',True);cl=read(out/'cluster.json');cmds=[]
 for i,node in enumerate(cl['nodes']):
  cmds.append(['env',*[k+'='+str(v)for k,v in cl['network'].items()],PY,'-u',str(SELF),'node','--node-index',str(i)])
 write(out/'COMMANDS.json',cmds);write(out/'CONTROLLER.json',record());launch(cl,cmds,out/'nodes')

def launch1():
 source_ready();out=path('full',True);cl=read(out/'cluster.json');node=cl['nodes'][0]
 env=dict(cl['network'],CUDA_VISIBLE_DEVICES='0',RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',P2_TRAIN_MEMORY_GIB='134')
 cmd=['env',*[k+'='+str(v)for k,v in env.items()],PY,'-u',str(WORKER),'--arm','full','--config',str(out/'config.json'),'--resume',read(out/'MIGRATION.json')['destination']]
 write(out/'COMMAND.json',dict(command=cmd,world=1,global_batch=31,accumulation=31));write(out/'LAUNCH_CONTROLLER.json',record())
 subprocess.run(transport(cl,node,cmd),check=True)

def spawn(command,label):
 with(WORK/(label+'.log')).open('a')as log:p=subprocess.Popen([PY,'-u',str(SELF),command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 s=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split();write(WORK/(label+'_CONTROLLER.json'),dict(pid=p.pid,start_ticks=s[19],at=time.time()))
 return p

def wait_for(predicate,stage,timeout=1800):
 until=time.time()+timeout
 while time.time()<until:
  cancelled()
  if predicate():return
  time.sleep(2)
 raise TimeoutError(stage)

def rows_after(out,step):
 p=out/'metrics.jsonl'
 if not p.exists():return[]
 return[r for r in map(json.loads,p.read_text().splitlines())if r['step']>step]

def check_rows(rows,event):
 groups=['connector','ltx_lora','whisper','features']+(['events']if event else[])
 for r in rows:
  for k in ['av','event','feature_video','feature_audio','weighted_total']:
   if not math.isfinite(r[k]):raise ValueError('Nonfinite training loss')
  want=r['av']+(.05 if event else 0)*r['event']+.025*(r['feature_video']+r['feature_audio'])
  if abs(want-r['weighted_total'])>1e-8 or not event and(r['event']!=0 or r['per_event']):raise ValueError('Wrong loss/branch')
  if any(not math.isfinite(r['grad_norm'][k])or r['grad_norm'][k]<=0 for k in groups):raise ValueError('Missing/nonfinite gradient')

def run():
 activate();obs=observation_module();p=protocol()
 with(WORK/'controller.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(WORK/'CONTROLLER.json',record());source_ready()
  emit('waiting_full5900',target=6000)
  wait_for(lambda:read(path('full')/'status.json')['step']>=p['prepare_noevent_at_full_step'],'waiting5900',86400)
  emit('saving_noevent_boundary');old=single_probe(path('noevent'));write(WORK/'NOEVENT_BEFORE_STOP.json',old)
  if old['workers']:request_stop(path('noevent')/'STOP',p['owner'],reason='Prepare state-preserving1-to31 handoff before full6000')
  wait_for(lambda:not single_probe(path('noevent'))['workers'],'single native save/exit',1800)
  emit('migrating_noevent1to31');no=migrate('noevent');write(WORK/'NOEVENT_READY.json',no)
  emit('waiting_full6000')
  wait_for(lambda:(path('full')/'step_006000.complete.json').exists(),'full6000 save',86400)
  wait_for(lambda:_old_exited(obs),'full31 native exit',900)
  emit('checking_selected_resources')
  wait_for(lambda:obs.ready(obs.probe()),'selected31 ready',1800)
  source_ready();emit('launching_noevent31_while_preparing_full_single',noevent_step=no['step'])
  a=spawn('launch31','noevent31')
  emit('migrating_full31to1');full=migrate('full')
  wait_for(lambda:(_gpu0_idle()),'node11 GPU0 ready',1800)
  source_ready();emit('launching_successors',noevent_step=no['step'],full_step=full['step'])
  b=spawn('launch1','full_gpu0')
  started=time.time()
  while time.time()-started<1800:
   cancelled()
   if a.poll()is not None or b.poll()is not None:raise RuntimeError('A successor launcher terminated during startup')
   nr=rows_after(path('noevent',True),no['step']);fr=rows_after(path('full',True),full['step'])
   check_rows(nr,False);check_rows(fr,True)
   if len(nr)>=3 and len(fr)>=2:
    ns=obs.OLD.probe(path('noevent',True));ns['reserved_gpu']='node11:0';fs=single_probe(path('full',True))
    if ns['healthy31']and fs['healthy']:
     result=dict(success=True,at=time.time(),noevent_step=nr[-1]['step'],full_step=fr[-1]['step'],noevent_cluster=ns,full_single=fs,noevent_migration=no,full_migration=full,effective_batch=31)
     write(WORK/'SWAP_ACCEPTED.json',result);emit('both_running_accepted',noevent_step=nr[-1]['step'],full_step=fr[-1]['step']);break
   time.sleep(10)
  else:raise TimeoutError('Successors have not both completed verified real updates')
  while a.poll()is None or b.poll()is None:
   for arm,child in [('noevent',a),('full',b)]:
    if child.poll()not in(None,0):write(WORK/(arm+'_TRAINING_ERROR.json'),dict(at=time.time(),returncode=child.returncode));raise RuntimeError('Owned successor failed; preserve other training')
   time.sleep(30)
  emit('both_launchers_exited')

def _old_exited(obs):
 s=obs.probe();write(WORK/'FULL_EXIT_PROBE.json',s)
 return obs.observed(s)and all(not n['workers']for n in s['nodes'])
def _gpu0_idle():
 s=single_probe(path('noevent'));write(WORK/'GPU0_READY_PROBE.json',s)
 return not s['workers']and not s['apps']and s['used_mib']<2048

if __name__=='__main__':
 try:
  parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','launch31','launch1','node']);parser.add_argument('--node-index',type=int);args=parser.parse_args()
  if args.command=='prepare':print(json.dumps(prepare()),flush=True)
  elif args.command=='run':run()
  elif args.command=='node':source_ready();launch_node(args.node_index)
  elif args.command=='launch31':launch31()
  else:launch1()
 except Exception:
  write(WORK/('ERROR_'+str(os.getpid())+'.json'),dict(at=time.time(),traceback=traceback.format_exc()));raise
