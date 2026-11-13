"""Explicitly activated29-GPU continuation; preparation never launches GPU work."""
import argparse,copy,fcntl,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
SELF=Path(__file__).resolve();WORK=SELF.parent;ROOT=WORK.parents[1];STATE=WORK/'cluster29'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
spec=importlib.util.spec_from_file_location('single_gpu_frozen',WORK/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,sha=base.read,base.sha
STEPS=base.STEPS;PYTHON=base.PYTHON;METRIC_PYTHON=base.METRIC_PYTHON
GUARD=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def cfg():return read(STATE/'cluster.json')
def slots(c):
 expected={'node12':list(range(8)),'node8':list(range(8)),'node4':list(range(8)),'node11':list(range(5))}
 assert len(c['nodes'])==4 and {n['name']:n['gpus'] for n in c['nodes']}==expected
 out=[dict(rank=i,node=n['name'],host=n['host'],gpu=g) for i,(n,g) in enumerate((n,g) for n in c['nodes'] for g in n['gpus'])]
 assert len(out)==29 and len({(x['node'],x['gpu']) for x in out})==29
 return out

def verify():
 p=read(STATE/'PREPARED.json');assert p['entry_sha256']==sha(SELF) and p['base_entry_sha256']==sha(WORK/'run.py')
 assert p['shared_items_sha256']==sha(WORK/'shared_items.json') and p['conditions_sha256']==sha(WORK/'CONDITIONS.json')
 assert read(WORK/'INTEGRITY_COMPLETE.json')['success'];slots(cfg())
 assert p['cluster_sha256']==sha(STATE/'cluster.json')
 assert all(sha(path)==h for path,h in p['native_code'].items())
 for step,h in p['protocol_sha256'].items():assert sha(base.child(int(step))/'protocol.json')==h
 return p

def prepare():
 base.prepare();assert read(WORK/'INTEGRITY_COMPLETE.json')['success']
 c=copy.deepcopy(read(base.TRAIN/'cluster.json'))
 for n in c['nodes']:
  if n['name']=='node11':n['gpus']=list(range(5))
 c['training_script']=str(SELF);c['purpose']='15 LTX checkpoints x64 test tuning; same frozen science'
 mapping=slots(c)
 if (STATE/'PREPARED.json').exists():return verify()
 save(STATE/'cluster.json',c)
 expected_sources=read(WORK/'PLAN.json')['identity']['source']
 assert all(sha(p)==h for p,h in expected_sources.items())
 for rec in read(WORK/'CHECKPOINTS.json'):
  v=read(WORK/'integrity'/f'{rec["step"]:06d}.json');assert v['sha256']==rec['sha256'] and v['stamp']==base.stamp(rec['path'])
 report=dict(success=True,at=time.time(),cluster_sha256=sha(STATE/'cluster.json'),entry_sha256=sha(SELF),base_entry_sha256=sha(WORK/'run.py'),shared_items_sha256=sha(WORK/'shared_items.json'),conditions_sha256=sha(WORK/'CONDITIONS.json'),protocol_sha256={str(s):sha(base.child(s)/'protocol.json') for s in STEPS},slots=mapping,total_gpus=29,source_verified=True,checkpoints_sha_verified=15,outputs=960,source_protocols_unchanged=True,automatic_start=False,activate_requires_released_flag=True,reserved_node11=[5,6,7],handoff='Preflight all assigned cards, then freeze old controller, let children finish current sample/score, terminate only old controller; reuse exact atomic outputs and sample locks',scheduling='Up to2 renderers/checkpoint across29 cards; one GPU task/card; release GPUs into ready decode and score blocks; no duplicated checkpoint exports',native_code={str(p):sha(p) for p in [ROOT/'src/p2_all_lora_followups_eval_20261109.py',ROOT/'src/p2_testsets_eval/shared_items.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'scripts/p2_pipelined_testsets_eval.py',GUARD]})
 save(STATE/'PREPARED.json',report);print(json.dumps(dict(prepared=True,gpus=29,launched=False,reserved_node11=[5,6,7])))
 return report

PROBE_CODE=r'''
import os,json,subprocess,time
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');work=root/'experiments/p2_ltx15_test64_sft2752_20261113'
r=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={}
for line in r.splitlines():
 p=[s.strip() for s in line.split(',')];gpus[int(p[0])]=dict(uuid=p[1],used=float(p[2]),free=float(p[3])-float(p[2]),util=float(p[4]))
apps=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 p=[s.strip() for s in line.split(',')]
 if len(p)!=3:continue
 pid=int(p[0]);proc=Path('/proc')/str(pid)
 try:cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();state=proc.joinpath('stat').read_text().split(') ',1)[1][0]
 except FileNotFoundError:continue
 apps.append(dict(pid=pid,uuid=p[1],memory=p[2],cmd=cmd,state=state,owned=str(work) in cmd))
print(json.dumps(dict(at=time.time(),ips=subprocess.check_output(['hostname','-I'],text=True).split(),gpus=gpus,apps=apps,shared_entry=work.joinpath('cluster29.py').exists())))
'''
def probe_node(node):
 from listen_cluster import transport
 c=cfg();r=subprocess.run(transport(c,node,[str(PYTHON),'-c',PROBE_CODE]),capture_output=True,text=True,timeout=25)
 if r.returncode:raise ConnectionError(r.stderr[-400:])
 report=json.loads(r.stdout);assert node['host'] in report['ips'] and report['shared_entry'];return report

def blockers(node,report,allow_own_gpu4=False):
 bad=[]
 for gpu in node['gpus']:
  g=report['gpus'].get(str(gpu),report['gpus'].get(gpu));assert g is not None
  apps=[a for a in report['apps'] if a['uuid']==g['uuid'] and a['state']!='Z']
  own_allowed=allow_own_gpu4 and node['name']=='node11' and gpu==4
  foreign=[a for a in apps if not (own_allowed and a['owned'])]
  if foreign or (not apps and g['used']>2048) or (not own_allowed and g['used']>2048):bad.append(dict(node=node['name'],gpu=gpu,used_mib=g['used'],foreign_pids=[a['pid'] for a in foreign]))
 return bad

def check_resources(allow_own_gpu4=False):
 c=cfg();reports={};errors=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures={n['name']:(n,pool.submit(probe_node,n)) for n in c['nodes']}
  for name,(node,f) in futures.items():
   try:reports[name]=f.result();errors+=blockers(node,reports[name],allow_own_gpu4)
   except Exception as exc:errors.append(dict(node=name,error=repr(exc)))
 r=dict(at=time.time(),available=not errors,blockers=errors,nodes=reports,allow_current_gpu4_task=allow_own_gpu4)
 save(STATE/'RESOURCE_CHECK.json',r);return r

def handoff_stops(create=False):
 paths=[WORK/'STOP',*[base.child(step)/'STOP' for step in STEPS]]
 if create:
  # Check every marker before mutating any, and before freezing the controller.
  for path in paths:
   if path.exists():assert read(path).get('owner')=='cluster29_handoff','Unrelated STOP: '+str(path)
  for path in paths:save(path,dict(owner='cluster29_handoff',at=time.time(),reason='Drain current samples before29-GPU migration'))
 else:
  for path in paths:
   if path.exists() and read(path).get('owner')=='cluster29_handoff':path.unlink()

def handoff():
 """Only called explicitly on node11 after all other cards passed admission."""
 assert cfg()['nodes'][-1]['name']=='node11'
 assert cfg()['nodes'][-1]['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 if (STATE/'HANDOFF.json').exists():return
 launch=read(WORK/'CONTROLLER.json');pid=launch['pid'];proc=Path('/proc')/str(pid)
 cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
 live=bool(cmd) and proc.joinpath('stat').read_text().split(') ',1)[1][0]!='Z'
 # Child native claim loops inspect child/STOP; root STOP alone cannot drain them.
 for path in [WORK/'STOP',*[base.child(step)/'STOP' for step in STEPS]]:
  if path.exists():assert read(path).get('owner')=='cluster29_handoff','Unrelated STOP: '+str(path)
 if live:
  assert str(WORK/'run.py') in cmd and 'controller' in cmd
  os.kill(pid,signal.SIGSTOP)
 old=[]
 for f in (WORK/'processes').glob('*.json'):
  r=read(f)
  if base.own_live(r):old.append(r)
 try:handoff_stops(create=True)
 except BaseException:
  handoff_stops()
  if live:os.kill(pid,signal.SIGCONT)
  raise
 save(STATE/'HANDOFF_PENDING.json',dict(at=time.time(),old_controller=pid,old_children=old))
 started=time.time()
 while any(base.own_live(r) for r in old):
  save(STATE/'handoff_status.json',dict(at=time.time(),stage='draining_current_samples',live_children=[r['pid'] for r in old if base.own_live(r)]))
  if time.time()-started>600:
   # Safe rollback: resume old scheduling instead of killing unfinished samples.
   handoff_stops()
   if live:os.kill(pid,signal.SIGCONT)
   raise TimeoutError('GPU4 work did not drain in600s; original controller resumed')
  time.sleep(3)
 if live:
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  for _ in range(20):
   try:
    if proc.joinpath('stat').read_text().split(') ',1)[1][0]=='Z':break
   except FileNotFoundError:break
   time.sleep(.5)
  else:raise RuntimeError('Old controller not exited; leave STOP and do not start cluster')
 handoff_stops()
 for f in WORK.glob('failure_*.json'):
  if read(f).get('at',0)>=started:
   history=STATE/'handoff_history';history.mkdir(exist_ok=True);f.rename(history/f.name)
 save(STATE/'HANDOFF.json',dict(success=True,at=time.time(),prior_controller=pid,drained_child_pids=[r['pid'] for r in old],completed_outputs_preserved=True,old_controller_exited=True))

def device(rank):
 c=cfg();s=slots(c)[rank];assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(s['gpu'])
 assert s['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 uuid=subprocess.check_output(['nvidia-smi','-i',str(s['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 acceptance=read(STATE/'ACTIVATED.json');assert uuid==acceptance['uuids'][str(rank)]
 return s

def worker(stage,rank,step=0):
 verify();s=device(rank)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(0)
 torch.cuda.set_per_process_memory_fraction((96 if stage=='render' else 38)*2**30/torch.cuda.get_device_properties(0).total_memory)
 from p2_testsets_eval import runtime as rt
 from p2_listen.config import extractor_config
 d=read(WORK/'DATA_CONFIG.json');c=extractor_config(d);items=read(WORK/'shared_items.json');progress=rt.progress
 record=dict(at=time.time(),pid=os.getpid(),hostname=socket.gethostname(),**s,stage=stage,step=step)
 save(STATE/'devices'/f'{stage}_{rank}_{step}.json',record)
 lock=STATE/'locks'/f'{stage}_{rank}_{step}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('a') as lf,ExitStack() as stack:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stack.enter_context(patch.object(rt,'data_config',lambda:(d,c)))
  stack.enter_context(patch.object(rt,'progress',lambda w,st,unused,**kw:progress(w,st,rank,**kw)))
  if stage=='render':
   import p2_all_lora_followups_eval_20261109 as native
   import p2_ltx_events.inference as old
   from p2_ltx_all_lora import inference,checkpoint,config
   p=read(base.child(step)/'protocol.json');plan=read(WORK/'PLAN.json');normal=plan['normalized_config'];identity=plan['identity']
   checked_validate=config.validate
   def load(path,device='cuda:0'):
    assert Path(path)==Path(p['arms'][0]['checkpoint']);v=base.pinned(step)
    def normalized(conf):
     assert conf==identity['experiment'];return checked_validate(copy.deepcopy(normal))
    with patch.object(checkpoint,'checked',lambda _:v),patch.object(config,'source_identity',lambda:identity['source']),patch.object(config,'validate',normalized):
     loaded=inference.load_model(path,device)
    assert len(loaded[0].text_targets)==1632 and list(loaded[0].state_dict())==list(v['model'])
    save(STATE/'model_acceptance'/f'{step}_{rank}.json',dict(**record,passed=True,strict_tensors=4809,targets=1632,original_sha256=p['arms'][0]['original_sha256']))
    return loaded
   native.STEP=step;native.ARM=p['arms'][0]['name'];native.WORK=base.child(step)
   with patch.object(old,'load_model',load):native.render(base.child(step),items,0,p)
   # Peers may still hold the final sample; only controller declares full completion.
  elif stage=='decode':
   with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']):
    for st in STEPS:
     if base.stopped():break
     w=base.child(st);p=read(w/'protocol.json');arm=p['arms'][0]['name']
     if any(rt.ready(w/'latents'/arm/x['key']) and not rt.ready(w/'generated'/arm/x['key']) for x in items):rt.decode(w,items,rank,p,p['arms'])
  save(STATE/'worker_complete'/f'{stage}_{rank}_{step}.json',dict(**record,completed_at=time.time()))

def environment(stage,gpu):
 return dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if stage in ('metric','merge') else '')
def command(stage,rank,step=0,shard=0):
 s=slots(cfg())[rank]
 if stage=='metric':
  name=read(base.child(step)/'protocol.json')['arms'][0]['name']
  cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(base.child(step)),'--rank',str(shard),'--arms',name]
 else:cmd=[str(PYTHON),'-u',str(SELF),'worker','--stage',stage,'--rank',str(rank),'--step',str(step)]
 return ['env',*[f'{k}={v}' for k,v in environment(stage,s['gpu']).items()],*cmd]

def launch(stage,rank,step=0,shard=0,attempt=1):
 from listen_cluster import transport
 c=cfg();s=slots(c)[rank];node=next(n for n in c['nodes'] if n['name']==s['node'])
 cmd=['env','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',str(PYTHON),str(GUARD),'--guard',*command(stage,rank,step,shard)]
 log=STATE/'logs'/f'{stage}_{rank}_{step}_{shard}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('ab',buffering=0) as stream:p=subprocess.Popen(transport(c,node,cmd),stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT)
 rec=dict(at=time.time(),pid=p.pid,stage=stage,rank=rank,step=step,shard=shard,attempt=attempt,log=str(log),**{k:s[k] for k in ['node','gpu']})
 save(STATE/'processes'/f'{stage}_{rank}_{step}_{shard}.json',rec);return dict(process=p,**rec)

def snapshot():
 from p2_testsets_eval.runtime import ready
 items=read(WORK/'shared_items.json');out={}
 for step in STEPS:
  w=base.child(step);arm=read(w/'protocol.json')['arms'][0]['name']
  done={x['key'] for x in items if ready(w/'generated'/arm/x['key'])}
  latent={x['key'] for x in items if ready(w/'latents'/arm/x['key'])}|done
  remaining=[x for x in items if x['key'] not in latent]
  scores={i for i in range(4) if (f:=w/'progress/metrics'/f'{arm}_{i:02d}.json').exists() and read(f).get('status')=='complete'}
  out[step]=dict(done=done,latent=latent,remaining=len(remaining),cost=sum(x['geometry']['native_frames'] for x in remaining),scores=scores,merged=(w/'COMPLETE.json').exists())
 return out

def choose_render(rank,state,active):
 counts=collections.Counter(j['step'] for j in active.values() if j['stage']=='render')
 candidates=[s for s in STEPS if state[s]['remaining']>counts[s] and counts[s]<2]
 if not candidates:return None
 preferred=STEPS[min(rank//2,14)]
 if preferred in candidates:return preferred
 return max(candidates,key=lambda s:(-counts[s],state[s]['cost']/(counts[s]+1),-s))

# Imported explicitly here so module import stays CPU-only.
import collections

def queue():
 from p2_progressive_eval_20261108 import ResourcePoller
 verify();assert (STATE/'ACTIVATED.json').exists()
 assert cfg()['nodes'][-1]['host'] in subprocess.check_output(['hostname','-I'],text=True).split()
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (STATE/'controller.lock').open('a') as lock,(WORK/'controller.lock').open('a') as oldlock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(STATE/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname()))
  poller=ResourcePoller(cfg(),probe_node,interval=10);active={};attempts={};failed={};published=set();merges={}
  mapping=slots(cfg());next_update=0
  def start(stage,rank,step=0,shard=0):
   key=(stage,step,shard) if stage=='metric' else (stage,rank,step)
   attempts[key]=attempts.get(key,0)+1;active[key]=launch(stage,rank,step,shard,attempts[key])
  try:
   while not base.stopped() and not (STATE/'STOP').exists():
    for key,job in list(active.items()):
     rc=job['process'].poll()
     if rc is None:continue
     if job['process'].stdin:job['process'].stdin.close()
     del active[key]
     if rc:
      failed[key]=failed.get(key,0)+1
      if failed[key]>=(6 if rc==255 else 3):raise RuntimeError('Repeated worker error: '+job['log'])
    for step,proc in list(merges.items()):
     rc=proc.poll()
     if rc is None:continue
     del merges[step]
     if rc:raise RuntimeError(f'Merge failed for{step}')
    state=snapshot();reports=poller.snapshot()
    acceptance=read(STATE/'ACTIVATED.json')
    for node in cfg()['nodes']:
     missing=[s for s in mapping if s['node']==node['name'] and str(s['rank']) not in acceptance['uuids']]
     report=reports.get(node['name'],{})
     if missing and 'gpus' in report and not blockers(node,report):
      for slot in missing:acceptance['uuids'][str(slot['rank'])]=report['gpus'][str(slot['gpu'])]['uuid']
      acceptance.setdefault('late_join',[]).append(dict(at=time.time(),node=node['name'],ranks=[s['rank'] for s in missing]))
      save(STATE/'ACTIVATED.json',acceptance)
    used={j['rank'] for j in active.values()};started=collections.Counter()
    for s in mapping:
     rank=s['rank'];report=reports.get(s['node'],{})
     if str(rank) not in acceptance['uuids'] or rank in used or started[s['node']]>=2 or 'gpus' not in report:continue
     g=report['gpus'].get(str(s['gpu']),report['gpus'].get(s['gpu']));apps=[a for a in report['apps'] if a['uuid']==g['uuid'] and a['state']!='Z']
     # No task is launched onto occupied cards; no unrelated process is killed.
     if apps or g['used']>2048:continue
     score=None
     for step in STEPS:
      if state[step]['merged']:continue
      for shard in range(4):
       key=('metric',step,shard)
       if shard in state[step]['scores'] or key in active:continue
       if all(x['key'] in state[step]['done'] for x in read(base.child(step)/'shards'/f'{shard:02d}.json')):score=(step,shard);break
      if score:break
     candidate=choose_render(rank,state,active)
     backlog=sum(len(v['latent']-v['done']) for v in state.values());decoders=sum(j['stage']=='decode' for j in active.values())
     # Keep up to4 ready scorers and4 decoders while rendering remains; each
     # released card can switch, then all29 are available once sampling ends.
     active_render=sum(j['stage']=='render' for j in active.values())
     score_count=sum(j['stage']=='metric' for j in active.values())
     if score and (score_count<4 or candidate is None):
      step,shard=score
      if score not in published:base.publish(step,shard);published.add(score)
      start('metric',rank,step,shard)
     elif backlog>max(0,decoders*8) and (decoders<4 or candidate is None):start('decode',rank)
     elif candidate is not None:start('render',rank,candidate)
     elif backlog>decoders:start('decode',rank)
     else:continue
     used.add(rank);started[s['node']]+=1
    for step,v in state.items():
     if len(v['scores'])==4 and not v['merged'] and step not in merges:
      env=dict(os.environ,**environment('merge',0));env['CUDA_VISIBLE_DEVICES']=''
      log=STATE/'logs'/f'merge_{step}.log'
      with log.open('a') as f:merges[step]=subprocess.Popen([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(base.child(step))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    if time.time()>=next_update:
     info=dict(at=time.time(),stage='cluster29',generated=sum(len(v['done']) for v in state.values()),latent=sum(len(v['latent']) for v in state.values()),total=960,completed_checkpoints=[s for s,v in state.items() if v['merged']],active_gpus=len(used),max_gpus=29,active=[{k:j[k] for k in ['stage','rank','step','shard','node','gpu']} for j in active.values()])
     save(STATE/'status.json',info);save(WORK/'status.json',info);base.results();next_update=time.time()+15
    if all(v['merged'] for v in state.values()) and not active and not merges:
     base.results(final=True);save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=960,steps=STEPS,physical_gpus=29,screening=str(WORK/'SCREENING.json')));save(STATE/'COMPLETE.json',dict(success=True,at=time.time()));return
    time.sleep(3)
  finally:
   poller.close()
   for j in active.values():
    if j['process'].stdin:j['process'].stdin.close()
   for p in merges.values():
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)

def activate(released=False,allow_unavailable=False):
 assert released,'Pass --released only after user confirms cards released'
 verify()
 def accepted_resources(allow_own=False):
  r=check_resources(allow_own_gpu4=allow_own)
  fatal=[x for x in r['blockers'] if not (allow_unavailable and 'error' in x)]
  if fatal or not r['nodes']:raise RuntimeError('Assigned cards occupied or unavailable; see RESOURCE_CHECK.json')
  return r
 r=accepted_resources(True)
 handoff();r=accepted_resources()
 uuids={str(s['rank']):r['nodes'][s['node']]['gpus'][str(s['gpu'])]['uuid'] for s in slots(cfg()) if s['node'] in r['nodes']}
 save(STATE/'ACTIVATED.json',dict(at=time.time(),released_confirmed=True,uuids=uuids,initial_gpus=len(uuids),deferred_nodes=[n['name'] for n in cfg()['nodes'] if n['name'] not in r['nodes']],resource_check_sha256=sha(STATE/'RESOURCE_CHECK.json')))

 queue()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','activate','worker','queue']);p.add_argument('--released',action='store_true');p.add_argument('--allow-unavailable',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--step',type=int,default=0);p.add_argument('--stage',choices=['render','decode']);a=p.parse_args()
 try:
  if a.action=='prepare':prepare()
  elif a.action=='check':print(json.dumps(check_resources(allow_own_gpu4=True),ensure_ascii=False))
  elif a.action=='activate':activate(a.released,a.allow_unavailable)
  elif a.action=='queue':queue()
  else:worker(a.stage,a.rank,a.step)
 except BaseException:
  save(STATE/f'failure_{a.action}_{a.rank}_{a.step}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
