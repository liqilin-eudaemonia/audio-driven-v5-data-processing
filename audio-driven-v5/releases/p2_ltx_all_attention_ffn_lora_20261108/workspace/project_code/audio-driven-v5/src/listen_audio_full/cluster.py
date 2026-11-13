"""Durable, three-host full-audio pipeline, GPU-gated behind completed E1+Feature."""
import argparse,collections,concurrent.futures,fcntl,json,os,shlex,signal,socket,sqlite3,subprocess,sys,time,uuid
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha,atomic_json
from .producer import verify_code
HOSTS=['172.29.252.11','172.29.15.91','172.29.225.51']
NAMES=['dedicated-developjob-lq-dev-machine-1nkpf','dedicated-developjob-lq-dev-machine-2-taaft','dedicated-developjob-lq-dev-machine-3-zhw7q']
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';SCRIPT=ROOT/'scripts/p2_full_audio_postprocess.py'
TRAIN_QUEUE=ROOT/'experiments/p2_e1_feature01_train50h_20261026'
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
PREDECESSOR=ROOT/'experiments/p2_feature01_train50h_20261026/production/runs/20261023/E1'

def optional(path):
 try:return read(path)
 except FileNotFoundError:return {}

def call(rank,args,timeout=90):
 cmd=[str(PYTHON),str(SCRIPT),*map(str,args)]
 if socket.gethostname()!=NAMES[rank]:cmd=['ssh','-i',KEY,'-o','UserKnownHostsFile='+str(TRAIN_QUEUE/'queue_known_hosts'),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','root@'+HOSTS[rank],shlex.join(cmd)]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,close_fds=True)
 if r.returncode:raise RuntimeError(f'Node{rank}: {r.stderr[-2000:]} {r.stdout[-1000:]}')
 return json.loads(r.stdout)

def inspect(run):
 found=[]
 for rawpid in Path('/sys/fs/cgroup/cgroup.procs').read_text().split():
  pid=int(rawpid)
  if pid==os.getpid():continue
  try:
   argv=[s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
   stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
   if stat[0]=='Z' or str(SCRIPT) not in argv:continue
   if argv[argv.index('--run')+1]!=str(run):continue
   role=argv[argv.index(str(SCRIPT))+1]
   if role not in ('index','prepare','node','worker','audit'):continue
   found.append(dict(pid=pid,birth=stat[19],role=role,pgid=os.getpgid(pid),argv=argv))
  except (FileNotFoundError,ProcessLookupError,ValueError,IndexError):continue
 return found

def gpu_idle():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 return not apps and len(rows)==8 and all(int(v)<1024 for v in rows)

def agent(run,rank,action,generation=None):
 if socket.gethostname()!=NAMES[rank]:raise ValueError('Host/rank mismatch')
 if action=='status':return dict(rank=rank,owned=inspect(run),idle=gpu_idle(),node=optional(run/'nodes'/f'{rank}.json'))
 verify_code(run)
 if (run/'STOP').exists():raise RuntimeError('Full processing STOP')
 if action=='stop-node':
  owned=[x for x in inspect(run) if x['role'] in ('node','worker')]
  for sig in (signal.SIGTERM,signal.SIGKILL):
   for item in owned:
    current={x['pid']:x for x in inspect(run)}
    if item['pid'] in current and current[item['pid']]['birth']==item['birth']:
     try:os.killpg(item['pgid'],sig)
     except ProcessLookupError:pass
   if sig==signal.SIGTERM:time.sleep(5)
  return dict(stopped=True)
 role=action.removeprefix('start-')
 if role not in ('index','prepare','node','audit'):raise ValueError('Unknown agent action')
 if role!='node' and rank!=0:raise ValueError('CPU producer/auditor run on node0')
 with (run/f'launch-{role}-{rank}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  existing=[x for x in inspect(run) if x['role']==role]
  if existing:return dict(adopted=True,processes=existing)
  state=optional(run/'status.json')
  if role=='node':
   if state.get('generation')!=generation or not state.get('gpu_authorized'):raise ValueError('Stale/unauthorized GPU launch')
   if not gpu_idle():raise RuntimeError('GPU not free; never terminate unrelated tasks')
  if role=='audit' and (state.get('generation')!=generation or not state.get('gpu_authorized')):raise ValueError('Stale audit generation')
  cmd=[str(PYTHON),str(SCRIPT),role,'--run',str(run)]
  if role=='node':cmd+=['--node-rank',str(rank),'--generation',generation]
  if role=='audit':cmd+=['--generation',generation]
  logpath=run/'logs'/f'{role}-{rank}.log';logpath.parent.mkdir(exist_ok=True)
  with logpath.open('a') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',**({'CUDA_VISIBLE_DEVICES':''} if role!='node' else {})})
  return dict(pid=child.pid,role=role,generation=generation)

def node_worker(run,rank,generation):
 if socket.gethostname()!=NAMES[rank]:raise ValueError('Host mismatch')
 from listen_benchmark.full_fusion import lock
 with lock(run/'nodes'/f'{rank}.lock') as acquired:
  if not acquired:raise RuntimeError('Node already active')
  verify_code(run);ranks=list(range(rank*32,(rank+1)*32));children={};retry={};started=time.time()
  while True:
   if (run/'STOP').exists():raise RuntimeError('Full processing STOP')
   alive=inspect(run);reports={r:optional(run/'workers'/f'{r:03d}.json') for r in ranks};done=[]
   for worker_rank in ranks:
    report=reports[worker_rank]
    proc=children.get(worker_rank)
    if proc is not None and proc.poll() is not None:
     if proc.returncode!=0:retry[worker_rank]=time.time()+30
     del children[worker_rank];proc=None
    if report.get('generation')==generation and report.get('finished') and proc is None:
     done.append(worker_rank);continue
    if proc is not None:continue
    matching=[x for x in alive if x['role']=='worker' and x['argv'][x['argv'].index('--rank')+1]==str(worker_rank)]
    if matching:continue  # Adopt existing worker after node-supervisor restart.
    if time.time()<retry.get(worker_rank,0):continue
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(worker_rank%8),'FULL_AUDIO_GENERATION':generation,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
    with (run/'logs'/f'worker-{worker_rank:03d}.log').open('a') as log:
     children[worker_rank]=subprocess.Popen([str(PYTHON),str(SCRIPT),'worker','--run',str(run),'--rank',str(worker_rank)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
    time.sleep(.1)
   result=dict(rank=rank,generation=generation,finished=len(done)==32,finished_workers=len(done),active=len(children),
    completed=sum(r.get('completed',0) for r in reports.values()),audited_tasks=sum(r.get('audited_tasks',0) for r in reports.values()),failed_tasks=sum(r.get('failed_tasks',0) for r in reports.values()),updated=time.time(),started=started)
   atomic_json(run/'nodes'/f'{rank}.json',result)
   if result['finished']:return result
   time.sleep(5)

def predecessor_ready():
 state=optional(TRAIN_QUEUE/'queue/state.json');status=optional(PREDECESSOR/'status.json')
 if state.get('phase')!='complete' or state.get('stage')!='E1' or status.get('step')!=1268 or status.get('stopped') is not False:return None
 path=PREDECESSOR/'step_001268.pt';marker=read(path.with_suffix('.complete.json'))
 if marker['step']!=1268 or sha(path)!=marker['sha256']:raise ValueError('Predecessor final checkpoint invalid')
 return dict(path=str(path),sha256=marker['sha256'],step=1268)

def finalize(run,generation):
 from .indexing import connect
 from .worker import audit_task
 from p2_listen.config import load,activate
 import torch
 torch.set_num_threads(1);verify_code(run);c=load(run/'view/config.json');activate(c)
 producer=read(run/'producer_complete.json');launch=read(run/'launch.json');tasks=[]
 for rank in range(launch['shards']):
  path=run/'queues'/f'{rank:03d}.jsonl';expect=producer['queues'][str(rank)]
  if expect['tasks']:
   if sha(path)!=expect['sha256']:raise ValueError('Task queue changed')
   with path.open() as f:tasks.extend(json.loads(l) for l in f)
 counts=collections.Counter();classes=collections.Counter();temp=run/f'audit-{generation}.sqlite.tmp';temp.unlink(missing_ok=True)
 db=sqlite3.connect(temp);db.execute('CREATE TABLE verified(sid TEXT PRIMARY KEY)');bindings=[]
 # Full actual-source/payload/label audit; completion receipts alone never suffice.
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  iterator=iter(tasks)
  while batch:=list(__import__('itertools').islice(iterator,32)):
   for task,result in zip(batch,pool.map(lambda t:audit_task(run,c,t,final=True),batch)):
    db.executemany('INSERT INTO verified VALUES(?)',[(sid,) for sid in result['ids']]);counts.update(result['counts']);classes.update(result['classes'])
    report=run/'final_audits'/f'{task["task_id"]}.json';bindings.append(dict(task_id=task['task_id'],path=str(report),sha256=sha(report)))
   db.commit();atomic_json(run/'audit_progress.json',dict(completed=counts['samples'],total=producer['samples'],updated=time.time()))
 db.execute('ATTACH DATABASE ? AS scope',(str(run/'scope.sqlite'),))
 if db.execute('SELECT count(*) FROM verified').fetchone()[0]!=producer['samples'] or db.execute('SELECT count(*) FROM scope.samples LEFT JOIN verified USING(sid) WHERE verified.sid IS NULL').fetchone()[0] or db.execute('SELECT count(*) FROM verified LEFT JOIN scope.samples USING(sid) WHERE scope.samples.sid IS NULL').fetchone()[0]:raise ValueError('Final ID set mismatch')
 db.close();os.replace(temp,run/'audit_ids.sqlite')
 if counts['samples']!=launch['expected_samples'] or len(classes)!=7:raise ValueError('Final count/taxonomy mismatch')
 report=dict(status='passed',generation=generation,samples=counts['samples'],counts=dict(counts),class_sample_counts=dict(classes),
  tasks=bindings,scope_sha256=sha(run/'scope.json'),code_manifest_sha256=sha(run/'code_manifest.json'),config_sha256=sha(run/'view/config.json'),
  all_source_audio_and_new_payload_hashes_verified=True,training_started=False,formal_training_admitted=False,wav_files_written=0,video_cache_copies=0,finished=time.time())
 atomic_json(run/'audit.json',report)
 complete=dict(status='passed',generation=generation,samples=counts['samples'],audit_sha256=sha(run/'audit.json'),scope_sha256=sha(run/'scope.json'),config_sha256=sha(run/'view/config.json'),code_manifest_sha256=sha(run/'code_manifest.json'),training_started=False)
 atomic_json(run/'COMPLETE.json',complete);atomic_json(run/'audit_generations'/f'{generation}.json',complete);return complete

def supervisor_tick(run,rank,state):
 verify_code(run)
 def save(**values):state.update(last_error=None);state.update(values,updated=time.time(),leader=dict(rank=rank,pid=os.getpid(),host=socket.gethostname()));atomic_json(run/'status.json',state)
 if (run/'STOP').exists():save(phase='paused');return
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  nodes=list(pool.map(lambda r:call(r,['agent','--run',run,'--node-rank',r,'--action','status']),range(3)))
 if not (run/'scope.json').exists():
  if not any(x['role']=='index' for x in nodes[0]['owned']):call(0,['agent','--run',run,'--node-rank',0,'--action','start-index'])
  save(phase='scope_index',index=optional(run/'index_progress.json'));return
 if (not (run/'producer_complete.json').exists() or any((run/'preparation_repairs').glob('*.json'))) and not any(x['role']=='prepare' for x in nodes[0]['owned']):call(0,['agent','--run',run,'--node-rank',0,'--action','start-prepare'])
 if not (run/'view/config.json').exists():save(phase='waiting_view_config');return
 if not state.get('gpu_authorized'):
  predecessor=predecessor_ready()
  if predecessor is None:save(phase='cpu_prepare_waiting_E1',producer=optional(run/'producer_status.json'));return
  if not all(n['idle'] for n in nodes):save(phase='waiting_gpu_resources',predecessor=predecessor);return
  save(phase='gpu',gpu_authorized=True,predecessor=predecessor,generation=state.get('generation') or uuid.uuid4().hex)
 generation=state['generation']
 for n in nodes:
  nr=n['rank'];status=n['node'];done=status.get('finished') and status.get('generation')==generation
  alive=[x for x in n['owned'] if x['role'] in ('node','worker')]
  if done and not alive:continue
  if any(x['role']=='node' for x in alive):
   if status and time.time()-status.get('updated',time.time())>180:
    call(nr,['agent','--run',run,'--node-rank',nr,'--action','stop-node'])
   else:continue
  if alive:call(nr,['agent','--run',run,'--node-rank',nr,'--action','stop-node'])
  call(nr,['agent','--run',run,'--node-rank',nr,'--action','start-node','--generation',generation])
 done=all(n['node'].get('finished') and n['node'].get('generation')==generation and not any(x['role'] in ('node','worker') for x in n['owned']) for n in nodes)
 if done and (run/'producer_complete.json').exists():
  if any((run/'preparation_repairs').glob('*.json')) or any(x['role']=='prepare' for x in nodes[0]['owned']):
   save(phase='waiting_preparation_settle');return
  report=run/'audit_generations'/f'{generation}.json'
  if report.exists():
   complete=read(report)
   if sha(run/'audit.json')!=complete['audit_sha256']:raise ValueError('Final audit summary changed')
   save(phase='complete',completion=complete);return
  if not any(x['role']=='audit' for x in nodes[0]['owned']):
   failure=run/'errors/audit-None.json'
   if failure.exists():
    archive=run/'resolved_errors'/f'audit-{generation}-{time.time_ns()}.json';archive.parent.mkdir(parents=True,exist_ok=True);failure.replace(archive)
    save(phase='gpu',generation=uuid.uuid4().hex,recovery_reason='Final real-payload audit failed; rerun cache validation and repair');return
   call(0,['agent','--run',run,'--node-rank',0,'--action','start-audit','--generation',generation])
  save(phase='final_audit',audit=optional(run/'audit_progress.json'));return
 completed=sum(n['node'].get('completed',0) for n in nodes)
 signature=[completed,sum(n['node'].get('audited_tasks',0) for n in nodes),sum(n['node'].get('finished_workers',0) for n in nodes)]
 if signature!=state.get('progress_signature') or 'progress_at' not in state:state['progress_at']=time.time()
 state['progress_signature']=signature
 if (run/'producer_complete.json').exists() and time.time()-state['progress_at']>900:
  for nr in range(3):call(nr,['agent','--run',run,'--node-rank',nr,'--action','stop-node'])
  save(phase='gpu',generation=uuid.uuid4().hex,progress_at=time.time(),recovery_reason='No full-cache progress for 15 minutes');return
 save(phase='gpu',completed=completed,nodes=[n['node'] for n in nodes],producer=optional(run/'producer_status.json'))

def daemon(run,rank,detach=False):
 if socket.gethostname()!=NAMES[rank]:raise ValueError('Daemon host/rank mismatch')
 if detach:
  logpath=run/f'watchdog-{rank}.log'
  with logpath.open('a') as log:child=subprocess.Popen([str(PYTHON),str(SCRIPT),'daemon','--run',str(run),'--node-rank',str(rank)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'})
  for _ in range(100):
   state=optional(run/f'watchdog-{rank}.json')
   if state.get('pid')==child.pid:return state
   if child.poll() is not None:raise RuntimeError('Full watchdog exited at startup')
   time.sleep(.2)
  raise TimeoutError('Full watchdog startup acknowledgement timeout')
 with (run/f'watchdog-{rank}.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   with (run/'leader.lock').open('a') as lease:
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
     atomic_json(run/f'watchdog-{rank}.json',dict(pid=os.getpid(),role='standby',time=time.time()));time.sleep(10);continue
    state=optional(run/'status.json')
    while True:
     atomic_json(run/f'watchdog-{rank}.json',dict(pid=os.getpid(),role='leader',time=time.time()))
     try:
      supervisor_tick(run,rank,state)
      if state.get('phase')=='complete':return dict(status='complete')
     except Exception as error:
      state.update(last_error=str(error),retry_after_seconds=30,updated=time.time());atomic_json(run/'status.json',state);print(time.time(),str(error),flush=True)
     time.sleep(30)
