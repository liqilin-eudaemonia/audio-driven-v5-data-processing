"""Authorized once-only chain: wait for seed137 finish, run238, aggregate both."""
import concurrent.futures,fcntl,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from aggregate import SOURCES,WORK,read,save,aggregate
ROOT=WORK.parents[1];PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';FIRST=SOURCES[137];SECOND=SOURCES[238]
def ready(w):
 try:
  c=read(w/'COMPLETE.json');s=read(w/'status.json');r=read(w/'RESULTS.json')
  return c.get('success') is True and c.get('outputs')==1024 and c.get('combinations')==4 and s.get('generated')==1024 and s.get('latent')==1024 and s.get('active_gpus')==0 and not s.get('active') and r.get('completed_combinations')==4
 except (FileNotFoundError,json.JSONDecodeError):return False

def controller_live(w):
 try:
  pid=read(w/'CONTROLLER.json')['pid'];p=Path('/proc')/str(pid)
  return p.joinpath('stat').read_text().split(') ',1)[1][0]!='Z' and str(w/'run.py').encode() in p.joinpath('cmdline').read_bytes()
 except (FileNotFoundError,ProcessLookupError):return False

def stopped():return any((p/'STOP').exists() for p in [WORK,FIRST,SECOND])
def report(stage,**fields):save(WORK/'status.json',dict(at=time.time(),pid=os.getpid(),stage=stage,**fields))
def load_runner():
 s=importlib.util.spec_from_file_location('seed238_runner',SECOND/'run.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.verify();return m

def first_media_acceptance(m):
 receipt=WORK/'SEED238_LIVE_ACCEPTANCE.json'
 if receipt.exists():return
 checks=[]
 for c in m.plans():
  p=read(Path(c['work'])/'protocol.json')
  for x in read(SECOND/'shared_items.json'):
   d=Path(c['work'])/'latents'/c['key']/x['key']
   if not (d/'complete.json').exists():continue
   a=read(d/'metadata.json');assert a['seed']==238 and a['step']==c['ltx'] and a['planning_adapter']==p['planner_adapter_identity'] and a['generation']==p['generation'] and a['planning_sha256']==m.sha(Path(c['work'])/'conditions'/x['key']/'qwen/plan.json')
   assert not d.is_symlink() and not a['target_description_access'] and not a['listener_future_input']
   checks.append(dict(combination=c['key'],sample=x['key'],seed=a['seed'],new_latent=True));break
 if len(checks)==4:save(receipt,dict(success=True,at=time.time(),checks=checks))

def main():
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (WORK/'followup.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(WORK/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=os.uname().nodename));m=load_runner();child=None
  try:
   while not ready(FIRST) or controller_live(FIRST):
    if stopped():report('paused_by_stop');return
    state=read(FIRST/'status.json');report('waiting_seed137_completion',first_generated=state['generated'],first_combinations=len(state['completed_combinations']),first_active_gpus=state['active_gpus']);time.sleep(15)
   save(WORK/'FIRST_SEED_ACCEPTED.json',dict(success=True,at=time.time(),complete_sha256=m.sha(FIRST/'COMPLETE.json'),results_sha256=m.sha(FIRST/'RESULTS.json')))
   attempts=0
   while not ready(SECOND):
    if stopped():report('paused_by_stop');return
    if controller_live(SECOND):
     first_media_acceptance(m);report('seed238_running_existing',state=read(SECOND/'status.json'));time.sleep(15);continue
    try:
     with concurrent.futures.ThreadPoolExecutor(2) as pool:reports={n['name']:r for n,r in zip(m.config()['nodes'],pool.map(m.probe,m.config()['nodes']))}
     free=all(reports[s['node']]['gpus'][str(s['gpu'])]['used']<2048 and reports[s['node']]['gpus'][str(s['gpu'])]['uuid'] not in reports[s['node']]['apps'] for s in m.slots())
    except Exception as e:report('waiting_resources',error=repr(e));time.sleep(15);continue
    if not free:report('waiting_resources',reason='Some authorized GPUs still occupied; no other tasks stopped');time.sleep(15);continue
    attempts+=1
    if attempts>2:raise RuntimeError('Seed238 controller failed twice; inspect second-seed log')
    with (SECOND/'controller.log').open('ab',buffering=0) as f:child=subprocess.Popen([str(PYTHON),'-u',str(SECOND/'run.py'),'activate'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(SECOND/'LAUNCHED.json',dict(at=time.time(),pid=child.pid,after_first_seed_complete=True,first_seed_complete_sha256=m.sha(FIRST/'COMPLETE.json'),attempt=attempts,seed=238))
    while child.poll() is None:
     if stopped():child.terminate();report('paused_by_stop');return
     first_media_acceptance(m);report('seed238_running',controller_pid=child.pid,attempt=attempts);time.sleep(15)
    rc=child.returncode;child=None
    if rc:report('seed238_retry_pending',returncode=rc,attempt=attempts);time.sleep(15)
   first_media_acceptance(m);result=aggregate();report('complete',seeds=[137,238],combinations=len(result['results']),total_evaluations=2048)
  finally:
   if child is not None and child.poll() is None:child.terminate()
if __name__=='__main__':
 try:main()
 except BaseException:
  save(WORK/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
