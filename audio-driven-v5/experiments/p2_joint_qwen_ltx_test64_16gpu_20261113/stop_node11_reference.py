import json,time,os,signal,subprocess,sys,shutil
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');W=ROOT/'experiments/p2_reference_methods_seamless6624_20261112';R=ROOT/'experiments/p2_reference_methods_responsenet586_20261112';OUT=ROOT/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113'
def read(p):return json.loads(Path(p).read_text())
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pause.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def process(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,state=p.joinpath('stat').read_text().split(') ',1)[1][0],command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),pgid=os.getpgid(pid))
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,state='absent',command='')
def owned(pid):
 p=process(pid);return p['state'] not in ['Z','absent'] and any(str(w)+'/' in p['command'] for w in [W,R])
assert '172.29.189.119' in subprocess.check_output(['hostname','-I'],text=True).split()
B=W/'pause_20261113_user_node11_release';B.mkdir(exist_ok=True)
for w in [W,R]:
 for n in ['CONTROLLER_STATE.json','MONITOR.json','status.json','RESOURCE_POLICY.json','FOLLOWUPS.json','FOLLOWUPS_STATUS.json']:
  p=w/n;target=B/w.name/n
  if p.exists() and not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
monitors=[]
J=ROOT/'experiments/p2_vico_additional_baselines_20261107/jobs'
for name in ['reference_seamless6624_20261113_monitor_resident_v1','reference_responsenet586_20261112_followup']:
 p=J/name/'running.json'
 if p.exists():
  r=read(p)
  for key in ['pid','supervisor_pid']:
   if r.get(key) and owned(r[key]):monitors.append(process(r[key]))
controllers=[]
for w in [W,R]:
 for name in ['monitor/controller.json','MONITOR_PROCESS.json']:
  p=w/name
  if p.exists() and owned(read(p)['pid']):controllers.append(process(read(p)['pid']))
# Prevent any new work before releasing GPU workers.
for r in monitors+controllers:
 if owned(r['pid']):os.kill(r['pid'],signal.SIGSTOP)
policy=read(W/'RESOURCE_POLICY.json');policy.update(mode='paused_by_user_release_node11',allowed_gpus=[],disabled_gpus=list(range(8)),user_latest='2026-11-13: stop all our node11 jobs includingGPU7; no automatic resume',updated_at=time.time());save(W/'RESOURCE_POLICY.json',policy)
follow=read(W/'FOLLOWUPS.json');save(W/'FOLLOWUPS.json',dict(schema=follow['schema'],entries=[],paused_by_user=True,paused_at=time.time(),saved_entries=str(B/W.name/'FOLLOWUPS.json')))
for w in [W,R]:save(w/'PAUSED_BY_USER.json',dict(at=time.time(),reason='Release all node11 resources includingGPU7',owner='user',automatic_resume=False,backup=str(B)))
for r in monitors+controllers:
 if owned(r['pid']):os.kill(r['pid'],signal.SIGTERM);os.kill(r['pid'],signal.SIGCONT)
state=read(W/'CONTROLLER_STATE.json');workers=[];groups=[]
for tag,r in state['active'].items():
 rr=read(Path(r['directory'])/'running.json');pid=rr['pid'];sup=r['supervisor_pid'];a=process(pid);b=process(sup)
 if owned(pid):
  assert owned(sup) and a['pgid']==b['pgid']==sup,(a,b)
  workers.append(dict(tag=tag,worker=a,supervisor=b,attempt=r['attempt'],directory=r['directory']));groups.append(sup);os.killpg(sup,signal.SIGTERM);os.killpg(sup,signal.SIGCONT)
tracked=[r['pid'] for r in monitors+controllers]+[v[k]['pid'] for v in workers for k in ['worker','supervisor']]
for _ in range(20):
 if not any(owned(pid) for pid in tracked):break
 time.sleep(1)
forced=[]
for sup in groups:
 if owned(sup) or any(owned(v['worker']['pid']) for v in workers if v['supervisor']['pid']==sup):
  try:os.killpg(sup,signal.SIGKILL);forced.append(sup)
  except ProcessLookupError:pass
for pid in tracked:
 if owned(pid):
  os.kill(pid,signal.SIGKILL);forced.append(pid)
for _ in range(10):
 if not any(owned(pid) for pid in tracked):break
 time.sleep(1)
remaining=[process(pid) for pid in tracked if owned(pid)]
for v in workers:
 d=Path(v['directory']);exitfile=d/f"exit_{v['attempt']:03d}.json"
 if exitfile.exists():shutil.copyfile(exitfile,B/f"{v['tag']}_original_exit.json")
 save(exitfile,dict(returncode=0,planned_defer=True,reason='Explicit user pause releasingnode11 GPU7',worker_pid=v['worker']['pid'],ended=time.time(),gpu=7))
state.update(active={},paused_by_user=True,paused_at=time.time(),leases={},cooldown={},empty_since={});save(W/'CONTROLLER_STATE.json',state)
for w in [W,R]:
 p=w/'status.json';s=read(p) if p.exists() else {};s.update(stage='paused_by_user',at=time.time(),active=[],active_gpus={},paused_workers=[],automatic_resume=False,pause_record=str(OUT/'NODE11_GPU7_PAUSED.json'));save(p,s)
mon=read(W/'MONITOR.json');mon.update(stage='paused_by_user',at=time.time(),automatic_resume=False);save(W/'MONITOR.json',mon)
f=read(W/'FOLLOWUPS_STATUS.json');f.update(stage='paused_by_user',at=time.time(),automatic_resume=False)
for s in f.get('entries',{}).values():s.update(stage='paused_by_user',at=time.time())
save(W/'FOLLOWUPS_STATUS.json',f)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
report=dict(at=time.time(),success=not remaining,scope='All node11 currentQwen/LTX plus reference GPU7 and ResponseNet followup',monitor_and_controller=monitors+controllers,workers=workers,forced_pids=forced,remaining=remaining,gpu_snapshot=gpu,gpu_apps=apps,progress_before=read(B/W.name/'status.json'),backup=str(B),resume_requires_user_instruction=True,results_preserved=True)
save(OUT/'NODE11_GPU7_PAUSED.json',report);save(W/'PAUSE_RECEIPT.json',report);save(R/'PAUSE_RECEIPT.json',dict(at=time.time(),success=not remaining,automatic_followup_disabled=True,source=str(OUT/'NODE11_GPU7_PAUSED.json')))
print(json.dumps(dict(success=not remaining,stopped_gpu_workers=len(workers),remaining=remaining,gpu_snapshot=gpu,apps=apps),ensure_ascii=False));assert not remaining
