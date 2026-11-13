"""End-of-window evidence audit. Never stops or restarts production."""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];M=W/'monitoring_10h'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(M/'plan.json')
assert time.time()>=plan['end'],'The authorized monitoring window has not ended'
terminal=read(M/'complete.json');state=read(M/'state.json')
assert terminal['start']==plan['start'] and terminal['end']==plan['end'] and terminal['at']>=plan['end']
assert terminal['report_slots']==list(range(21)) and state['stage']=='complete'
reports=[];reviews=[]
for slot in range(21):
 p=M/f'report-{slot:02d}.json';r=read(p);v=read(M/f'review-{slot:02d}.json')
 assert v['slot']==slot and v['report_sha256']==sha(p)
 reports.append(r);reviews.append(v)
assert reports[-1]['at']>=plan['end']
cfg=read(W/'config.json');frozen=read(W/'performance_check/optimization_frozen.json')
assert cfg['pipeline_id']==frozen['pipeline_id']
assert all(sha(ROOT/p)==h for p,h in cfg['code_hashes'].items())
assert read(W/'validation/pts_tail_cluster_audit.json')['passed']
assert read(W/'validation/pts_tail_recovery_audit.json')['passed']
remote=r'''
import json,sys,time,socket
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
w=Path(WORK);read=lambda p:json.loads(p.read_text());cfg=read(w/'config.json');nw=w/'nodes'/NODE
assert socket.gethostname()==cfg['nodes'][NODE]['hostname']
s=read(nw/'supervisor.json');watch=read(nw/'watchdog.json')
assert alive(s) and alive(watch) and not s.get('error') and not s.get('blocked_gpus')
assert not (nw/'STOP_SUPERVISION').exists()
rows=[]
for g in cfg['nodes'][NODE]['gpu_ids']:
 h=read(w/'workers'/f'{NODE}-gpu-{g}.json');c=s['children']['gpu-'+str(g)]
 assert alive(c) and h['pid']==c['pid']
 cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert 'listen_j_omni.py worker' in cmd and WORK in cmd
 assert h['execution_revision']==REV and h['preparation_profile']==cfg['preparation_profiles'][f'{NODE}-gpu-{g}']
 assert time.time()-h['heartbeat']<60 and time.time()-h['last_progress_at']<180
 assert h['completed_clips']>=5 and not h.get('last_error')
 assert h['stage'] in ('inference','preparing')
 rows.append(dict(gpu=g,pid=c['pid'],start_ticks=c['start_ticks'],completed=h['completed_clips'],held=h['held_clips'],failed=h['failed_clips'],heartbeat_age=time.time()-h['heartbeat'],progress_age=time.time()-h['last_progress_at'],execution_revision=h['execution_revision']))
print(json.dumps(dict(at=time.time(),hostname=socket.gethostname(),workers=rows,watchdog_alive=True,supervisor_alive=True)))
'''
def probe(item):
 n,c=item;code=f'ROOT={str(ROOT)!r}\nWORK={str(W)!r}\nNODE={n!r}\nREV={frozen["expected_execution_revision"]!r}\n'+remote
 cmd=['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+c['host'],'python3 -']
 r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=30)
 if r.returncode:raise RuntimeError(n+': '+r.stderr[-2000:])
 return n,json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:live=dict(pool.map(probe,cfg['nodes'].items()))
for n,d in live.items():
 old={x['gpu']:x for x in reports[19]['nodes'][n]['workers']}
 for x in d['workers']:
  o=old[x['gpu']];delta=x['completed']-o['completed'] if x['pid']==o['pid'] else None
  if delta is not None:assert delta>0
  x['normal_completed_since_report19_same_pid']=delta
q=read(W/'queues/caption/state.json');assert not q['failed']
journal=[]
with (M/'health.jsonl').open() as f:
 for line in f:journal.append(json.loads(line))
assert journal and journal[-1]['at']>=plan['end']
summary=dict(samples=len(journal),first_at=journal[0]['at'],last_at=journal[-1]['at'],max_gap_s=max(b['at']-a['at'] for a,b in zip(journal,journal[1:])),recovery_action_observations=sum(bool(d.get('actions')) for r in journal for d in r['nodes'].values()),observation_error_observations=sum(bool(d.get('observation_error')) for r in journal for d in r['nodes'].values()))
out=dict(at=time.time(),passed=True,window=plan,report_count=len(reports),review_count=len(reviews),live_nodes=live,source_failed_count=0,resolved_clips=q['completed_clips'],target_clips=cfg['total_clips'],resolved_delta_since_first_retained_report=q['completed_clips']-reports[0]['resolved_clips'],pipeline_id=cfg['pipeline_id'],all_runtime_code_hashes_match=True,journal=summary,limitations=['First retained report is about five minutes after goal start. Continuous health journal starts at 16:36 UTC; earlier evidence consists of snapshots and recorded interventions.','Planned worker replacements include short loading intervals. Observation gaps and the recovered source failure are retained in reviews. This audit does not assert uninterrupted 100% GPU utilization.','Monitoring window complete; full production annotation remains running and is not declared complete.'],review_evidence=[dict(slot=i,report_sha256=v['report_sha256']) for i,v in enumerate(reviews)])
(M/'final_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
