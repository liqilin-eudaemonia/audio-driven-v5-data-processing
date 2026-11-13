"""Bounded read-only report analysis; output is evidence for agent review."""
import argparse,hashlib,json,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('slot',type=int);a=p.parse_args()
w=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
r=read(w/f'report-{a.slot:02d}.json');prev=read(w/f'report-{a.slot-1:02d}.json') if a.slot else r
# One fixed journal, no recursive file enumeration.
j=[]
with (w/'health.jsonl').open() as f:
 for line in f:
  d=json.loads(line)
  if prev['at']<d['at']<=r['at']:j.append(d)
changes=[];checks={};gaps=[];actions=[];errors=[];unavailable=[];worker_errors={}
for n,d in r['nodes'].items():
 rows=d.get('workers',[]);old={x['gpu']:x for x in prev['nodes'][n].get('workers',[])};valid=[]
 for x in rows:
  o=old.get(x['gpu'],{})
  if o.get('pid')!=x.get('pid'):changes.append(dict(node=n,gpu=x['gpu'],before=o.get('pid'),after=x.get('pid')))
  elif x.get('matching_heartbeat') and o.get('matching_heartbeat'):valid.append(x['completed']-o['completed'])
 utils=[x['gpu_util'] for s in j for x in s['nodes'].get(n,{}).get('workers',[]) if x.get('gpu_util') is not None]
 usage=(d.get('cpu_usage_usec',0)-prev['nodes'][n].get('cpu_usage_usec',0))/1e6/(r['at']-prev['at']) if a.slot else None
 checks[n]=dict(alive=sum(x.get('alive',False) for x in rows),matching_heartbeat=sum(x.get('matching_heartbeat',False) for x in rows),max_progress_age_s=max((x.get('progress_age_s',0) for x in rows),default=None),errors=sorted({x['last_error'] for x in rows if x.get('last_error')}),mean_gpu_util_in_window_samples=statistics.mean(utils) if utils else None,mean_cpu_cores_in_window=usage,normal_completed_delta_same_pid=sum(valid),valid_counter_pairs=len(valid))
for s in j:
 for n,d in s['nodes'].items():
  if d.get('actions'):actions.append(dict(at=s['at'],node=n,actions=d['actions']))
  if d.get('observation_error'):errors.append(dict(at=s['at'],node=n,error=d['observation_error']))
  for x in d.get('workers',[]):
   if not x.get('matching_heartbeat'):gaps.append(dict(at=s['at'],node=n,gpu=x['gpu'],pid=x.get('pid'),alive=x.get('alive')))
   if not x.get('alive'):unavailable.append(dict(at=s['at'],node=n,gpu=x['gpu'],pid=x.get('pid'),stage=x.get('stage')))
   if x.get('last_error'):
    key=(n,x['gpu'],x.get('pid'),x['last_error'])
    z=worker_errors.setdefault(key,dict(node=n,gpu=x['gpu'],pid=x.get('pid'),error=x['last_error'],first_at=s['at'],last_at=s['at'],samples=0));z['last_at']=s['at'];z['samples']+=1
s=read(w/'state.json');stat=(Path('/proc')/str(s['pid'])/'stat').read_text().split(') ')[1].split();live=stat[0]!='Z' and stat[19]==s['start_ticks'] and time.time()-s['heartbeat']<180
out=dict(reviewed_at=time.time(),slot=a.slot,report_sha256=hashlib.sha256((w/f'report-{a.slot:02d}.json').read_bytes()).hexdigest(),checks=checks,worker_changes=changes,resolved_clips=r['resolved_clips'],window_resolved_clips=r['resolved_clips']-prev['resolved_clips'],window_clips_per_second=(r['resolved_clips']-prev['resolved_clips'])/(r['at']-prev['at']) if a.slot else None,journal_sample_count=len(j),max_poll_gap_s=max([b['at']-aa['at'] for aa,b in zip(j,j[1:])] or [0]),journal_recovery_actions=actions,journal_observation_errors=errors,heartbeat_observation_gaps=gaps,guardian_pid_identity_verified=live,failed_sources=r.get('failed_sources'),goal_window_complete=False)
out['journal_unavailable_workers']=unavailable
out['journal_worker_errors']=list(worker_errors.values())
print(json.dumps(out,ensure_ascii=False,indent=2))
