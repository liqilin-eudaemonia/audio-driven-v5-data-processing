"""Bounded read-only observation, fixed worker files; no result tree scan."""
import json,sys,time,subprocess,statistics
from pathlib import Path
w=Path(__file__).resolve().parent.parent;n=sys.argv[1];cfg=json.loads((w/'config.json').read_text());assert __import__('socket').gethostname()==cfg['nodes'][n]['hostname']
def cpu():
 d=dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines());return int(d['usage_usec'])
def workers():
 return {g:json.loads((w/'workers'/f'{n}-gpu-{g}.json').read_text()) for g in cfg['nodes'][n]['gpu_ids']}
def nod_count():
 old=cfg['nodes'][n].get('old_work')
 return json.loads((Path(old)/'queues/nod/state.json').read_text())['completed_clips'] if old else None
def quality():
 p=w/'visual_quality/nodes'/n/'controller.json'
 return json.loads(p.read_text()) if p.exists() else None

duration=int(sys.argv[3]) if len(sys.argv)>3 else 60
assert 10<=duration<=600 and duration%2==0
t0=time.time();c0=cpu();before=workers();nod_before=nod_count();quality_before=quality();samples=[]
for i in range(duration//2):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10)
 devices={int(z[0]):[float(v) for v in z[1:]] for line in raw.strip().splitlines() if (z:=line.split(','))}
 hs=workers();samples.append(dict(at=time.time(),devices=devices,stages={g:h['stage'] for g,h in hs.items()}))
 time.sleep(max(0,t0+(i+1)*2-time.time()))
after=workers();elapsed=time.time()-t0
counts={g:after[g]['completed_clips']-before[g]['completed_clips'] for g in before};assert all(before[g]['pid']==after[g]['pid'] for g in before)
report=dict(node=n,start=t0,end=time.time(),elapsed_s=elapsed,clips=sum(counts.values()),clips_per_second=sum(counts.values())/elapsed,per_gpu_clips=counts,cpu_cores=(cpu()-c0)/1e6/elapsed,cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),mean_gpu_util=statistics.mean(v[1] for s in samples for v in s['devices'].values()),per_gpu_util={g:statistics.mean(s['devices'][g][1] for s in samples) for g in before},mean_memory_mib=statistics.mean(v[0] for s in samples for v in s['devices'].values()),stage_counts={stage:sum(x==stage for s in samples for x in s['stages'].values()) for stage in ('inference','preparing','waiting')},samples=samples)
quality_after=quality()
if quality_before and quality_after and quality_before['pid']==quality_after['pid']:
 report.update(quality_workers=quality_after['workers'],quality_completed_clips=sum(quality_after['decisions_this_process'].values())-sum(quality_before['decisions_this_process'].values()))
 report['quality_clips_per_second']=report['quality_completed_clips']/elapsed
report['preparation_wait_seconds_per_gpu_second']=sum(after[g].get('preparation_wait_s',0)-before[g].get('preparation_wait_s',0) for g in before)/(elapsed*len(before))
nod_after=nod_count()
if nod_before is not None:report.update(nod_start=nod_before,nod_end=nod_after,nod_clips_per_second=(nod_after-nod_before)/elapsed)
report['preparation_wait_metric_note']='Sum of source-stream preparation waits. Adaptive reader waits overlap GPU inference; this is not exclusive GPU idle time and is not directly comparable across modes.'
report['execution_modes']={g:dict(request_limit=before[g].get('production_batch_size',1),adaptive=before[g].get('adaptive_request_groups',False),revision=before[g].get('execution_revision')) for g in before}
report['by_submission_mode']={}
for mode in ['single','adaptive4']:
 gs=[g for g in before if ('adaptive4' if before[g].get('production_batch_size')==4 and before[g].get('adaptive_request_groups') else 'single')==mode]
 if gs:report['by_submission_mode'][mode]=dict(gpus=gs,clips=sum(counts[g] for g in gs),clips_per_gpu_second=sum(counts[g] for g in gs)/elapsed/len(gs),mean_gpu_util=statistics.mean(s['devices'][g][1] for s in samples for g in gs),mean_preparation_wait_fraction=sum(after[g].get('preparation_wait_s',0)-before[g].get('preparation_wait_s',0) for g in gs)/elapsed/len(gs))
destination=Path(sys.argv[2]) if len(sys.argv)>2 else w/'performance_check'
destination.mkdir(parents=True,exist_ok=True)
(destination/f'{n}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
