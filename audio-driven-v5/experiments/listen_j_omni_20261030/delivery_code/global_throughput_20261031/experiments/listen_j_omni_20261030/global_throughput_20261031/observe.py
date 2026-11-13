"""Whole-host production observation; fixed state paths, no directory traversal."""
import json,sys,time,subprocess,socket,statistics,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent.parent
node,label,duration=sys.argv[1],sys.argv[2],int(sys.argv[3])
cfg=json.loads((W/'config.json').read_text());assert socket.gethostname()==cfg['nodes'][node]['hostname']
def states():return {g:json.loads((W/'workers'/f'{node}-gpu-{g}.json').read_text()) for g in cfg['nodes'][node]['gpu_ids']}
def cpu():return int(dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
start=time.time();before=states();c0=cpu();samples=[];pending={};observed={}
for step in range(duration//5):
    s=states()
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    devices={int(parts[0]):list(map(float,parts[1:])) for line in raw.splitlines() if (parts:=line.split(','))}
    samples.append(dict(at=time.time(),gpu=devices,counts={g:x['completed_clips'] for g,x in s.items()}))
    for x in s.values():
        sid=x.get('sample_id')
        if sid and sid not in observed:pending.setdefault(sid,time.time())
    for sid,seen_at in list(pending.items()):
        path=W/'caption/results'/hashlib.sha256(sid.encode()).hexdigest()[:2]/f'{sid}.json'
        if path.exists():
            r=json.loads(path.read_text());pending.pop(sid);observed[sid]=None
            if r['completed_at']>=start:
                observed[sid]=dict(sample_id=sid,duration=r['source']['duration_s'],
                    prepared_seconds=r['preparation_metrics']['seconds'],shared=r['preparation_metrics']['shared_decode_used'],
                    nod_reuse=r['input_evidence']['nod_supplement'].get('crop_reuse'),
                    crop_changed=r.get('crop_provenance',{}).get('coordinates_changed'),
                    generation_passed=r['generation_checks_passed'])
        elif time.time()-seen_at>60:pending.pop(sid)
    time.sleep(max(0,start+(step+1)*5-time.time()))
after=states();elapsed=time.time()-start;valid=all(before[g]['pid']==after[g]['pid'] for g in before)
counts={g:after[g]['completed_clips']-before[g]['completed_clips'] for g in before};mix=[r for r in observed.values() if r]
out=dict(node=node,label=label,start=start,end=time.time(),elapsed=elapsed,valid_no_restarts=valid,
    clips=sum(counts.values()),clips_per_second=sum(counts.values())/elapsed,per_gpu_clips=counts,
    cpu_cores=(cpu()-c0)/1e6/elapsed,gpu_util=statistics.mean(v[0] for x in samples for v in x['gpu'].values()),
    model_failures=sum(after[g]['failed_clips']-before[g]['failed_clips'] for g in before),
    errors={g:x['last_error'] for g,x in after.items() if x.get('last_error')},
    sampled_outputs=mix,samples=samples,slots=cfg['bounded_crop_input']['nod_slots_per_host'])
dest=Path(__file__).resolve().parent/label;dest.mkdir(exist_ok=True);(dest/f'{node}.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['samples','sampled_outputs']}))
