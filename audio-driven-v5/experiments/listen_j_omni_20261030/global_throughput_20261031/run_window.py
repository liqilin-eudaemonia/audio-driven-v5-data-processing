import sys,json,concurrent.futures,statistics,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent
sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote
label=sys.argv[1];duration=int(sys.argv[2]);assert 60<=duration<=600 and duration%5==0
dest=D/label;dest.mkdir(exist_ok=True)
def pipeline():
    out=dict(at=time.time())
    for name,path in [('quality','visual_quality/queues/measure/state.json'),('caption','queues/caption/state.json')]:
        s=json.loads((W/path).read_text())
        out[name]={k:(len(v) if isinstance(v,(dict,list)) else v) for k,v in s.items()}
    return out
(dest/'pipeline_before.json').write_text(json.dumps(pipeline(),indent=2))
(dest/'queue_before.json').write_bytes((W/'queues/caption/state.json').read_bytes())
def run(node):
    args=['python3',str(D/'observe.py'),node,label,str(duration)]
    code=f'import subprocess,json;p=subprocess.run({args!r},capture_output=True,text=True);print(json.dumps(dict(code=p.returncode,out=p.stdout,err=p.stderr[-2000:])))'
    out=remote(node,code,timeout=duration+45)
    if out['code']:raise RuntimeError(node+': '+out['err'])
    return json.loads((D/label/f'{node}.json').read_text())
with concurrent.futures.ThreadPoolExecutor(4) as p:
    futures={n:p.submit(run,n) for n in ['node17','node8','node4','node12']}
    rows=[f.result() for f in futures.values()]
(dest/'queue_after.json').write_bytes((W/'queues/caption/state.json').read_bytes())
(dest/'pipeline_after.json').write_text(json.dumps(pipeline(),indent=2))
assert all(r['valid_no_restarts'] and not r['errors'] for r in rows)
mix=[x for r in rows for x in r['sampled_outputs']]
out=dict(label=label,at=time.time(),slots=rows[0]['slots'],clips=sum(r['clips'] for r in rows),
    clips_per_second=sum(r['clips_per_second'] for r in rows),gpu_util=statistics.mean(r['gpu_util'] for r in rows),
    model_failures=sum(r['model_failures'] for r in rows),sampled_count=len(mix),
    sample_duration_mean=statistics.mean(x['duration'] for x in mix) if mix else None,
    sample_duration_median=statistics.median(x['duration'] for x in mix) if mix else None,
    sample_changed_crop_fraction=statistics.mean(bool(x['crop_changed']) for x in mix) if mix else None,
    nodes=[{k:r[k] for k in ['node','clips','clips_per_second','cpu_cores','gpu_util','model_failures']} for r in rows])
(D/label/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
