"""Half-hour read-only runtime audit; remediation remains with the active goal."""
import concurrent.futures,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
from listen_cluster import transport
from run_glare_cache_cluster import status
CONFIG=ROOT/'experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json'
cfg=c.config(CONFIG);m=c.load_run(cfg);work=Path(cfg['work']);out=work/'monitoring';out.mkdir(exist_ok=True)

def probe(node,state):
    ids=[state.get('pid')]+[v['pid'] for v in state.get('workers',{}).values()];ids=[p for p in ids if p]
    code='''import os,json,subprocess
ids=IDS
r={'processes':{p:os.path.exists('/proc/%d'%p) for p in ids},'load':os.getloadavg()}
p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
r['gpu']=p.stdout;r['gpu_query_returncode']=p.returncode
print(json.dumps(r))'''.replace('IDS',repr(ids))
    try:
        p=subprocess.run(transport(c.read(cfg['ssh_config']),node,[cfg['python'],'-c',code]),capture_output=True,text=True,timeout=35)
        return json.loads(p.stdout) if p.returncode==0 else {'observation_error':p.stderr[-1000:]}
    except Exception:return {'observation_error':traceback.format_exc()[-1000:]}

def tick():
    s=status(cfg,m);s['monitor_pid']=os.getpid();s['alerts']=[]
    for stage,q in s['queue'].items():
        if q['failed_tasks']:s['alerts'].append(stage+': quarantined failed tasks '+str(q['failed_task_ids']))
    for name,n in s['nodes'].items():
        if n.get('stage')=='running' and time.time()-n.get('heartbeat',0)>90:s['alerts'].append(name+': stale supervisor heartbeat; verify live processes')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        probes=list(pool.map(lambda n:probe(n,s['nodes'][n['name']]),cfg['nodes']))
    for n,p in zip(cfg['nodes'],probes):
        s['nodes'][n['name']]['live_probe']=p
        if p.get('observation_error'):s['alerts'].append(n['name']+': observation failed; recheck before recovery')
        elif not all(p['processes'].values()):s['alerts'].append(n['name']+': recorded process missing; inspect role and current replacement')
    totals={stage:{'new':0,'reused':0} for stage in c.STAGES}
    finished={stage:set() for stage in c.STAGES}
    for stage in c.STAGES:
        for p in (work/'tasks'/stage).glob('*/done.json'):
            r=c.read(p);finished[stage].add(r['task']['task_id']);native=r.get('result',{}).get('native_state',r.get('native_state',{}))
            totals[stage]['new']+=native.get('new' if stage=='prosody' else 'completed',0);totals[stage]['reused']+=native.get('reused',0)
    for n in s['nodes'].values():
        for slot in n.get('slots',{}).values():
            active=slot['task'];stage=active['stage'];state=slot['state']
            if active['task']['task_id'] in finished[stage]:continue
            totals[stage]['new']+=state.get('new' if stage=='prosody' else 'completed',0);totals[stage]['reused']+=state.get('reused',0)
    s['totals']=totals
    previous=out/'PERIODIC_LATEST.json'
    if previous.exists():
        old=c.read(previous);dt=s['at']-old['at'];s['new_per_second']={k:(totals[k]['new']-old['totals'][k]['new'])/dt for k in totals}
        for k,v in s['new_per_second'].items():
            prior=old.get('new_per_second',{}).get(k)
            if prior and v<prior*.7 and not s['queue'][k]['complete']:s['alerts'].append(k+': throughput fell >30%; inspect phase allocation, reuse and tails')
    c.save(out/(str(int(s['at']))+'.periodic.json'),s);c.save(previous,s)
    print(json.dumps({'at':s['at'],'totals':totals,'alerts':s['alerts']}),flush=True)
    return s

with c.locked(out/'periodic.lock'):
    while True:
        try:
            s=tick()
            if s['stopped'] or (s['complete'] and c.read(work/'finalize_status.json')['stage']=='complete'):break
        except Exception:
            c.save(out/'MONITOR_ERROR.json',{'at':time.time(),'error':traceback.format_exc()});print(traceback.format_exc(),flush=True)
        time.sleep(1800)
