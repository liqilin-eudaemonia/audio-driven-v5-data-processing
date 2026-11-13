"""Hourly four-node audit, cheap liveness checks, scoped full-state recovery."""
from pathlib import Path
import concurrent.futures, csv, fcntl, importlib.util, json, math, os
import signal, statistics, subprocess, sys, threading, time, traceback

G = Path(__file__).resolve().parent
ROOT = G.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_listen.contracts import read, atomic_json, sha
from listen_cluster import transport

LEGACY = G.parent/'p2_ablation28_guard9h_20261111'
spec = importlib.util.spec_from_file_location('prior_guard_helpers', LEGACY/'watch.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
Q = M.Q
FULL = G.parent/'p2_full_resume5500_to7000_after_ablation_20261112'
PREVIOUS = G.parent/'p2_no_whisper_resume1760_20261112'
ABLATIONS = G.parent/'p2_all_lora_ablations28_20261111'
RELIABLE = G.parent/'p2_stage2_joint_full6000_trial28_20261112/reliable_entry.py'

# Healthy-node inspection needs only GPU processes and their ancestors. Scanning
# every /proc entry timed out on node4 although its direct GPU query took3.35s.
REMOTE_PROBE = M.PROBE.split('controllers=[]\n',1)[0] + r'''
controllers=[]
candidates=set(KNOWN_PIDS)
for worker in processes:
 if SCRIPT not in worker['argv'] or 'train' not in worker['argv']:continue
 pid=worker['pid']
 for depth in range(8):
  try:
   st=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
   pid=int(st[1])
  except (FileNotFoundError,ProcessLookupError,PermissionError):break
  if pid<=1 or pid in candidates:break
  candidates.add(pid)
for pid in candidates:
 p=pathlib.Path('/proc')/str(pid)
 try:
  raw=(p/'cmdline').read_bytes();a=raw.decode(errors='replace').split(chr(0))
  if SCRIPT in a and 'node'in a:
   st=(p/'stat').read_text().rsplit(')',1)[1].split();env=(p/'environ').read_bytes().split(bytes([0]));visible=next((v.split(b'=',1)[1].decode()for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
   controllers.append(dict(pid=pid,argv=a,start=st[19],visible=visible))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(dict(gpus=gpus,processes=processes,controllers=controllers)))
'''

def known_controllers(arm,node):
    for path in [G/'LATEST.json',*sorted(G.glob('hour*.json'),reverse=True)]:
        if not path.exists():continue
        value=read(path)
        if value['health']['arm']!=arm:continue
        row=next((n for n in value['cluster']['nodes'] if n['node']==node),None)
        if row and row.get('controllers'):return [p['pid'] for p in row['controllers']]
    return []

def phase():
    if (FULL/'COMPLETE.json').exists(): return 'complete'
    if (PREVIOUS/'COMPLETE.json').exists(): return 'full'
    return 'no_whisper'

def info(arm):
    if arm == 'no_whisper':
        return dict(controller=PREVIOUS, script=ABLATIONS/'runner.py',
                    entry=PREVIOUS/'launch.py', out=ABLATIONS/'no_whisper', target=6643, cadence=500)
    return dict(controller=FULL, script=FULL/'worker.py', entry=FULL/'continue_after.py',
                out=FULL/'train', target=7000, cadence=100)

def event(kind, **kw):
    value = dict(at=time.time(), kind=kind, **kw)
    with (G/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)

def owner(arm):
    d = info(arm)['controller']
    if not (d/'CONTROLLER.json').exists(): return None
    try:
        value=M.record(read(d/'CONTROLLER.json')['pid'])
        return value if str(info(arm)['entry']) in value['cmd'].split() else None
    except (FileNotFoundError, ProcessLookupError): return None

def live(value): return value is not None and M.alive(value)

def owned(p, arm, mode):
    a = p['argv']; d = info(arm)
    if str(d['script']) not in a or mode not in a: return False
    return arm == 'full' or ('--arm' in a and a[a.index('--arm')+1] == 'no_whisper')

def probe(arm):
    def one(n):
        error = None
        for attempt in range(3):
            try:
                code = REMOTE_PROBE.replace('SCRIPT', repr(str(info(arm)['script']))).replace('KNOWN_PIDS',repr(known_controllers(arm,n['name'])))
                p = subprocess.run(transport(Q.CL,n,[Q.PY,'-c',code]),capture_output=True,text=True,timeout=45)
                if p.returncode: raise RuntimeError(p.stderr[-700:])
                v = json.loads(p.stdout)
                cards = [g for g in v['gpus'] if g['index'] in n['gpus']]
                uuids = {g['uuid'] for g in cards}
                procs = [p for p in v['processes'] if p['uuid'] in uuids]
                workers = [p for p in procs if owned(p,arm,'train')]
                return dict(node=n['name'],cards=cards,workers=workers,
                            controllers=[p for p in v['controllers'] if owned(p,arm,'node')],
                            foreign=[p for p in procs if not owned(p,arm,'train')])
            except Exception as e:
                error=f'SSH/GPU probe timed out ({e.timeout}s)' if isinstance(e,subprocess.TimeoutExpired) else repr(e)
                time.sleep(2)
        return dict(node=n['name'],error=error)
    with concurrent.futures.ThreadPoolExecutor(4) as e: rows=list(e.map(one,Q.CL['nodes']))
    observed=all('error' not in n and len(n['cards'])==len(Q.CL['nodes'][i]['gpus']) for i,n in enumerate(rows))
    healthy=observed and all(len(n['workers'])==len(n['cards']) and
              len({p['uuid'] for p in n['workers']})==len(n['cards']) for n in rows)
    return dict(at=time.time(),nodes=rows,observed=observed,healthy28=healthy)

def rows(path, count=600):
    if not path.exists(): return []
    values=[]
    for line in path.read_text().splitlines():
        try: values.append(json.loads(line))
        except json.JSONDecodeError: pass
    return values[-count:]

SCHEDULES={}
BASELINES={}
def speed(arm, values):
    if not values: return dict(enough=False)
    if arm not in SCHEDULES:
        c=read(info(arm)['out']/'config.json')
        if arm=='no_whisper': schedule=Q.R.Schedule(c)
        else:
            spec=importlib.util.spec_from_file_location('full_guard_runtime',FULL/'run_runtime.py')
            r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);schedule=r.Schedule(c)
        with Path(c['sample_list']).open() as f: duration=[float(r['duration_s']) for r in csv.DictReader(f)]
        SCHEDULES[arm]=(schedule,duration)
    schedule,duration=SCHEDULES[arm]; normalized=[]
    for r in values:
        if arm=='full': a,b=schedule.bounds(r['step'])
        else: a,b=(r['step']-1)*28,r['step']*28
        pairs=schedule.pairs[a:b]
        normalized.append(r['seconds']/statistics.mean(duration[i] for i,_ in pairs))
    if arm not in BASELINES and len(normalized)>=100:
        BASELINES[arm]=statistics.median(normalized[:-50])
        atomic_json(G/'SPEED_BASELINES.json',BASELINES)
    baseline=BASELINES.get(arm)
    recent=statistics.median(normalized[-50:])
    return dict(enough=baseline is not None,normalized_ratio=recent/baseline if baseline else None,
                baseline=baseline,recent=recent,mean50_seconds=statistics.mean(r['seconds'] for r in values[-50:]))

def health(arm):
    d=info(arm);out=d['out'];path=out/'metrics.jsonl'
    values=rows(path);status=read(out/'status.json') if (out/'status.json').exists() else dict(step=5500)
    age=time.time()-path.stat().st_mtime if path.exists() else None
    own=owner(arm);is_live=live(own);step=status['step']
    saving=step%d['cadence']==0 or step==d['target']
    limit=1200 if saving else 600
    flags=[]
    if age is not None and age>limit: flags.append('stalled')
    if not is_live: flags.append('controller_missing')
    if any(not math.isfinite(r[k]) for r in values[-10:] for k in ['av','event','feature_video','feature_audio','weighted_total']): flags.append('nonfinite')
    startup=arm=='full' and not values
    if startup and is_live:
        # Full-state SHA/loading and longest-canary need several minutes.
        state=read(FULL/'STATE.json')
        handoff=read(PREVIOUS/'COMPLETE.json') if (PREVIOUS/'COMPLETE.json').exists() else {}
        age=time.time()-max(state['at'],handoff.get('at',0))
        if age<1500: flags=[f for f in flags if f!='stalled']
        elif 'stalled' not in flags:flags.append('stalled')
    return dict(at=time.time(),arm=arm,step=step,target=d['target'],metrics_age=age,
                saving=saving,startup=startup,flags=flags,owner=own,controller_live=is_live,
                speed=speed(arm,values),last_rows=values[-10:])

def contract(arm):
    d=info(arm);out=d['out'];c=read(out/'config.json');i=read(out/'run.json')
    assert c==i['experiment'] and i['world']==28 and c['training']['global_batch']==28
    assert c['training']['save_every']==d['cadence']
    assert c['loss_weights']==dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
    assert c['qwen_lora']['enabled'] is False
    if arm=='no_whisper': current=Q.R.source()
    else:
        spec=importlib.util.spec_from_file_location('full_contract_runtime',FULL/'run_runtime.py')
        r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);current=r.source()
        assert c['whisper']['enable_video'] and c['whisper']['enable_audio']
    assert current==i['source'] and all(sha(p)==h for p,h in current.items())
    return dict(success=True,source_files=len(current),config_sha256=sha(out/'config.json'))

def terminate_owned(arm,snap,own):
    assert snap['observed']
    if live(own):
        assert str(info(arm)['entry']) in own['cmd']
        os.kill(own['pid'],signal.SIGTERM)
        for _ in range(20):
            if not live(own): break
            time.sleep(1)
    snap=probe(arm)
    if not snap['observed']: raise RuntimeError('Cannot verify remote exits; do not duplicate launch')
    for n,row in zip(Q.CL['nodes'],snap['nodes']):
        for p in row['workers']+row['controllers']:
            assert p['visible']==','.join(map(str,n['gpus']))
            code=("import pathlib,os,signal;p=pathlib.Path('/proc')/str("+str(p['pid'])+
                  ");a=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0));"
                  "s=(p/'stat').read_text().rsplit(')',1)[1].split()[19];assert s=="+repr(p['start'])+
                  " and "+repr(str(info(arm)['script']))+" in a and ('train'in a or 'node'in a);"
                  "os.kill("+str(p['pid'])+",signal.SIGTERM)")
            subprocess.run(transport(Q.CL,n,[Q.PY,'-c',code]),capture_output=True,text=True,timeout=40,check=True)
    time.sleep(3)
    after=probe(arm)
    if not after['observed'] or any(n['workers'] or n['controllers'] or n['foreign'] for n in after['nodes']):
        raise RuntimeError('Compute still present; preserve other tasks, no duplicate launch')

def restart(arm,snap,h):
    d=info(arm)
    if (d['out']/'STOP').exists() or (d['controller']/'STOP').exists() or (ABLATIONS/'STOP').exists():
        raise RuntimeError('User STOP preserved; automatic recovery prohibited')
    contract(arm)
    terminate_owned(arm,snap,h['owner'])
    for p in [d['controller']/'FAILURE.json', ABLATIONS/'FAILURE.json' if arm=='no_whisper' else FULL/'FAILURE.json']:
        if p.exists(): p.rename(G/(p.parent.name+'_failure_'+str(time.time_ns())+'.json'))
    with (G/(arm+'_recovery.log')).open('a') as f:
        p=subprocess.Popen([Q.PY,'-u',str(RELIABLE),str(d['entry'])],stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                           env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    event('recovery_started',arm=arm,controller=M.record(p.pid),policy='latest complete native model/Adam/rank RNG')

def ensure_waiter():
    if (FULL/'STOP').exists() or (FULL/'train/STOP').exists():return
    if live(owner('full')) or (FULL/'COMPLETE.json').exists():return
    # Waiting controller may have exited when preceding run failed; restore it only
    # after that run is live again or has completed. No selected GPU is touched.
    if phase()=='no_whisper' and not live(owner('no_whisper')): return
    p=FULL/'FAILURE.json'
    if p.exists():p.rename(G/('waiter_failure_'+str(time.time_ns())+'.json'))
    with (G/'waiter_recovery.log').open('a') as f:
        child=subprocess.Popen([Q.PY,'-u',str(RELIABLE),str(FULL/'continue_after.py')],
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    event('handoff_waiter_restarted',pid=child.pid)

def run():
    protocol=read(G/'protocol.json');hour=0;pending=None;last_repair=0
    while (G/f'hour{hour:02d}.json').exists():hour+=1
    if (G/'SPEED_BASELINES.json').exists(): BASELINES.update(read(G/'SPEED_BASELINES.json'))
    with (G/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(G/'CONTROLLER.json',M.record(os.getpid()))
        while True:
            now=time.time();arm=phase();full=now>=protocol['start']+hour*3600 or now>=protocol['deadline']
            try:
                if arm=='complete':
                    h=dict(at=now,arm=arm,step=7000,target=7000,controller_live=False,flags=[],speed={})
                    snap=probe('full') if full else None
                    if snap is not None:
                        assert snap['observed'] and not any(n['workers'] or n['controllers'] for n in snap['nodes'])
                        report=dict(health=h,cluster=snap,requested_training_complete=read(FULL/'COMPLETE.json'))
                else:
                    if arm=='no_whisper':ensure_waiter()
                    h=health(arm)
                    ratio=h['speed'].get('normalized_ratio');slow=ratio is not None and ratio>1.3
                    urgent=bool(h['flags']) or slow
                    snap=probe(arm) if full or urgent else None
                    if snap is not None:
                        report=dict(health=h,cluster=snap,contract=contract(arm))
                        atomic_json(G/'LATEST.json',report)
                        event('audit',arm=arm,step=h['step'],observed=snap['observed'],healthy28=snap['healthy28'],flags=h['flags'],speed=h['speed'])
                        if snap['observed']:
                            failure=bool(h['flags'])
                            if failure and now-last_repair>900:
                                if pending and pending['arm']==arm and now-pending['at']>=55:
                                    restart(arm,snap,h);last_repair=time.time();pending=None
                                else: pending=dict(arm=arm,at=now)
                            else:pending=None
                            if slow:
                                atomic_json(G/'SLOWDOWN_ALERT.json',report)
                                event('slowdown_diagnostic',arm=arm,step=h['step'],normalized_ratio=ratio,
                                      foreign=[dict(node=n['node'],processes=n['foreign']) for n in snap['nodes']])
                atomic_json(G/'HEARTBEAT.json',dict(at=time.time(),arm=arm,step=h['step'],deadline=protocol['deadline'],watcher_live=True))
                if full:
                    if snap is None or not snap['observed']: raise RuntimeError('Hourly observation incomplete; retry same hour')
                    atomic_json(G/f'hour{hour:02d}.json',report);hour+=1
                if now>=protocol['deadline']:
                    atomic_json(G/'WINDOW_FINISHED.json',dict(success=True,at=time.time(),elapsed=time.time()-protocol['start'],hourly_reports=hour,training_state=arm))
                    return
            except Exception as error:
                event('inspection_error',error=repr(error),traceback=traceback.format_exc())
                atomic_json(G/'ALERT.json',dict(at=time.time(),error=repr(error)))
            time.sleep(min(60,max(1,protocol['deadline']-time.time())))

if __name__=='__main__':run()
