"""ResponseNet586 queue: resource admission, per-clip resume and original forwards."""
from runtime import *
from report import report,METHODS,RGB,NATIVE
from progress_scan import ReceiptScan
from resource_control import defer_owned,refresh_idle_clocks,pause_owned,resume_owned
PY=ROOT/'compare/.envs/vico_more/bin/python';AF=ROOT/'compare/.envs/avatarforcing/bin/python';DY=ROOT/'compare/.envs/dystream/bin/python';L2L=ROOT/'compare/.envs/l2l/bin/python';METRIC='/private/software/conda/envs/AnyTalker/bin/python'

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False

def launch(tag,script,args,python,gpu,gib):
    d=WORK/'processes'/tag;d.mkdir(parents=True,exist_ok=True)
    attempt=len(list(d.glob('exit_*.json')));log=d/f'run_{attempt:03d}.log';exitfile=d/f'exit_{attempt:03d}.json';cmd=[str(python),str(WORK/script),*args]
    wrapper='import subprocess,json,time;from pathlib import Path;'+f'd=Path({str(d)!r});f=open({str(log)!r},"w");p=subprocess.Popen({cmd!r},cwd={str(ROOT)!r},stdout=f,stderr=subprocess.STDOUT);'+f'(d/"running.json").write_text(json.dumps(dict(pid=p.pid,attempt={attempt},started=time.time(),command={cmd!r},gpu={gpu})));c=p.wait();Path({str(exitfile)!r}).write_text(json.dumps(dict(returncode=c,ended=time.time())))'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',EVAL_MEMORY_FRACTION='.15' if script=='video_generators.py' and args[0]=='dystream' else '.10')
    sup=subprocess.Popen([sys.executable,'-c',wrapper],cwd=ROOT,env=env,stdout=open(d/'supervisor.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(tag=tag,kind=tag.split('_')[0],directory=str(d),supervisor_pid=sup.pid,attempt=attempt,gpu=gpu,estimate_gib=gib,at=time.time(),command=cmd)
    save(d/'launch.json',r);return r

def refresh(active,failures):
    for tag,r in list(active.items()):
        d=Path(r['directory']);exitfile=d/f"exit_{r['attempt']:03d}.json"
        if exitfile.exists():
            result=read(exitfile);active.pop(tag)
            if result['returncode']!=0:failures[tag]=failures.get(tag,0)+1
            continue
        if not (d/'running.json').exists():
            if not alive(r['supervisor_pid']):active.pop(tag);failures[tag]=failures.get(tag,0)+1
            continue
        rr=read(d/'running.json')
        if rr['attempt']!=r['attempt']:
            if not alive(r['supervisor_pid']):active.pop(tag);failures[tag]=failures.get(tag,0)+1
            continue
        if not alive(rr['pid']) and not alive(r['supervisor_pid']):
            save(exitfile,dict(returncode=-999,reason='Owned worker and supervisor disappeared',ended=time.time()))

def resource_rows():
    value=read(WORK/'MONITOR.json')
    assert time.time()-value['at']<90,'Resource monitor stale; no new admission'
    rows=value['gpus']
    for gpu,uuid in GPU_UUIDS.items():assert rows[str(gpu)]['uuid']==uuid
    return rows

def run():
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(WORK/'REUSE_COMPLETE.json')['passed']
        statefile=WORK/'CONTROLLER_STATE.json';state=read(statefile) if statefile.exists() else dict(active={},failures={})
        active=state['active'];failures=state['failures'];ss=selected();lastlaunch=0;started=time.time();scan=ReceiptScan(s['id'] for s in ss)
        last_run=state.get('last_run',{});leases=state.get('leases',{});cooldown=state.get('cooldown',{});empty_since=state.get('empty_since',{})
        from scoring import destination
        while True:
            refresh(active,failures)
            features=len(scan.ids(WORK/'features/responsenet'))
            gt=len(scan.ids(WORK/'evaluation_gt/responsenet'))
            tracked=len(scan.ids(WORK/'gpavatar_references/responsenet'))
            missing=scan.ids(WORK/'gpavatar_references/responsenet','TRACK_UNAVAILABLE.json')
            for s in ss:
                if s['id'] not in missing:continue
                marker=WORK/'gpavatar_references/responsenet'/s['id']/'TRACK_UNAVAILABLE.json'
                for name in ['conan','fallon','stephen','trevor']:
                    m='l2l_'+name;d=folder(m,s)
                    if (d/'MOTION_COMPLETE.json').exists() and not (d/'complete.json').exists():publish(m,s,[marker,d/'motion.npz'],score_path=None,status='unavailable',reason=read(marker)['reason'],official_tracker_no_identity=True)
            motion={m:len(scan.ids(WORK/m/'responsenet','MOTION_COMPLETE.json')) for m in ['listenformer',*NATIVE]}
            resolved_ids={m:scan.ids(WORK/m/'responsenet') for m in RGB}
            resolved={m:len(ids) for m,ids in resolved_ids.items()}
            unavailable={m:len(resolved_ids[m]&missing) if m.startswith('l2l_') else 0 for m in RGB}
            blocks={m:[(destination(m,i)/'SUCCESS.json').exists() for i in range(len(groups()))] for m in RGB}
            scored={m:sum(len(groups()[i]) for i,done in enumerate(v) if done) for m,v in blocks.items()}
            try:resources=resource_rows()
            except (AssertionError,FileNotFoundError,ValueError):resources={};print('Waiting for fresh Monitor resource snapshot',flush=True)
            policy=read(WORK/'RESOURCE_POLICY.json');now=time.time()
            for tag,r in list(active.items()):
                if r['gpu'] not in policy['allowed_gpus']:
                    result=defer_owned(r,'GPU excluded by latest user allocation');active.pop(tag)
                    if result['returncode']!=0 and not result.get('planned_defer'):failures[tag]=failures.get(tag,0)+1
            refresh_idle_clocks(active,leases,cooldown,empty_since,resources,policy,now)
            for gpu in GPU_UUIDS:
                same=[(tag,r) for tag,r in active.items() if r['gpu']==gpu and r['estimate_gib']>0]
                external=resources.get(str(gpu),{}).get('external',[])
                expired=policy.get('time_slice_enabled',False) and str(gpu) in leases and now-leases[str(gpu)]>=policy['lease_seconds']
                if external:
                    for tag,r in same:
                        if not r.get('paused_at'):pause_owned(r,'Foreground GPU task: retain weights and suspend compute')
                elif resources.get(str(gpu),{}).get('eligible_idle',False):
                    for tag,r in same:
                        if r.get('paused_at'):resume_owned(r)
            candidates=[]
            def offer(tag,script,args=[],python=PY,gib=8):
                if tag not in active and failures.get(tag,0)<3:candidates.append((tag,script,args,python,gib))
            # One lane of each ready stage precedes additional replicas.
            if features<TOTAL:offer('features_0','feature_inputs.py',[],gib=8)
            if resolved['avatarforcing']<TOTAL:offer('af_0','video_generators.py',['avatarforcing'],AF,12)
            if resolved['dystream']<TOTAL:offer('dy_0','video_generators.py',['dystream'],DY,22)
            if motion['listenformer']<TOTAL:offer('lfmotion_0','motion_generators.py',['listenformer','--wait'],gib=2)
            if motion['dualtalk']<TOTAL:offer('dualmotion_0','motion_generators.py',['dualtalk','--wait'],gib=5)
            if min(motion[m] for m in NATIVE if m.startswith('l2l_'))<TOTAL:offer('l2lmotion_0','l2l_motion.py',['--wait'],L2L,4)
            if resolved['listenformer']<TOTAL:offer('lfrender_0','listenformer_render.py',['--wait'],gib=12)
            if tracked+len(missing)<TOTAL:offer('track_0','gpavatar_track.py',[],gib=12)
            if min(resolved[m] for m in RGB if m.startswith('l2l_'))<min(tracked+len(missing),min(motion[m] for m in NATIVE if m.startswith('l2l_'))):offer('gprender_0','gpavatar_render.py',['--wait'],gib=9)
            if gt<TOTAL:offer('gt_0','gt_features.py',[],gib=12)
            if not all((WORK/'results'/f'{m}_responsenet.json').exists() for m in RGB):
                offer('score_service','scoring_service.py',[],METRIC,8)
            for m in RGB:
                if all(blocks[m]) and not (WORK/'results'/f'{m}_responsenet.json').exists():offer('merge_'+m,'scoring.py',['merge',m],METRIC,0)
            if gt==TOTAL and all(motion[m]==TOTAL for m in NATIVE):
                if not all((WORK/'results'/f'{m}_responsenet_native.json').exists() for m in NATIVE):offer('native_0','native_score.py',['--dataset','responsenet'],PY,0)
            complete=all((WORK/'results'/f'{m}_responsenet.json').exists() for m in RGB) and all((WORK/'results'/f'{m}_responsenet_native.json').exists() for m in NATIVE)
            save(WORK/'status.json',dict(stage='complete' if complete else ('needs_attention' if any(n>=3 for n in failures.values()) else ('running' if any(r['estimate_gib']>0 and not r.get('paused_at') for r in active.values()) else ('paused_for_foreground_models_retained' if any(r.get('paused_at') for r in active.values()) else 'waiting_for_resources'))),scope=SCOPE,inputs=TOTAL,features=features,evaluation_gt=gt,gpavatar_references=tracked,gpavatar_unavailable=len(missing),motion=motion,video={m:resolved[m]-unavailable[m] for m in RGB},video_unavailable=unavailable,scored=scored,active=list(active),active_gpus={t:r['gpu'] for t,r in active.items()},paused_workers=[t for t,r in active.items() if r.get('paused_at')],resident_mode=True,failures=failures,resources=resources,resource_policy=policy,cooldown_until=cooldown,elapsed_s=time.time()-started,at=time.time()))
            save(statefile,dict(active=active,failures=failures,last_run=last_run,leases=leases,cooldown=cooldown,empty_since=empty_since,at=time.time()));report()
            if complete and not active:
                files=[*list((WORK/'results').glob('*.json')),WORK/'RESULTS.md',WORK/'image_metrics.csv',WORK/'RUN_SCOPE.json',WORK/'MODEL_MANIFEST.json',WORK/'METRIC_CODE.json']
                save(WORK/'COMPLETE.json',dict(passed=True,scope=SCOPE,files={str(p):sha(p) for p in files},at=time.time()));return
            if resources and time.time()-lastlaunch>=10:
                for tag,script,args,python,gib in sorted(candidates,key=lambda c:(0 if c[4]==0 else 1,last_run.get(c[0].split('_')[0],0))):
                    if gib==0:
                        if any(r['kind'] in ['merge','native'] for r in active.values()):continue
                        gpu=7
                    else:
                        eligible=[]
                        for gpu in GPU_UUIDS:
                            if gpu not in policy['allowed_gpus']:continue
                            if not resources[str(gpu)].get('eligible_idle',False) or time.time()<cooldown.get(str(gpu),0):continue
                            same=[r for r in active.values() if r['gpu']==gpu and r['estimate_gib']>0]
                            startup=sum(r['estimate_gib']*1024 for r in same if time.time()-r['at']<60)
                            available=resources[str(gpu)]['free_mib']-startup
                            if len(same)<policy['max_workers_per_gpu'] and available>policy['reserve_mib'][str(gpu)]+gib*1024:eligible.append((available,gpu))
                        if not eligible:continue
                        gpu=max(eligible)[1]
                    active[tag]=launch(tag,script,args,python,gpu,gib);lastlaunch=time.time();last_run[tag.split('_')[0]]=lastlaunch
                    if gib>0 and str(gpu) not in leases:leases[str(gpu)]=lastlaunch
                    save(statefile,dict(active=active,failures=failures,last_run=last_run,leases=leases,cooldown=cooldown,empty_since=empty_since,at=time.time()));break
            time.sleep(5)
if __name__=='__main__':run()
