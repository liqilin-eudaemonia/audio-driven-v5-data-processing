"""Capacity-gated, resumable dual200 execution; never stops other sessions' processes."""
import subprocess,time,os,fcntl,sys,traceback
from runtime import *
from report import report,METHODS
PY=ROOT/'compare/.envs/vico_more/bin/python';AF=ROOT/'compare/.envs/avatarforcing/bin/python';L2L=ROOT/'compare/.envs/l2l/bin/python';METRIC='/private/software/conda/envs/AnyTalker/bin/python'
RETIRED_STAGES={'gt_gpu4':'Historical GPU7-only DECA guard; replaced by gt_gpu4_v2', 'features_0':'ASCII text decoding failure; replaced by features_utf8_0', 'features_1':'ASCII text decoding failure; replaced by features_utf8_1', 'gt_gpu4_v2':'p2_listen script/package collision; replaced by gt_gpu4_v3', 'merge_listenformer_seamless':'Reader version changed across shards; shard00 rescored with current reader and merge retried'}

def alive(pid):
    try:
        value=Path(f'/proc/{pid}/stat').read_text();return value.split(') ')[1][0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def command_matches(pid):
    try:return b'p2_reference_methods_dualtest_20261112/' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def launch(tag,script,args,python=PY,gpu=7):
    d=WORK/'processes'/tag;d.mkdir(parents=True,exist_ok=True)
    attempt=len(list(d.glob('exit_*.json')));out=d/f'run_{attempt:02d}.log';exitfile=d/f'exit_{attempt:02d}.json'
    cmd=[str(python),str(WORK/script),*args]
    wrapper='import subprocess,json,os,time;from pathlib import Path;'+f'd=Path({str(d)!r});f=open({str(out)!r},"w");p=subprocess.Popen({cmd!r},cwd={str(ROOT)!r},stdout=f,stderr=subprocess.STDOUT);'+f'(d/"running.json").write_text(json.dumps(dict(pid=p.pid,started=time.time(),attempt={attempt},command={cmd!r})));c=p.wait();Path({str(exitfile)!r}).write_text(json.dumps(dict(returncode=c,ended=time.time())))'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    sup=subprocess.Popen([sys.executable,'-c',wrapper],cwd=ROOT,env=env,stdout=open(d/'supervisor.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
    save(d/'launch.json',dict(supervisor_pid=sup.pid,attempt=attempt,command=cmd,gpu=gpu,at=time.time()))
    return dict(tag=tag,kind=tag.split('_')[0],directory=str(d),supervisor_pid=sup.pid,attempt=attempt,gpu=gpu)

def refresh(active,failures):
    for tag,r in list(active.items()):
        d=Path(r['directory']);running=d/'running.json'
        if not running.exists():
            if alive(r.get('supervisor_pid',0)):continue
            active.pop(tag);failures[tag]=failures.get(tag,0)+1;continue
        rr=read(running)
        if not r.get('adopted') and rr.get('attempt')!=r['attempt']:
            if alive(r.get('supervisor_pid',0)):continue
            active.pop(tag);failures[tag]=failures.get(tag,0)+1;continue
        pid=rr['pid']
        if alive(pid) and command_matches(pid):continue
        expected=d/f"exit_{r['attempt']:02d}.json" if not r.get('adopted') else d/'exit.json'
        exits=[expected] if expected.exists() else []
        if not exits:continue
        last=read(exits[-1]);active.pop(tag)
        if last['returncode']!=0:failures[tag]=failures.get(tag,0)+1

def run():
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};failures={};started=time.time();lastlaunch=0
        statefile=WORK/'CONTROLLER_STATE.json'
        if statefile.exists():
            state=read(statefile);active=state['active'];failures=state['failures']
        # Adopt only known jobs explicitly launched by this task.
        jobs=ROOT/'experiments/p2_vico_additional_baselines_20261107/jobs'
        for suffix,tag in [('af_worker0','af_0'),('af_worker1','af_1'),('score_af_resp00','score_avatarforcing_responsenet_0')]:
            d=jobs/('reference_dualtest_20261112_'+suffix)
            if (d/'running.json').exists() and not (d/'exit.json').exists():active[tag]=dict(tag=tag,kind=tag.split('_')[0],directory=str(d),adopted=True)
        for dataset,spec in read(WORK/'protocol.json')['sources'].items():
            old=Path(spec['old_dystream'])/'RESULTS.json';assert sha(old)==spec['historical_results_sha256']
            r=read(old);r.update(reused_verified=True,input_verification=str(WORK/'DY_REUSE_COMPLETE.json'));save(WORK/'results'/f'dystream_{dataset}.json',r)
        ss=selected();assert len(ss)==400
        while True:
            refresh(active,failures)
            features=sum((WORK/'features'/s['dataset']/s['id']/'complete.json').exists() for s in ss)
            tracks=sum(any((WORK/'gpavatar_references'/s['dataset']/s['id']/name).exists() for name in ['complete.json','TRACK_UNAVAILABLE.json']) for s in ss)
            unavailable={m:0 for m in METHODS if m!='dualtalk'}
            for sample in ss:
                no_identity=WORK/'gpavatar_references'/sample['dataset']/sample['id']/'TRACK_UNAVAILABLE.json'
                if not no_identity.exists():continue
                for name in ['conan','fallon','stephen','trevor']:
                    method='l2l_'+name;d=folder(method,sample)
                    if (d/'MOTION_COMPLETE.json').exists() and not (d/'complete.json').exists():
                        publish(method,sample,[no_identity,d/'motion.npz'],score_path=None,status='unavailable',reason=read(no_identity)['reason'],official_tracker_no_identity=True)
                    if (d/'complete.json').exists():unavailable[method]+=1
            gt=sum((WORK/'evaluation_gt'/s['dataset']/s['id']/'complete.json').exists() for s in ss)
            motions={m:sum((folder(m,s)/'MOTION_COMPLETE.json').exists() for s in ss) for m in ['listenformer','dualtalk','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor']}
            videos={m:sum((folder(m,s)/'complete.json').exists() for s in ss) for m in METHODS if m!='dualtalk'}
            free_by_gpu={}
            for gpu,uuid in GPU_UUIDS.items():
                row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert row[0].strip()==uuid;free_by_gpu[gpu]=int(row[1])
            candidates=[]
            def offer(tag,script,args,python=PY,gib=4,gpu=7):
                if tag not in active and failures.get(tag,0)<3:candidates.append((tag,script,args,python,gib,gpu))
            # Extra GPU4 capacity prioritizes rendering and delayed evaluation work.
            if tracks and any(motions[m]>0 for m in motions if m.startswith('l2l_')) and min(videos[m] for m in videos if m.startswith('l2l_'))<400:
                for i in range(2,4):offer(f'gprender_{i}','gpavatar_render.py',['--wait'],gib=10,gpu=4)
            if gt<400 and motions['dualtalk']:offer('gt_gpu4_v3','gt_features.py',[],gib=12,gpu=4)
            # Prioritize supplying downstream work, with bounded concurrent processes.
            if features<400:
                for i in range(2):offer(f'features_utf8_{i}','feature_inputs.py',[],gib=4)
            if videos['avatarforcing']<400:
                for i in range(1 if any(motions[m]>0 for m in motions if m.startswith('l2l_')) and min(videos[m] for m in videos if m.startswith('l2l_'))<400 else 2):offer(f'af_{i}','video_generators.py',['avatarforcing'],AF,8)
            if features and motions['listenformer']<400:offer('lfmotion_0','motion_generators.py',['listenformer','--wait'],gib=2)
            if features and motions['dualtalk']<400:offer('dualmotion_0','motion_generators.py',['dualtalk','--wait'],gib=3)
            if features>=25 and min(motions[m] for m in motions if m.startswith('l2l_'))<400:offer('l2lmotion_0','l2l_motion.py',['--wait'],L2L,4)
            if motions['listenformer'] and videos['listenformer']<400:offer('lfrender_0','listenformer_render.py',['--wait'],gib=3)
            if tracks<400:offer('track_0','gpavatar_track.py',[],gib=9)
            if tracks and any(motions[m]>0 for m in motions if m.startswith('l2l_')):
                if min(videos[m] for m in videos if m.startswith('l2l_'))<400:
                    for i in range(2):offer(f'gprender_{i}','gpavatar_render.py',['--wait'],gib=7)
            # Existing GPU7 workers remain live; new bulk GT is assigned to GPU4.
            from scoring import samples,destination
            score_gpu=max(GPU_UUIDS,key=lambda gpu:(sum(r.get('gpu',7)==gpu for r in active.values())<(4 if gpu==4 else 10),free_by_gpu[gpu]))
            for method in videos:
                if method=='dystream':continue
                for dataset in ['responsenet','seamless']:
                    if (WORK/'results'/f'{method}_{dataset}.json').exists():continue
                    blocks=[(destination(method,dataset,i)/'SUCCESS.json').exists() for i in range(8)]
                    if all(blocks):offer('merge_'+method+'_'+dataset+('_reader_v2' if method=='listenformer' and dataset=='seamless' else ''),'scoring.py',['merge',method,dataset],METRIC,0)
                    elif sum(r['kind']=='score' for r in active.values())<2:
                        for i,done in enumerate(blocks):
                            if not done and all((folder(method,s)/'complete.json').exists() for s in samples(dataset,i)):
                                offer(f'score_{method}_{dataset}_{i}','scoring.py',['score',method,dataset,'--index',str(i)],METRIC,8,gpu=score_gpu)
            for dataset in ['responsenet','seamless']:
                group=[sample for sample in ss if sample['dataset']==dataset]
                native_methods=[m for m in motions if m!='listenformer']
                if all((WORK/'evaluation_gt'/dataset/sample['id']/'complete.json').exists() for sample in group) and all((folder(m,sample)/'MOTION_COMPLETE.json').exists() for m in native_methods for sample in group):
                    if not all((WORK/'results'/f'{m}_{dataset}_native.json').exists() for m in native_methods):offer('native_'+dataset,'native_score.py',['--dataset',dataset],PY,0)
            image_done=len(list((WORK/'results').glob('*.json')))
            completed=all((WORK/'results'/f'{m}_{d}.json').exists() for m in videos for d in ['responsenet','seamless']) and all((WORK/'results'/f'{m}_{d}_native.json').exists() for m in motions if m!='listenformer' for d in ['responsenet','seamless'])
            save(WORK/'status.json',dict(stage='complete' if completed else ('needs_attention' if any(n>=3 for tag,n in failures.items() if tag not in RETIRED_STAGES) else 'running'),scope='fixed200_each',inputs=400,features=features,gpavatar_references=sum((WORK/'gpavatar_references'/sample['dataset']/sample['id']/'complete.json').exists() for sample in ss),gpavatar_unavailable=sum((WORK/'gpavatar_references'/sample['dataset']/sample['id']/'TRACK_UNAVAILABLE.json').exists() for sample in ss),evaluation_gt=gt,motion=motions,video={m:n-unavailable[m] for m,n in videos.items()},video_unavailable=unavailable,result_files=image_done,active=list(active),failures=failures,retired_stages=RETIRED_STAGES,gpu_free_mib=free_by_gpu[7],gpu_free_mib_by_gpu=free_by_gpu,active_gpus={t:r.get('gpu',7) for t,r in active.items()},elapsed_s=time.time()-started,at=time.time()))
            save(statefile,dict(active=active,failures=failures,at=time.time()));report()
            if completed:
                files=[*list((WORK/'results').glob('*.json')),WORK/'RESULTS.md',WORK/'image_metrics.csv',WORK/'native_motion_metrics.csv',WORK/'RUN_SCOPE.json',WORK/'MODEL_MANIFEST.json']
                save(WORK/'COMPLETE.json',dict(passed=True,scope='fixed200_each',files={str(p):sha(p) for p in files},at=time.time()));return
            if time.time()-lastlaunch>=20:
                for tag,script,args,python,gib,gpu in candidates:
                    # Per-card capacity; existing GPU7 processes keep their assignment.
                    slots=sum(r.get('gpu',7)==gpu for r in active.values())
                    if (gib==0 or (slots<(4 if gpu==4 else 10) and free_by_gpu[gpu]>32768+gib*1024)):
                        active[tag]=launch(tag,script,args,python,gpu);lastlaunch=time.time()
                        save(statefile,dict(active=active,failures=failures,at=time.time()));break
            time.sleep(10)
if __name__=='__main__':run()
