"""Eight matched Event evaluations; safely hand off the training allocation."""
from pathlib import Path
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time,traceback
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
TRAIN=ROOT/'experiments/p2_loss_search28_20261110'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse'),str(LEGACY),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON,METRIC_PYTHON,partition
NAMES=('baseline','event001','event020','event100');SEEDS=(137,238)

def folder(name,seed):return WORK/'evaluations'/f'{name}_500_seed{seed}'

def prepare():
    import eval_runtime as E
    from p2_combined_whisper_eval import link
    cluster=read(TRAIN/'runs/baseline/cluster.json')
    for name in NAMES:
        first=folder(name,137);first.mkdir(parents=True,exist_ok=True)
        cp=TRAIN/'runs'/name/'step_000500.pt'
        receipt=read(cp.parent/'STAGE_500.json');assert receipt['sha256']==read(cp.with_suffix('.complete.json'))['sha256']
        old=LEGACY/'evaluations'/f'{name}_500'
        if(old/'export/RECEIPT.json').exists():link(old/'export',first/'export')
        save(first/'REQUEST.json',dict(checkpoint=str(cp),samples=64),immutable=True)
        print('Preparing',name,flush=True);p=E.prepare(first)
        for seed in SEEDS:
            out=folder(name,seed);out.mkdir(parents=True,exist_ok=True)
            if seed!=137:
                for directory in ['export','conditions','whisper','negative']:link(first/directory,out/directory)
                for f in ['REQUEST.json','DATA_CONFIG.json','shared_items.json']:save(out/f,read(first/f),immutable=True)
            q=copy.deepcopy(p);q.pop('scoring_device',None)
            q.update(seed=seed,workers=28,cluster=str(out/'cluster.json'),metrics_cluster=str(out/'cluster.json'),
                worker_cli=str(LEGACY/'evaluate28.py'),shared_items_manifest=str(out/'shared_items.json'),
                authorized_scope='node12/node8/node4 GPU0-7 and node11 GPU0-3; training must be fully stopped')
            save(out/'cluster.json',cluster)
            for i,items in enumerate(partition(read(first/'shared_items.json'),28)):save(out/'shards'/f'{i:02d}.json',items)
            save(out/'protocol.json',q)
            assert sha(out/'shared_items.json')==q['shared_items_sha256']
        # Ready media remain in place; the old worker is stopped before using the new claims.
        if(old/'latents').exists():link(old/'latents',first/'latents')
        if(old/'generated').exists():link(old/'generated',first/'generated')
    source={str(p):sha(p)for p in [Path(__file__),WORK/'parallel_events.py',LEGACY/'dual_seed_compare.py']}
    source.update(E.code_identity())
    save(WORK/'PLAN.json',dict(candidates=list(NAMES),seeds=list(SEEDS),samples=64,outputs=512,
        membership_sha256=sha(LEGACY/'tuning64.json'),source=source,cluster=cluster,
        automatic_training_resume=False,at=time.time()))
    save(WORK/'PREPARED.json',dict(success=True,at=time.time(),source=source,outputs=512))

def alive(d):
    p=Path('/proc')/str(d['pid'])/'stat'
    if not p.exists():return False
    s=p.read_text().rsplit(')',1)[-1].split()
    return s[0]not in('Z','X')and s[19]==str(d['start_ticks'])

def stop_old_evaluation():
    # Waiting controller first; only terminate the controller identities we own.
    for name in ['DUAL_SEED_CONTROLLER.json','STAGE_CONTROLLER.json']:
        path=LEGACY/name
        if path.exists():
            d=read(path)
            if alive(d):
                cmd=(Path('/proc')/str(d['pid'])/'cmdline').read_bytes()
                if not any(s in cmd for s in [b'dual_seed_queue.py',b'gpu7_score.py']):raise ValueError('Unexpected old controller')
                os.kill(d['pid'],signal.SIGTERM)
    # stdin guards release only our remote child. Verify actual disappearance.
    from listen_cluster import transport
    cl=read(TRAIN/'runs/baseline/cluster.json');n=cl['nodes'][3]
    code="import pathlib,json\nfound=[]\nfor p in pathlib.Path('/proc').iterdir():\n try:\n  if p.name.isdigit()and b'gpu7_score.py\\x00worker'in(p/'cmdline').read_bytes():found.append(int(p.name))\n except (FileNotFoundError,ProcessLookupError,PermissionError):pass\nprint(json.dumps(found))"
    for _ in range(60):
        r=subprocess.run(transport(cl,n,[str(PYTHON),'-c',code]),capture_output=True,text=True,timeout=30,check=True)
        if not json.loads(r.stdout):break
        time.sleep(2)
    else:raise RuntimeError('Old owned GPU7 worker remains; no overlapping evaluation launch')
    save(LEGACY/'SUPERSEDED_BY_EVENT4_EVAL28.json',dict(work=str(WORK),at=time.time(),training_resume=False))

def pause_training():
    from p2_loss_search28.checkpoint import checked
    from p2_loss_search28.resources import idle
    cluster=read(TRAIN/'runs/baseline/cluster.json');controller=read(TRAIN/'CONTROLLER.json')
    (TRAIN/'STOP').write_text('User requested Event4 double-seed evaluation before further training.\n')
    requested={}
    def stop_current():
        state=read(TRAIN/'STATE.json');name=state['candidate'];out=TRAIN/'runs'/name
        if name not in requested:
            requested[name]=read(out/'status.json')['step']if(out/'status.json').exists()else 0
            save(WORK/'PAUSE_REQUEST.json',dict(candidates=requested,controller=controller,at=time.time()))
        (out/'STOP').write_text('Save complete current state and release28 GPUs for Event evaluation.\n')
        return name,out
    stop_current()
    while alive(controller):stop_current();time.sleep(5)
    name,out=stop_current();before=requested[name]
    with (TRAIN/'queue.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        markers=sorted(out.glob('step_*.complete.json'));assert markers
        cp=markers[-1].with_suffix('').with_suffix('.pt');v=checked(cp)
        assert v['step']>=before and len(v['rank_rng'])==28 and v['identity']['world']==28
        while True:
            snap=idle(cluster);save(WORK/'RELEASE_RESOURCES.json',snap)
            if snap['idle']:break
            time.sleep(15)
        save(WORK/'TRAINING_PAUSED_ACCEPTED.json',dict(success=True,candidate=name,
             checkpoint=str(cp),step=v['step'],sha256=read(cp.with_suffix('.complete.json'))['sha256'],
             rank_rng=28,adam_verified=True,allocation_idle=True,at=time.time()))

def command(argv,log):
    with log.open('ab',buffering=0)as stream:
        p=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.PIPE)
        try:
            while p.poll()is None:
                if(WORK/'STOP').exists():
                    (log.parent/'STOP').touch(exist_ok=True)
                    p.stdin.close()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    raise RuntimeError('Evaluation STOP')
                time.sleep(2)
            if p.returncode:raise RuntimeError(f'Command failed {p.returncode}: {log}')
        finally:
            if p.stdin and not p.stdin.closed:p.stdin.close()

def seal_score(work):
    from reference_events import freeze
    from stage_driver import verify_scoring
    freeze(work);p=read(work/'protocol.json');arm=p['arms'][0]['name'];base='metrics/'+arm+'/seamless/'
    save(work/'SCORING_COMPLETE.json',dict(success=True,at=time.time(),
        checkpoint_sha256=read(work/'export/RECEIPT.json')['original_sha256'],
        evidence={f:sha(work/f)for f in ['FINAL_AUDIT.json','RESULTS.json','REFERENCE_EVENTS_ACCEPTED.json',
           'event_pairs_fixed_reference.jsonl','fixed_reference_event_summary.json',base+'pairs.jsonl',base+'audio_pairs.jsonl',
           'events/PARALLEL_EXECUTION_ACCEPTED.json','events/DETECTION_COMPLETE.json','events/detector_contract.json']},
        partial_event_coverage=read(work/'EVENTS_COMPLETE.json')['status']!='complete',detector_human_calibration_pending=True))
    verify_scoring(work)

def report():
    from dual_seed_compare import compare,strict_mean
    from p2_event_presence.contracts import rows
    from stage_driver import verify_scoring
    data={};native={};evidence={}
    for name in NAMES:
        data[name]={};native[name]={}
        for seed in SEEDS:
            out=folder(name,seed);verify_scoring(out);p=read(out/'protocol.json');arm=p['arms'][0]
            assert p['seed']==seed and arm['step']==500
            for item in read(out/'shared_items.json'):
                m=read(out/'generated'/arm['name']/item['key']/'metadata.json')
                assert m['seed']==seed and m['checkpoint_sha256']==arm['original_sha256']
            base=out/'metrics'/arm['name']/'seamless'
            data[name][seed]=dict(events=rows(out/'event_pairs_fixed_reference.jsonl'),video=rows(base/'pairs.jsonl'),audio=rows(base/'audio_pairs.jsonl'))
            native[name][str(seed)]=read(out/'RESULTS.json')[arm['name']+'/seamless']['metrics']
            evidence[str(out/'SCORING_COMPLETE.json')]=sha(out/'SCORING_COMPLETE.json')
    result=compare(data,read(LEGACY/'tuning64.json'),read(LEGACY/'PROMOTION_POLICY.json'))
    result.update(native_metrics_by_seed=native,native_metric_means={n:{k:strict_mean([native[n][str(s)].get(k)for s in SEEDS])for k in native[n]['137']}for n in NAMES},
        evidence=evidence,automatic_training_resume=False,best_model_selected=False,plan_sha256=sha(WORK/'PLAN.json'))
    save(WORK/'COMPARISON.json',result)

def run():
    plan=read(WORK/'PLAN.json');tests=read(WORK/'TESTS_ACCEPTED.json');assert tests['success']
    for f,h in plan['source'].items():
        if sha(f)!=h:raise ValueError('Prepared source changed')
    with(WORK/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,at=time.time()))
        stop_old_evaluation()
        if not(WORK/'TRAINING_PAUSED_ACCEPTED.json').exists():pause_training()
        from p2_loss_search28.resources import idle
        import eval_runtime as E
        from stage_driver import verify_scoring
        from listen_cluster import transport
        for name in NAMES:
            for seed in SEEDS:
                out=folder(name,seed)
                if(WORK/'STOP').exists():return
                if(out/'SCORING_COMPLETE.json').exists():verify_scoring(out);continue
                save(WORK/'STATUS.json',dict(stage='generation_metrics',candidate=name,seed=seed,at=time.time()))
                if not(out/'FINAL_AUDIT.json').exists():
                    if not idle(plan['cluster'])['idle']:raise RuntimeError('28-GPU allocation not released')
                    command([str(PYTHON),'-u',str(LEGACY/'evaluate28.py'),'queue','--work',str(out)],out/'eval28.log')
                E.final(out)
                if not idle(plan['cluster'])['idle']:raise RuntimeError('Generation workers have not exited')
                save(WORK/'STATUS.json',dict(stage='event_scoring',candidate=name,seed=seed,at=time.time()))
                guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
                argv=[str(PYTHON),str(guard),'--guard',str(PYTHON),'-u',str(WORK/'parallel_events.py'),'run','--work',str(out)]
                command(transport(plan['cluster'],plan['cluster']['nodes'][0],argv),out/'parallel_events.log')
                seal_score(out)
        report();save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=512,training_remains_stopped=True))
        save(WORK/'STATUS.json',dict(stage='awaiting_experiment_direction',at=time.time(),training_resumed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    try:globals()[args.action]()
    except Exception as exc:
        save(WORK/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
