"""Dedicated 3-node/24-GPU dynamically claimed evaluation tasks, training paused."""
import os,sys,time,json,signal,subprocess,fcntl,uuid,socket,threading
from pathlib import Path
from eval_common import HERE,ROOT,PY,read,write,sha,environment,ready,prepare,build_jobs
WORK=Path(os.environ.get('P2_EVAL_CLUSTER_WORK',str(HERE/'cluster24')))
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import digest

def publish(task):
    path=WORK/'tasks'/(task['id']+'.json')
    if path.exists():assert read(path)==task
    else:write(path,task)
def state(t):
    p=WORK/'states'/(t['id']+'.json')
    return read(p) if p.exists() else {}
def done(t):return state(t).get('status')=='complete'
def all_tasks():return [read(p) for p in sorted((WORK/'tasks').glob('*.json'))]

def semantic_tasks():
    inputs=ROOT/'experiments/p2_test48_planA_step200_eval_20261028/semantic/prepared/inputs.jsonl'
    jobs=[json.loads(l) for l in inputs.read_text().splitlines()];assert len(jobs)==48
    finished=set();runs=[]
    for p in sorted((HERE/'semantic').glob('input_*.jsonl')):
        run=HERE/'semantic'/p.stem.replace('input_','run_')
        rows=[json.loads(l) for l in p.read_text().splitlines()]
        if (run/'protocol.json').exists():runs.append(dict(path=str(run),inputs=str(p),job_ids=[j['job_id'] for j in rows]))
        if (run/'summary.json').exists():finished.update(j['job_id'] for j in rows)
    for i,j in enumerate(jobs):
        if j['job_id'] in finished:continue
        inp=WORK/'semantic_inputs'/f'{i:03d}.jsonl';inp.parent.mkdir(parents=True,exist_ok=True)
        inp.write_text(json.dumps(j,ensure_ascii=False)+'\n');run=WORK/'semantic_runs'/f'{i:03d}'
        runs.append(dict(path=str(run),inputs=str(inp),job_ids=[j['job_id']]))
        publish(dict(id=f'semantic_{i:03d}',kind='semantic',priority=0,cmd=['/private/software/conda/envs/vllm/bin/python',
            str(ROOT/'scripts/run_listener_semantic_prompt.py'),'--inputs',str(inp),'--output',str(run),'--batch','1','--memory','.65','--attempts','3']))
    write(WORK/'semantic_index.json',dict(jobs=jobs,runs=runs,reused_completed_samples=len(finished)))

def recover48():
    import p2_guided_chain.backend
    sys.path.append(str(ROOT/'scripts'))
    from p2_restore_full50h import recover_job
    index=read(WORK/'semantic_index.json');records=[];audits=[]
    for job in index['jobs']:
        runs=[];protocols={}
        for entry in index['runs']:
            if job['job_id'] not in entry['job_ids']:continue
            run=Path(entry['path'])
            if not (run/'protocol.json').exists():continue
            proto=read(run/'protocol.json');source=[json.loads(l) for l in Path(entry['inputs']).read_text().splitlines()]
            if digest(source)!=proto['input_sha256']:raise ValueError('Semantic input identity mismatch')
            runs.append(run);protocols[str(run)]=proto
        _,events,audit,selected,reviews=recover_job(job,runs,protocols);original=job['record']
        audio=lambda es:sorted((e for e in es if e['modality']=='audio'),key=lambda e:e['event_id'])
        assert audio(events)==audio(original['events'])
        records.append(dict(original,events=events,annotation_view='full50h_hard_check_with_original_fallback',
            original_record_sha256=digest(original),recovery_policy='safe_components_else_verbatim_original; semantic_review_record_only'))
        audits.append(dict(audit,sample_id=original['sample_id'],selected_proposal=str(selected) if selected else None,review_records=reviews,audio_unchanged=True))
    by={r['sample_id']:r for r in records};assert len(records)==len(by)==48
    for split in ['test48','test24']:
        ids=[o['sample_id'] for o in read(HERE/split/'observations.json')]
        p=HERE/split/'integrated_labels.jsonl';p.write_text(''.join(json.dumps(by[s],ensure_ascii=False)+'\n' for s in ids))
        write(HERE/split/'annotation_audit.json',[a for a in audits if a['sample_id'] in ids])
        write(HERE/split/'COMPLETE.json',dict(status='complete',samples=len(ids),excluded=0,labels_sha256=sha(p),model_review='record_only'))
    write(HERE/'semantic48_status.json',dict(phase='complete',completed_samples=48,total_samples=48,execution='dedicated_24gpu'))

def condition_tasks(jobs):
    ids=read(HERE/'plans/protocol.json')['sample_ids']
    # Four samples share a loaded encoder; workers dynamically take the next shard.
    for start in range(0,len(ids),4):
        p=WORK/'ids'/f'new_{start:03d}.json';write(p,ids[start:start+4])
        for part in ['text','input_plan']:
            publish(dict(id=f'cache_{part}_{start:03d}',kind='cache',priority=1,
                cmd=[str(PY),str(HERE/'eval_worker.py'),'cache',part,str(p)]))
    legacy_ids=[o['sample_id'] for o in read(HERE/'test48/observations.json')]
    run=next(j['run'] for j in jobs if j['family']=='legacy' and j['arm']=='legacy_E1F')
    for start in range(0,len(legacy_ids),4):
        p=WORK/'ids'/f'legacy_{start:03d}.json';write(p,legacy_ids[start:start+4])
        publish(dict(id=f'cache_legacy_{start:03d}',kind='cache',priority=1,
            cmd=[str(PY),str(HERE/'eval_worker.py'),'legacy-cache',run,str(p)]))

def generation_tasks(jobs):
    duration={o['sample_id']:o['requested_duration_s'] for split in ['train24','test48'] for o in read(HERE/split/'observations.json')}
    pending=[]
    for i,j in enumerate(jobs):
        if not ready(j['checkpoint']):pending.append(j);continue
        p=WORK/'generation_jobs'/f'{i:04d}.json';write(p,j)
        publish(dict(id=f'generate_{i:04d}',kind='generate',priority=2,duration=duration[j['sample_id']],job=j,
            cmd=[str(PY),str(HERE/'eval_worker.py'),'generate',str(p)]))
    write(WORK/'waiting_final_checkpoints.json',dict(jobs=pending,count=len(pending),reason='Training paused by user; never substitute intermediate checkpoints'))

def worker(rank):
    gpu=rank%8;stop=[False]
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,lambda *_:stop.__setitem__(0,True))
    WORK.mkdir(exist_ok=True);(WORK/'logs').mkdir(exist_ok=True);(WORK/'locks').mkdir(exist_ok=True)
    while not stop[0] and not (WORK/'STOP').exists():
        tasks=sorted(all_tasks(),key=lambda t:(t['priority'],-t.get('duration',0),t['id']));claimed=False
        for t in tasks:
            if done(t) or state(t).get('failures',0)>=3:continue
            lock=(WORK/'locks'/(t['id']+'.lock')).open('a')
            try:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if done(t):continue
                free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<125000:continue
                claimed=True;previous=state(t);failures=previous.get('failures',0)
                logpath=WORK/'logs'/(t['id']+'_'+uuid.uuid4().hex[:8]+'.log');started=time.time()
                write(WORK/'states'/(t['id']+'.json'),dict(status='running',rank=rank,host=socket.gethostname(),gpu=gpu,pid=os.getpid(),failures=failures,started=started,log=str(logpath)))
                env=environment(gpu)
                if t['kind']=='semantic':env.update(LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',PYTHONHASHSEED='0')
                # Archive only an incomplete generation; complete artifacts use native reuse.
                if t['kind']=='generate':
                    out=Path(t['job']['output'])
                    if (out/'complete.json').exists():
                        from eval_chain import valid_generation
                        if valid_generation(t['job']):write(WORK/'states'/(t['id']+'.json'),dict(status='complete',reused=True));break
                    elif out.exists():
                        archive=WORK/'failed_attempts'/(t['id']+'_'+uuid.uuid4().hex);archive.parent.mkdir(exist_ok=True);out.rename(archive)
                with logpath.open('w') as log:
                    p=subprocess.Popen(t['cmd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
                    try:
                        while p.poll() is None:
                            if stop[0] or (WORK/'STOP').exists() or time.time()-started>3600:
                                os.killpg(p.pid,signal.SIGTERM)
                                try:p.wait(timeout=10)
                                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                                break
                            time.sleep(2)
                    finally:
                        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
                success=p.returncode==0
                if t['kind']=='semantic' and not success and 'Context overflow; no truncation allowed' in logpath.read_text()[-12000:]:success=True
                if t['kind']=='generate' and success:
                    from eval_chain import valid_generation
                    success=valid_generation(t['job'])
                write(WORK/'states'/(t['id']+'.json'),dict(status='complete' if success else 'retry',rank=rank,returncode=p.returncode,
                    failures=failures+int(not success and not stop[0]),seconds=time.time()-started,log=str(logpath)))
                break
            finally:lock.close()
        if (WORK/'AVAILABLE_COMPLETE.json').exists():break
        if not claimed:time.sleep(3)

def node(index):
    children=[];stopping=[False]
    def stop(*_):
        stopping[0]=True
        for p in children:
            if p.poll() is None:p.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for gpu in range(8):children.append(subprocess.Popen([str(PY),str(__file__),'worker',str(index*8+gpu)]))
        codes=[p.wait() for p in children]
        if any(codes) and not stopping[0]:raise RuntimeError(str(codes))
    finally:stop()

def controller():
    WORK.mkdir(exist_ok=True)
    lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (HERE/'TRAINING_PAUSED.json').exists() and (ROOT/'experiments/p2_full50h_duration384_20261028/STOP').exists()
    semantic_tasks()
    import p2_guided_chain.backend
    sys.path.append(str(ROOT/'scripts'));from listen_cluster import launch
    cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json');cancel=threading.Event();errors=[]
    commands=[[str(PY),str(__file__),'node',str(i)] for i in range(3)]
    def run_cluster():
        try:launch(cfg,commands,WORK/'launches'/str(time.time_ns()),cancel_event=cancel)
        except Exception as exc:errors.append(repr(exc))
    thread=threading.Thread(target=run_cluster);thread.start();jobs=None;generation_published=False
    try:
        while not (WORK/'STOP').exists():
            if errors:raise RuntimeError(errors)
            ts=all_tasks();failed=[t['id'] for t in ts if state(t).get('failures',0)>=3]
            if failed:raise RuntimeError('Three failed attempts: '+str(failed))
            if jobs is None and all(done(t) for t in ts if t['kind']=='semantic'):
                recover48();jobs=prepare();condition_tasks(jobs)
            if jobs is not None and not generation_published and all(done(t) for t in all_tasks() if t['kind']=='cache'):
                generation_tasks(jobs);generation_published=True
            if generation_published:
                ts=all_tasks();gens=[t for t in ts if t['kind']=='generate']
                groups=sorted({(t['job']['arm'],t['job']['split']) for t in gens})
                for arm,split in groups:
                    if all(done(t) for t in gens if t['job']['arm']==arm and t['job']['split']==split):
                        publish(dict(id=f'metrics_{arm}_{split}',kind='metrics',priority=3,cmd=[str(PY),str(HERE/'eval_metrics.py'),'evaluate',arm,split]))
                ts=all_tasks()
                if all(done(t) for t in ts):
                    write(WORK/'AVAILABLE_COMPLETE.json',dict(status='available_final_weights_evaluated',outputs=sum(t['kind']=='generate' for t in ts),waiting_final_checkpoint_outputs=read(WORK/'waiting_final_checkpoints.json')['count'],training_remains_paused=True))
                    break
            ts=all_tasks();counts={k:dict(total=sum(t['kind']==k for t in ts),complete=sum(t['kind']==k and done(t) for t in ts),running=sum(t['kind']==k and state(t).get('status')=='running' for t in ts)) for k in ['semantic','cache','generate','metrics']}
            write(HERE/'eval_status.json',dict(phase='dedicated_3nodes_24gpu',tasks=counts,at=time.time(),training_paused=True,checkpoint_policy='final_only',nodes=[n['name'] for n in cfg['nodes']]))
            write(HERE/'semantic48_status.json',dict(phase='complete' if jobs else 'dedicated_24gpu_integrating',reused_samples=read(WORK/'semantic_index.json')['reused_completed_samples'],tasks=counts['semantic']))
            time.sleep(5)
        if (WORK/'AVAILABLE_COMPLETE.json').exists():thread.join(timeout=90)
    finally:cancel.set();thread.join(timeout=60)

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='node':node(int(sys.argv[2]))
    elif sys.argv[1]=='controller':
        try:controller()
        except Exception as exc:write(HERE/'eval_status.json',dict(phase='failed_24gpu',error=repr(exc),at=time.time()));raise
    else:raise ValueError(sys.argv[1])
