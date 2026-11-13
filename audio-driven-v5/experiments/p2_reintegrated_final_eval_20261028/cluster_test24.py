"""All pipelines use the unchanged original independent test24 and two seeds."""
import os,sys,time,threading,fcntl
from pathlib import Path
from eval_common import HERE,ROOT,PY,read,write,sha,ready
WORK=HERE/'cluster24_test24'
os.environ['P2_EVAL_CLUSTER_WORK']=str(WORK)
import cluster_eval as queue
assert queue.WORK==WORK

def prepare_scope():
    from eval_chain import valid_generation
    original=read(ROOT/'experiments/p2_test24_fourway_eval_20261027/observations.json')
    assert read(HERE/'test24/observations.json')==original
    ids={o['sample_id'] for o in original};assert len(ids)==24
    previous=HERE/'jobs.before_test24_correction.json'
    if not previous.exists():write(previous,read(HERE/'jobs.json'))
    jobs=[];reused=[]
    for j in read(previous):
        if j['sample_id'] not in ids:continue
        if j['family']=='matrix' and j['split']!='test24':continue
        new=dict(j,split='test24',output=str(HERE/'outputs_test24'/j['arm']/f'{j["sample_id"]}__seed{j["seed"]}'))
        target=Path(new['output']);source=Path(j['output'])
        if valid_generation(j):
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.symlink_to(source.resolve(),target_is_directory=True)
            if not valid_generation(new):raise ValueError('Reused output failed validation')
            reused.append(dict(output=str(target),source=str(source),sample_id=j['sample_id'],seed=j['seed'],arm=j['arm']))
        jobs.append(new)
    assert len(jobs)==336
    for arm in {j['arm'] for j in jobs}:
        rows=[j for j in jobs if j['arm']==arm]
        assert len(rows)==48 and {(j['sample_id'],j['seed']) for j in rows}=={(s,seed) for s in ids for seed in [137,251]}
    write(HERE/'jobs.json',jobs);write(HERE/'jobs.test24.json',jobs)
    write(WORK/'reuse_audit.json',reused)
    write(WORK/'scope.json',dict(samples=24,sample_ids=sorted(ids),seeds=[137,251],outputs_per_pipeline=48,
        available_pipelines=6,available_outputs=288,pending_feature_final_outputs=48,total_planned_outputs=336,
        input_observations_sha256=sha(HERE/'test24/observations.json'),annotation_sha256=sha(HERE/'test24/integrated_labels.jsonl'),
        policy='Same original independent test24 in every arm; updated text only in legacy inference',reused_outputs=len(reused)))
    return jobs

def controller():
    WORK.mkdir(exist_ok=True)
    lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (HERE/'TRAINING_PAUSED.json').exists() and (HERE/'test48/COMPLETE.json').exists()
    jobs=prepare_scope();durations={o['sample_id']:o['requested_duration_s'] for o in read(HERE/'test24/observations.json')};pending=[]
    for i,j in enumerate(jobs):
        if not ready(j['checkpoint']):pending.append(j);continue
        p=WORK/'generation_jobs'/f'{i:04d}.json';write(p,j)
        task=dict(id=f'generate_{i:04d}',kind='generate',priority=2,duration=durations[j['sample_id']],job=j,
            cmd=[str(PY),str(HERE/'eval_worker.py'),'generate',str(p)])
        queue.publish(task)
        from eval_chain import valid_generation
        if valid_generation(j):write(WORK/'states'/(task['id']+'.json'),dict(status='complete',reused=True))
    assert len(pending)==48
    write(WORK/'waiting_final_checkpoints.json',dict(jobs=pending,count=48,reason='AV+Feature paused at587; final1249 required'))
    import p2_guided_chain.backend
    sys.path.append(str(ROOT/'scripts'));from listen_cluster import launch
    cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json');cancel=threading.Event();errors=[]
    commands=[['env','P2_EVAL_CLUSTER_WORK='+str(WORK),str(PY),str(HERE/'cluster_eval.py'),'node',str(i)] for i in range(3)]
    def run():
        try:launch(cfg,commands,WORK/'launches'/str(time.time_ns()),cancel_event=cancel)
        except Exception as exc:errors.append(repr(exc))
    thread=threading.Thread(target=run);thread.start()
    try:
        while not (WORK/'STOP').exists():
            if errors:raise RuntimeError(errors)
            tasks=queue.all_tasks();failed=[t['id'] for t in tasks if queue.state(t).get('failures',0)>=3]
            if failed:raise RuntimeError('Three failed attempts: '+str(failed))
            gens=[t for t in tasks if t['kind']=='generate']
            for arm in sorted({t['job']['arm'] for t in gens}):
                if all(queue.done(t) for t in gens if t['job']['arm']==arm):
                    queue.publish(dict(id=f'metrics_{arm}_test24',kind='metrics',priority=3,
                        cmd=[str(PY),str(HERE/'eval_metrics.py'),'evaluate',arm,'test24']))
            tasks=queue.all_tasks()
            counts={k:dict(total=sum(t['kind']==k for t in tasks),complete=sum(t['kind']==k and queue.done(t) for t in tasks),
                running=sum(t['kind']==k and queue.state(t).get('status')=='running' for t in tasks)) for k in ['generate','metrics','metric_shard']}
            write(HERE/'eval_status.json',dict(phase='dedicated_24gpu_original_test24_only',tasks=counts,at=time.time(),
                samples_per_pipeline=24,outputs_per_pipeline=48,available_outputs=288,pending_feature_final_outputs=48,
                reused_outputs=read(WORK/'scope.json')['reused_outputs'],training_paused=True))
            if all(queue.done(t) for t in tasks):
                result=dict(status='available_final_weights_evaluated',outputs=288,samples_per_pipeline=24,pending_outputs=48,training_remains_paused=True)
                write(WORK/'AVAILABLE_COMPLETE.json',result);write(HERE/'eval_status.json',dict(phase='original_test24_complete',**result));break
            time.sleep(5)
        if (WORK/'AVAILABLE_COMPLETE.json').exists():thread.join(timeout=90)
    finally:cancel.set();thread.join(timeout=60)

if __name__=='__main__':
    try:controller()
    except Exception as exc:write(HERE/'eval_status.json',dict(phase='test24_queue_failed',error=repr(exc),at=time.time()));raise
