"""Prepare, launch and resume single-auxiliary Whisper training on8+8+8+6 GPUs."""
import argparse,copy,csv,fcntl,json,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
BASE=ROOT/'experiments/p2_whisper_continue30_20261106'

def output(arm):return BASE/arm


def prepare():
    import torch
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_continue30 import make_remaining,install,normalized
    from p2_whisper_ablation.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_j_event_feature_continue24 import equal
    torch.set_num_threads(2);BASE.mkdir(exist_ok=True)
    parent_path=ROOT/'configs/p2_whisper_event_20261106.json';parent=read(parent_path)
    saved=checked(Path(parent['output'])/'step_004000.pt')
    from p2_whisper_ablation.config import source_identity
    if saved['identity']!=read(Path(parent['output'])/'run.json') or saved['identity']['source']!=source_identity() or saved['step']!=4000 or len(saved['rank_rng'])!=31:raise ValueError('Parent4000 invalid')
    install()
    from p2_whisper_ablation import config as C
    for arm in ['event','feature']:
        parent_path=ROOT/f'configs/p2_whisper_{arm}_20261106.json';parent=read(parent_path);out=output(arm);out.mkdir(exist_ok=False)
        prefix=4000 if arm=='event' else 0;pairs,proof=make_remaining(parent,prefix);plan=out/'remaining.schedule.json'
        atomic_json(plan,dict(pairs=pairs,proof=proof),immutable=True)
        c=copy.deepcopy(parent);c['output']=str(out);c['training'].update(global_batch=30,steps=proof['steps'])
        c['duration_sampling'].update(window=480,start_step=prefix,manifest=str(plan),sha256=sha(plan))
        ranks=list(range(24,31))+[7]+list(range(8,24))+list(range(6))
        c['continuation30']=dict(parent_config=str(parent_path),parent_config_sha256=sha(parent_path),prefix_step=prefix,rng_parent_ranks=ranks)
        normalized(c);C.validate_baseline(c,read(c['baseline_config']));C.Schedule(c)
        atomic_json(out/'config.json',c,immutable=True)
        pre=read(Path(parent['output'])/'preflight.json');pre['source']=C.source_identity()
        pre['continuation_proof']=proof;atomic_json(out/'preflight.json',pre,immutable=True)
        cluster=read(ROOT/'experiments/p2_whisper_four_node_migration_20261106/event_new_cluster.json')
        by={n['name']:n for n in cluster['nodes']};cluster['nodes']=[by[n] for n in ['node8','node12','node4','node11']]
        for n in cluster['nodes']:n['gpus']=list(range(6 if n['name']=='node11' else 8));n['role']='master' if n['name']=='node8' else 'slave'
        cluster['master_port']=29931 if arm=='event' else 29932;cluster['training_script']=str(SELF)
        atomic_json(out/'cluster.json',cluster,immutable=True)
        if arm=='event':
            identity=copy.deepcopy(saved['identity']);identity.update(experiment=c,source=C.source_identity(),world=30)
            value=dict(saved,identity=identity,rank_rng=[saved['rank_rng'][i] for i in ranks]);target=out/'step_004000.pt'
            atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(step=4000,sha256=sha(target)))
            verified=checked(target)
            assert equal(verified['model'],saved['model']) and equal(verified['optimizer'],saved['optimizer'])
            assert equal(verified['rank_rng'],value['rank_rng'])
            atomic_json(out/'run.json',identity,immutable=True)
            atomic_json(out/'status.json',dict(step=4000,budget=c['training']['steps'],complete=False))
            atomic_json(out/'MIGRATION.json',dict(success=True,parent_checkpoint=str(Path(parent['output'])/'step_004000.pt'),model_optimizer_bitwise_equal=True,rng_bitwise_equal=True,rank_map=ranks,trajectory_identical=False,**proof))
        else:atomic_json(out/'PLAN.json',proof)
    atomic_json(BASE/'PREPARED.json',dict(success=True,at=time.time()))


def commands(c,resume,logs):
    cluster=read(Path(c['output'])/'cluster.json');result=[]
    for i,n in enumerate(cluster['nodes']):
        cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v) for k,v in cluster['network'].items()],cluster['python'],str(SELF),'node','--arm',c['loss'][3:],'--node-index',str(i),'--logs',str(logs)]
        if resume:cmd+=['--resume',str(resume)]
        result.append(cmd)
    return result


def node(arm,index,resume,logs):
    c=read(output(arm)/'config.json');cluster=read(output(arm)/'cluster.json');n=cluster['nodes'][index]
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,n['gpus']))
    offset=sum(len(x['gpus']) for x in cluster['nodes'][:index]);ps=[];fs=[];logs.mkdir(parents=True,exist_ok=True)
    try:
        for local,rank in enumerate(range(offset,offset+len(n['gpus']))):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='30',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cluster['nodes'][0]['host'],MASTER_PORT=str(cluster['master_port']))
            cmd=[cluster['python'],str(SELF),'train','--arm',arm]
            if resume:cmd+=['--resume',str(resume)]
            f=(logs/f'rank{rank:02d}.log').open('a');fs.append(f);ps.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT))
        atomic_json(logs/f"{n['name']}_workers.json",dict(node=n['name'],gpus=n['gpus'],ranks=list(range(offset,offset+len(n['gpus']))),pids=[p.pid for p in ps]))
        while any(p.poll() is None for p in ps):
            if any(p.poll() not in (None,0) for p in ps):raise RuntimeError('Training rank failed')
            time.sleep(2)
        if any(p.returncode for p in ps):raise RuntimeError('Training failed')
    finally:
        for p in ps:
            if p.poll() is None:p.terminate()
        for p in ps:
            try:p.wait(15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in fs:f.close()


def run():
    from p2_seamless200_whisper_eval import idle_snapshot
    from listen_cluster import launch
    from queue_p2_whisper_ablation31 import resume_point,verify_row,recent_rows
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        atomic_json(BASE/'controller.json',process_record())
        def state(stage,**kw):atomic_json(BASE/'state.json',dict(stage=stage,at=time.time(),**kw));print(stage,kw,flush=True)
        for arm in ['event','feature']:
            out=output(arm);c=read(out/'config.json');cluster=read(out/'cluster.json')
            if (out/'TRAINING_COMPLETE.json').exists() and read(out/'status.json')['complete']:continue
            for attempt in range(1,6):
                while True:
                    if (BASE/'STOP').exists() or (out/'STOP').exists():state('stopped',arm=arm);return
                    resources=idle_snapshot(cluster);atomic_json(BASE/'resources.json',resources)
                    if resources['all_idle']:break
                    state('waiting_selected30',arm=arm);time.sleep(10)
                resume=resume_point(out);logs=out/'queue'/f'attempt_{attempt}_{time.time_ns()}';cancel=threading.Event();done=threading.Event();started=time.time()
                state('training30',arm=arm,logs=str(logs),resume=str(resume) if resume else None,budget=c['training']['steps'])
                def watch():
                    while not done.wait(15):
                        try:
                            if (BASE/'STOP').exists():(out/'STOP').touch()
                            p=out/'metrics.jsonl';age=time.time()-max(started,p.stat().st_mtime if p.exists() else started)
                            if age>1800:raise TimeoutError('No training progress')
                            rows=recent_rows(p)
                            if rows and p.stat().st_mtime>=started:
                                row=rows[-1];verify_row(row,arm)
                                atomic_json(out/'RESUME_VERIFIED.json',dict(success=True,world=30,arm=arm,step=row['step'],sample=row,at=time.time()))
                                atomic_json(BASE/'speed.json',dict(arm=arm,step=row['step'],seconds=row['seconds'],budget=c['training']['steps'],at=time.time()))
                        except Exception as e:atomic_json(BASE/'WATCH_ERROR.json',dict(error=repr(e),at=time.time()));cancel.set();return
                t=threading.Thread(target=watch,daemon=True);t.start()
                try:launch(cluster,commands(c,resume,logs/'ranks'),logs,cancel_event=cancel)
                except Exception as e:
                    state('retry',arm=arm,error=repr(e),attempt=attempt)
                    if attempt==5:raise
                    time.sleep(10);continue
                finally:done.set();t.join(20)
                final=read(out/'status.json')
                if final['complete']:
                    from p2_whisper_ablation.checkpoint import checked
                    v=checked(out/f"step_{final['step']:06d}.pt");assert v['identity']==read(out/'run.json') and len(v['rank_rng'])==30
                    atomic_json(out/'TRAINING_COMPLETE.json',dict(success=True,step=final['step'],at=time.time()));break
                if (out/'STOP').exists():state('stopped',arm=arm);return
                raise RuntimeError('Unexpected end before completion')
        state('complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','node','train']);p.add_argument('--arm',choices=['event','feature']);p.add_argument('--node-index',type=int);p.add_argument('--resume',type=Path);p.add_argument('--logs',type=Path);a=p.parse_args()
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    if a.command=='prepare':prepare();return
    from p2_whisper_continue30 import install
    install()
    if a.command=='run':run()
    elif a.command=='node':node(a.arm,a.node_index,a.resume,a.logs)
    else:
        from p2_whisper_ablation.training import train
        train(read(output(a.arm)/'config.json'),a.resume)

if __name__=='__main__':main()
