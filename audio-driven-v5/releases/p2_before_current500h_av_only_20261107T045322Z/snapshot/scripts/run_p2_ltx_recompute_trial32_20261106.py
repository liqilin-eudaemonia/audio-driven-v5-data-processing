"""Owned graceful pause, full-loss ABBA, exact migration or original-policy recovery."""
from pathlib import Path
import sys,os,json,time,copy,threading,subprocess,fcntl,runpy,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,digest,atomic_json
from listen_cluster import launch
OLD=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'
NEW=ROOT/'experiments/p2_ltx_clip_recompute32_20261106'
WORK=ROOT/'experiments/p2_ltx_clip_recompute_trial32_20261106'
NATIVE=ROOT/'scripts/p2_ltx_events.py';PROFILE=ROOT/'scripts/p2_ltx_recompute_profile32_20261106.py'

def state(stage,**kw):
    value=dict(stage=stage,at=time.time(),pid=os.getpid(),**kw);atomic_json(WORK/'state.json',value);print(json.dumps(value),flush=True)

def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def latest(out):
    steps=sorted(out.glob('step_*.complete.json'))
    if not steps:raise ValueError('No complete checkpoint')
    return steps[-1].with_suffix('').with_suffix('.pt')

def wait_idle(cluster):
    module=runpy.run_path(str(OLD/'coordinator.py'),run_name='resource_probe_only')
    deadline=time.time()+900
    while True:
        observations=module['probe_all'](cluster);busy=module['blockers'](observations,False)
        atomic_json(WORK/'resources.json',dict(at=time.time(),observations=observations,blockers=busy))
        if not busy:return
        if time.time()>deadline:raise TimeoutError('GPUs did not become idle; no unrelated jobs were killed')
        time.sleep(5)

def monitor(out,cluster):
    (out/'monitor.py').write_text((OLD/'monitor.py').read_text()) if out!=OLD else None
    # Previous monitor stops with its old controller; identify liveness before replacing.
    record=read(out/'MONITOR.json') if (out/'MONITOR.json').exists() else {}
    pid=record.get('pid');alive=False
    if pid:
        try:alive=(Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
        except OSError:pass
    if alive:return
    with (out/'monitor.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([cluster['python'],'-u',str(out/'monitor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    atomic_json(out/'MONITOR.json',dict(pid=proc.pid,at=time.time()))

def migrate(origin,out,identity):
    import torch
    from p2_ltx_events.checkpoint import checked
    from p2_listen.cache import atomic_torch
    saved=checked(origin);target=out/origin.name
    if target.exists():raise ValueError('Never overwrite an existing checkpoint')
    value=dict(saved,identity=identity)
    atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(step=value['step'],sha256=sha(target)))
    loaded=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
    assert loaded['identity']==identity and len(loaded['rank_rng'])==32
    for k in ('model','optimizer','rank_rng'):
        if not equal(saved[k],loaded[k]):raise ValueError('Migration changed '+k)
    atomic_json(out/'MIGRATION.json',dict(step=saved['step'],origin=str(origin),origin_sha256=read(origin.with_suffix('.complete.json'))['sha256'],world=32,model_optimizer_all32rng_bitwise_equal=True,global_batch=32,save_every=300,data_loss_schedule_unchanged=True,policy=identity['experiment']['training']['gradient_checkpointing'],at=time.time()))
    return target

def launch_training(out,checkpoint,cluster,profile=False):
    cli=runpy.run_path(str(NATIVE),run_name='recompute_commands_only')
    commands=cli['commands'](cluster,out/'config.json',resume=checkpoint)
    if profile:
        for command in commands:command[command.index(str(NATIVE))]=str(PROFILE)
    atomic_json(out/'COMMANDS.json',commands)
    atomic_json(out/'QUEUE_STATE.json',dict(state='training',phase='startup',controller_pid=os.getpid(),resume=str(checkpoint),at=time.time()))
    monitor(out,cluster)
    state('production_three_quarters' if profile else 'production_all',resume=str(checkpoint),output=str(out))
    launch(cluster,commands,out/('launch_'+str(int(time.time()))))
    atomic_json(out/'QUEUE_STATE.json',dict(state='finished',controller_pid=os.getpid(),at=time.time()))

def prepare():
    import torch
    torch.set_num_threads(2)
    from p2_ltx_events.config import source_identity
    run=read(OLD/'run.json');parent=read(OLD/'config.json')
    assert run['source']==source_identity() and run['experiment']==parent and run['world']==32
    from p2_ltx_recompute_profile_20261106 import install,normalize_profile,PARENT_SHA
    install()
    from p2_ltx_events import config as C,training as T
    from p2_ltx_events.data import data_identity
    c=copy.deepcopy(parent);c['output']=str(NEW);c['training']['gradient_checkpointing']='three_quarters'
    c['recompute_profile']=dict(parent_config=str(OLD/'config.json'),parent_sha256=PARENT_SHA,policy='three_quarters',world=32)
    assert normalize_profile(c)==parent
    C.validate_baseline(c,read(c['hidden_event_parent_config']))
    did=data_identity(c);assert did==run['data_identity']
    pre=read(OLD/'preflight.json');assert pre['source']==run['source'] and pre['data_identity']==did
    pre.update(configuration_sha256=digest(c),source=C.source_identity())
    T.check_preflight(c,pre,C.source_identity(),did)
    NEW.mkdir(exist_ok=True)
    atomic_json(NEW/'config.json',c,immutable=True);atomic_json(NEW/'preflight.json',pre,immutable=True)
    cluster=read(OLD/'cluster.json');cluster['master_port']=29838
    cluster.update(training_script=str(PROFILE),training_config=str(NEW/'config.json'))
    assert cluster['network']['P2_TRAIN_MEMORY_GIB']=='134'
    atomic_json(NEW/'cluster.json',cluster,immutable=True)
    # Profile must reject changes to all non-execution choices.
    for field,value in [('global_batch',16),('save_every',50),('connector_lr',1e-3)]:
        bad=copy.deepcopy(c);bad['training'][field]=value
        try:normalize_profile(bad)
        except ValueError:pass
        else:raise AssertionError('Profile accepted changed '+field)
    atomic_json(WORK/'CPU_VALIDATION.json',dict(success=True,parent_source_files=len(run['source']),new_source_files=len(C.source_identity()),same_data_identity=True,strict_profile_rejects_unrelated_changes=True,at=time.time()))
    return cluster,dict(run,experiment=c,source=C.source_identity())

def main():
    WORK.mkdir(exist_ok=True)
    with (WORK/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'PAUSE_RECEIPT.json').exists():raise ValueError('One-shot trial already started')
        state('cpu_validation_while_training_continues')
        cluster,new_identity=prepare();old_cluster=read(OLD/'cluster.json')
        if (OLD/'STOP').exists() or (NEW/'STOP').exists():raise RuntimeError('Existing STOP; preserve user intent')
        token=json.dumps(dict(owner='ltx_recompute32_trial',at=time.time(),pid=os.getpid()))
        with (OLD/'STOP').open('x') as f:f.write(token)
        state('waiting_graceful_complete_save',observed_step=read(OLD/'status.json')['step'])
        checkpoint=None;adopted=False
        try:
            deadline=time.time()+1200
            while not (OLD/'launch/completed.json').exists():
                if time.time()>deadline:raise TimeoutError('Graceful save still running; no forced termination')
                time.sleep(3)
            assert read(OLD/'launch/completed.json')['returncodes']==[0,0,0,0]
            checkpoint=latest(OLD);step=read(OLD/'status.json')['step'];assert int(checkpoint.stem.split('_')[-1])==step
            from p2_ltx_events.checkpoint import checked
            v=checked(checkpoint);assert v['identity']==read(OLD/'run.json') and len(v['rank_rng'])==32;del v
            atomic_json(WORK/'PAUSE_RECEIPT.json',dict(checkpoint=str(checkpoint),step=step,sha256=sha(checkpoint),at=time.time()))
            wait_idle(old_cluster)
            commands=[];script=ROOT/'scripts/p2_ltx_recompute_benchmark32_20261106.py'
            for i,n in enumerate(old_cluster['nodes']):
                commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v) for k,v in old_cluster['network'].items()],old_cluster['python'],str(script),'node','--config',str(OLD/'config.json'),'--work',str(WORK),'--checkpoint',str(checkpoint),'--node-index',str(i),'--start-step',str(step+1),'--steps','12'])
            state('benchmark32',checkpoint=str(checkpoint),matched_batches_per_arm=12,order=['all','three_quarters','three_quarters','all'])
            cancel=threading.Event();done=threading.Event();started=time.time()
            def watchdog():
                while not done.wait(15):
                    p=WORK/'progress.json';stamp=p.stat().st_mtime if p.exists() else started
                    if time.time()-stamp>900:atomic_json(WORK/'TIMEOUT.json',dict(at=time.time()));cancel.set();return
            watch=threading.Thread(target=watchdog,daemon=True);watch.start()
            try:launch(old_cluster,commands,WORK/'launch',cancel_event=cancel)
            finally:done.set();watch.join(20)
            result=read(WORK/'RESULT.json');state('benchmark_complete',adopt=result['adopt_three_quarters'])
            if result['adopt_three_quarters']:
                if (NEW/'STOP').exists() or (OLD/'STOP').read_text()!=token:raise RuntimeError('STOP changed during trial')
                state('migrating_verified_state',step=step)
                migrated=migrate(checkpoint,NEW,new_identity)
                atomic_json(NEW/'run.json',new_identity,immutable=True)
                atomic_json(NEW/'status.json',dict(step=step,budget=8882,complete=False))
                atomic_json(NEW/'ADOPTION_ACCEPTED.json',dict(success=True,trial=str(WORK/'RESULT.json'),baseline_mean_seconds=result['baseline_mean_seconds'],candidate_mean_seconds=result['candidate_mean_seconds'],all_four_arms_safe_and_bitwise_equal=True,process_memory_limit_gib=134,step=step,at=time.time()))
                atomic_json(OLD/'SUPERSEDED_BY.json',dict(path=str(NEW),step=step,at=time.time()))
                wait_idle(cluster);adopted=True
                try:launch_training(NEW,migrated,cluster,profile=True)
                except Exception:
                    # A runtime failure never strands all32 GPUs: revert the latest
                    # complete candidate checkpoint with exact model/Adam/RNG intact.
                    atomic_json(NEW/'RUNTIME_FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()))
                    (NEW/'STOP').touch()
                    wait_idle(cluster);newest=latest(NEW)
                    if int(newest.stem.split('_')[-1])>step:
                        checkpoint=migrate(newest,OLD,read(OLD/'run.json'))
                    adopted=False
                    raise
        except Exception:
            state('trial_or_candidate_failed',error=traceback.format_exc())
            atomic_json(WORK/'ERROR.json',dict(traceback=traceback.format_exc(),at=time.time()))
        finally:
            if not adopted and checkpoint is not None and (OLD/'STOP').exists() and (OLD/'STOP').read_text()==token:
                if (WORK/'CANCEL').exists():state('cancelled_original_remains_saved');return
                wait_idle(old_cluster);(OLD/'STOP').unlink()
                atomic_json(WORK/'ORIGINAL_RESTORED.json',dict(checkpoint=str(checkpoint),at=time.time()))
                launch_training(OLD,checkpoint,old_cluster,profile=False)

if __name__=='__main__':
    try:main()
    except Exception:
        WORK.mkdir(exist_ok=True);atomic_json(WORK/'CONTROLLER_ERROR.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
