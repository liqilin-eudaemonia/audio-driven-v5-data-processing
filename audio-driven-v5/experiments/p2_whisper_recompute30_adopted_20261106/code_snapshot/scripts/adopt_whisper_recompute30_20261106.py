"""Graceful immediate save, exact checkpoint metadata migration, restart native queue."""
import copy,os,sys,time,subprocess,json,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_whisper_recompute_profile_20261106 import BASE,PARENT,install,normalize_profile


def live(record):
    try:
        s=Path('/proc',str(record['pid']),'stat').read_text().split()
        return s[21]==record['start_ticks'] and s[2]!='Z'
    except FileNotFoundError:return False


def run():
    import torch
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx();install();torch.set_num_threads(2)
    from p2_whisper_ablation import config as C
    from p2_whisper_ablation.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_j_event_feature_continue24 import equal
    from queue_p2_whisper_ablation31 import resume_point
    from p2_seamless200_whisper_eval import idle_snapshot
    BASE.mkdir(exist_ok=True)
    with (BASE/'migration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (BASE/'LAUNCHED.json').exists():raise ValueError('Already adopted')
        for arm in ['event','feature']:
            out=BASE/arm;out.mkdir(exist_ok=True)
            path=PARENT/arm/'config.json';c=read(path);c['output']=str(out)
            if arm=='event':c['training']['gradient_checkpointing']='three_quarters'
            c['recompute_profile']=dict(parent_config=str(path),parent_sha256=sha(path),policy=c['training']['gradient_checkpointing'])
            normalize_profile(c);C.validate(c);C.validate_baseline(c,read(c['baseline_config']))
            atomic_json(out/'config.json',c,immutable=True)
            pre=read(PARENT/arm/'preflight.json');pre['source']=C.source_identity();atomic_json(out/'preflight.json',pre,immutable=True)
            cluster=read(PARENT/arm/'cluster.json');cluster['training_script']=str(ROOT/'scripts/p2_whisper_recompute_profile30_20261106.py')
            assert cluster['network']['P2_TRAIN_MEMORY_GIB']=='134'
            atomic_json(out/'cluster.json',cluster,immutable=True)
        original=read(PARENT/'controller.json');old=PARENT/'event'
        if not live(original):raise RuntimeError('Original controller not live before immediate save')
        if (PARENT/'STOP').exists() or (old/'STOP').exists():raise RuntimeError('Existing STOP; do not replace')
        token=json.dumps(dict(owner='adopt_three_quarters',at=time.time(),pid=os.getpid()))
        with (old/'STOP').open('x') as f:f.write(token)
        atomic_json(BASE/'state.json',dict(stage='waiting_immediate_complete_save',original=original,at=time.time()))
        deadline=time.time()+1200
        while live(original):
            if time.time()>deadline:raise TimeoutError('Save still running; preserve old STOP and do not kill')
            time.sleep(2)
        path=resume_point(old);saved=checked(path)
        if saved['identity']!=read(old/'run.json') or len(saved['rank_rng'])!=30:raise ValueError('Parent checkpoint identity/RNG differ')
        assert saved['step']==read(old/'status.json')['step']
        old_config=read(old/'config.json')
        assert saved['identity']['experiment']==old_config
        new=BASE/'event';c=read(new/'config.json');identity=copy.deepcopy(saved['identity']);identity.update(experiment=c,source=C.source_identity())
        value=dict(saved,identity=identity);target=new/path.name
        atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(step=value['step'],sha256=sha(target)))
        v=checked(target)
        assert equal(saved['model'],v['model']) and equal(saved['optimizer'],v['optimizer']) and equal(saved['rank_rng'],v['rank_rng'])
        atomic_json(new/'run.json',identity,immutable=True)
        atomic_json(new/'status.json',dict(step=v['step'],budget=c['training']['steps'],complete=False))
        atomic_json(new/'MIGRATION.json',dict(success=True,parent_checkpoint=str(path),parent_sha256=sha(path),checkpoint=str(target),step=v['step'],model_optimizer_rng_bitwise_equal=True,data_schedule_unchanged=True,world=30,policy='three_quarters',process_memory_limit_gib=134,user_accepted_trial_peak_reserved_gib=128.13671875,at=time.time()))
        # Original queue remains stopped to prevent two simultaneous queues.
        atomic_json(PARENT/'SUPERSEDED_BY.json',dict(path=str(BASE),at=time.time()))
        cluster=read(new/'cluster.json')
        while not idle_snapshot(cluster)['all_idle']:
            atomic_json(BASE/'state.json',dict(stage='waiting_selected30_idle',at=time.time()));time.sleep(5)
        with (BASE/'controller.log').open('ab',buffering=0) as log:
            p=subprocess.Popen([cluster['python'],'-u',str(ROOT/'scripts/p2_whisper_recompute_profile30_20261106.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
        atomic_json(BASE/'LAUNCHED.json',dict(pid=p.pid,step=v['step'],at=time.time()))
        print('LAUNCHED',p.pid,'FROM',v['step'],flush=True)

if __name__=='__main__':
    try:run()
    except Exception as exc:
        BASE.mkdir(exist_ok=True);atomic_json(BASE/'MIGRATION_ERROR.json',dict(error=repr(exc),at=time.time()));raise
