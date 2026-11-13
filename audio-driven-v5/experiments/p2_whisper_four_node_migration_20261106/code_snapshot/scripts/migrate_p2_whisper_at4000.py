"""Pause at complete4000, replace transport topology, resume same31-card state."""
import json,time,os,sys,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_seamless200_whisper_eval import idle_snapshot
W=ROOT/'experiments/p2_whisper_four_node_migration_20261106';Q=ROOT/'experiments/p2_whisper_single_aux31_20261106';O=ROOT/'experiments/p2_whisper_event_20261106'


def live(p):
    try:
        s=Path('/proc',str(p['pid']),'stat').read_text().split();return s[21]==p['start_ticks'] and s[2]!='Z'
    except FileNotFoundError:return False


def state(stage,**kw):
    v=dict(stage=stage,at=time.time(),**kw);atomic_json(W/'state.json',v);print(json.dumps(v),flush=True)


def prepare():
    W.mkdir(exist_ok=True)
    from p2_whisper_topology31 import layout
    for arm in ['event','feature']:
        out=ROOT/f'experiments/p2_whisper_{arm}_20261106';c=read(out/'cluster.json')
        atomic_json(W/f'{arm}_old_cluster.json',c,immutable=True)
        n=c['nodes'][0];n.update(name='node11',node='node11',host='172.29.189.119',ssh_key='/private/ssh_keys/root_dedicated-developjob-js-infer-machine-ws3qe_ed25519')
        c['training_script']=str(ROOT/'scripts/p2_whisper_topology31.py');mapping=layout(c)
        atomic_json(W/f'{arm}_new_cluster.json',c,immutable=True)
    atomic_json(W/'PLAN.json',dict(target_step=4000,world=31,mapping=mapping,
        data_model_loss_batch_sample_schedule_unchanged=True,rank_rng_mapping='logical ranks unchanged; old node17 ranks0..7 move to node11',
        reserved_gpu='node8:7',old_nodes=['node17','node12','node4','node8'],new_nodes=['node11','node12','node4','node8']),immutable=True)


def run():
    prepare()
    with (W/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (W/'NEW_QUEUE_STARTED.json').exists():raise ValueError('Migration already launched; do not repeat')
        if (O/'STOP').exists() or (Q/'STOP').exists():raise ValueError('Existing user STOP; do not override')
        old=read(Q/'controller.json');token=json.dumps(dict(owner='four_node_migration4000',pid=os.getpid(),at=time.time()))
        from p2_speaker_whisper_after_annotation import process_record
        atomic_json(W/'controller.json',process_record());state('armed_waiting_step4000',old_controller=old)
        while True:
            step=read(O/'status.json')['step']
            if step>=4000:break
            if not live(old):raise RuntimeError('Original controller disappeared before4000')
            if (O/'STOP').exists() or (Q/'STOP').exists():state('cancelled_by_user_stop');return
            time.sleep(.1)
        with (O/'STOP').open('x') as f:f.write(token)
        state('waiting_complete4000_and_old_exit',observed_step=step)
        while live(old):time.sleep(2)
        marker=O/'step_004000.complete.json';receipt=read(marker)
        if receipt['step']!=4000 or read(O/'status.json')['step']!=4000:raise ValueError('Not an exact4000 stop; inspect before resume')
        if sha(O/'step_004000.pt')!=receipt['sha256']:raise ValueError('4000 checkpoint checksum differs')
        atomic_json(W/'STOP4000_COMPLETE.json',dict(success=True,checkpoint=receipt,old_controller=old,at=time.time()))
        # Only stopped node launchers ever consume the topology change.
        for arm in ['event','feature']:
            atomic_json(ROOT/f'experiments/p2_whisper_{arm}_20261106/cluster.json',read(W/f'{arm}_new_cluster.json'))
        new=read(W/'event_new_cluster.json');state('waiting_new_four_idle')
        while True:
            if (Q/'STOP').exists() or not (O/'STOP').exists() or (O/'STOP').read_text()!=token:
                state('cancelled_by_user_stop');return
            resources=idle_snapshot(new);atomic_json(W/'resources.json',resources)
            if resources['all_idle']:break
            time.sleep(15)
        # Keep model/optimizer/RNG and all checkpoint-bound Python files unchanged.
        (O/'STOP').unlink()
        with (Q/'controller.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([new['python'],'-u',str(ROOT/'scripts/queue_p2_whisper_new_four31.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
        atomic_json(W/'NEW_QUEUE_STARTED.json',dict(pid=child.pid,at=time.time(),resume_step=4000,nodes=new['nodes']))
        state('new_queue_started',pid=child.pid)
        # Verify actual updates; a state file alone is not successful migration.
        deadline=time.time()+1800
        while time.time()<deadline:
            s=read(O/'status.json');queue=read(Q/'state.json')
            if s['step']>4000 and queue.get('resume')==str(O/'step_004000.pt'):
                logs=Path(queue['logs'])/'ranks';receipts=[read(p) for p in logs.glob('*_workers.json')]
                if len(receipts)==4 and sum(len(r['pids']) for r in receipts)==31 and {r['node'] for r in receipts}=={'node11','node12','node4','node8'}:
                    atomic_json(W/'RESUME_VERIFIED.json',dict(success=True,step=s['step'],queue=queue,workers=receipts,at=time.time()))
                    state('resume_verified',step=s['step']);return
            if child.poll() is not None:raise RuntimeError('New queue exited; inspect logs')
            time.sleep(5)
        raise TimeoutError('New31-rank resume not verified in30minutes')

if __name__=='__main__':
    try:run()
    except Exception as exc:
        W.mkdir(exist_ok=True);atomic_json(W/'ERROR.json',dict(error=repr(exc),at=time.time()));raise
