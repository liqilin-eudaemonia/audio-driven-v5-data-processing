"""Wait for current noWhisper completion and exit, then run the authorized full arm."""
from pathlib import Path
import fcntl, importlib.util, os, sys, threading, time, traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_runtime as R
from p2_listen.contracts import read, sha, atomic_json

PREVIOUS = R.W.parent/'p2_no_whisper_resume1760_20261112'
ABLATIONS = R.W.parent/'p2_all_lora_ablations28_20261111'

def alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1].split()[0] != 'Z'
    except (ProcessLookupError, FileNotFoundError): return False

def prerequisite():
    if not (PREVIOUS/'COMPLETE.json').exists(): return False
    result = read(PREVIOUS/'COMPLETE.json')
    assert result['success'] and result['step'] == 6643
    if alive(read(PREVIOUS/'CONTROLLER.json')['pid']): return False
    accepted = read(ABLATIONS/'no_whisper_ACCEPTED.json')
    assert accepted['success'] and accepted['step'] == 6643
    return True

def stopped():
    return (R.W/'STOP').exists() or (R.OUT/'STOP').exists()

def monitor(done):
    while not done.wait(3600):
        status = read(R.OUT/'status.json') if (R.OUT/'status.json').exists() else None
        atomic_json(R.W/'HOURLY_LATEST.json', dict(at=time.time(),status=status))

def run():
    with (R.W/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready = read(R.W/'READY.json')
        assert ready['source'] == R.source()
        assert ready['config_sha256'] == sha(R.OUT/'config.json')
        assert ready['cluster_sha256'] == sha(R.OUT/'cluster.json')
        atomic_json(R.W/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),target=7000,save_every=100))
        atomic_json(R.W/'STATE.json',dict(stage='waiting_for_current_no_whisper6643',at=time.time()))
        while not prerequisite():
            if stopped(): return
            if (PREVIOUS/'FAILURE.json').exists(): raise RuntimeError('Preceding training failed; no automatic skip')
            time.sleep(30)
        spec = importlib.util.spec_from_file_location('previous_recovery',R.W.parent/'p2_ablation28_guard9h_20261111/recover.py')
        recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recovery)
        with (ABLATIONS/'queue.lock').open('a') as previous_lock:
            fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not recovery.Q.resources():
                atomic_json(R.W/'STATE.json',dict(stage='waiting_for_selected28_gpus',at=time.time()))
                if stopped(): return
                time.sleep(30)
            if stopped(): return
            from listen_cluster import launch
            m = R.load()
            paths = sorted(R.OUT.glob('step_*.complete.json'),reverse=True)
            resume = paths[0].with_suffix('').with_suffix('.pt')
            checkpoint = m.checkpoint.checked(resume)
            assert checkpoint['identity'] == read(R.OUT/'run.json') and len(checkpoint['rank_rng']) == 28
            assert 5500 <= checkpoint['step'] <= 7000
            step = checkpoint['step']
            del checkpoint
            if step < 7000:
                cluster = read(R.OUT/'cluster.json')
                commands = [['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),
                             'PYTHONUNBUFFERED=1',*[k+'='+str(v) for k,v in cluster['network'].items()],
                             cluster['python'],'-u',str(R.W/'worker.py'),'node','--index',str(i),
                             '--resume',str(resume)] for i,n in enumerate(cluster['nodes'])]
                atomic_json(R.W/'COMMANDS.json', commands)
                atomic_json(R.W/'STATE.json',dict(stage='training',at=time.time(),resume=str(resume),target=7000))
                done = threading.Event()
                threading.Thread(target=monitor,args=(done,),daemon=True).start()
                try: launch(cluster,commands,R.W/('nodes_'+str(time.time_ns())))
                finally: done.set()
            final = R.OUT/'step_007000.pt'
            v = m.checkpoint.checked(final)
            assert v['step'] == 7000 and v['identity'] == read(R.OUT/'run.json') and len(v['rank_rng']) == 28
            assert all((R.OUT/f'step_{s:06d}.complete.json').exists() for s in range(5600,7001,100))
            atomic_json(R.W/'COMPLETE.json',dict(success=True,at=time.time(),step=7000,
                        checkpoint=str(final),sha256=read(final.with_suffix('.complete.json'))['sha256']))
            atomic_json(R.W/'STATE.json',dict(stage='complete',at=time.time(),step=7000))

if __name__ == '__main__':
    try: run()
    except BaseException as error:
        atomic_json(R.W/'FAILURE.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
