"""High-concurrency 24-GPU execution; preserves frozen J source and resumable queue."""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import uuid
import run_listen_pair_j_cluster as bounded
import run_listen_pair_j_full_cluster as full
from listen_pair_j_queue import lock, read, write, sha
from listen_pair_j_fast_queue import Queue
ROOT = bounded.ROOT
SCRIPT = Path(__file__).resolve()
key = full.key
def worker(args):
    execution = read(args.work / 'gpu_execution_node17_boost.json')
    gpu_ids = execution['gpu_ids'][args.node]
    device = 'cuda:0' if gpu_ids else 'cpu'
    visible = str(gpu_ids[args.slot % len(gpu_ids)]) if gpu_ids else ''
    os.environ['J_COMPUTE_DEVICE'] = device
    os.environ.update(CUDA_VISIBLE_DEVICES=visible, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1')
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0): raise RuntimeError('THP disable failed')
    allowed = sorted(os.sched_getaffinity(0)); upper = len(allowed) - 2 * (args.slot % (len(allowed) // 2))
    affinity = allowed[upper - 2:upper]
    if len(affinity) != 2: raise ValueError('Insufficient CPU affinity slots')
    os.sched_setaffinity(0, affinity); os.nice(10)
    work = args.work.resolve(); m = bounded.verify(work)
    node = next(n for n in m['nodes'] if n['name'] == args.node)
    if socket.gethostname() != node['hostname']: raise ValueError('Wrong node')
    for path, digest in execution['code_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Execution source changed: ' + path)
    k = key(args.node, args.slot); slot_lock = lock(work / 'workers' / (k + '.slot.lock'), False)
    if slot_lock is None: raise RuntimeError('Worker slot already held')
    stopped = threading.Event(); finished = threading.Event(); mutex = threading.Lock()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    state = dict(node=args.node, slot=args.slot, session=args.session, pid=os.getpid(),
                 affinity=affinity, torch_threads=2, blas_threads=1, device=device, gpu=visible, stage='starting',
                 pair=None, completed_pairs=0, completed_clips=0, reused=0, fresh=0, failures=0)
    def progress(**kw):
        with mutex:
            state.update(kw, heartbeat=time.time(), cpu_seconds=time.process_time())
            write(work / 'workers' / (k + '.status.json'), state)
    def check():
        c = read(work / 'control.json'); now = time.time()
        stop_file = work / 'stop.request.json'
        operator_stop = stop_file.exists() and read(stop_file)['at'] >= c.get('started', 0)
        if (stopped.is_set() or operator_stop or c.get('stop') or c.get('session') != args.session or
                args.slot >= c.get('enabled', {}).get(args.node, 0) or
                now - c.get('heartbeat', 0) > m['config']['lease_seconds']):
            from listen_pair_j_runtime import Paused
            raise Paused('Stopped, disabled, or controller lease lost')
    def pulse():
        while not finished.wait(10): progress()
    threading.Thread(target=pulse, daemon=True).start(); progress()
    from listen_pair_j_gpu_full_runtime import FullEngine
    from listen_pair_j_runtime import Paused
    handle = None
    try:
        check(); engine = FullEngine(work, check, progress); queue = Queue(work)
        while True:
            check(); claim_started = time.monotonic(); claim = queue.claim(args.node)
            claim_seconds = time.monotonic() - claim_started
            if claim is None:
                progress(stage='idle', pair=None); time.sleep(5); continue
            pair, handle = claim
            try:
                result = engine.process(queue.items[pair]); queue.finish(pair)
                counts = result.get('counts', {})
                progress(stage='between_pairs', pair=None, last_claim_seconds=claim_seconds, completed_pairs=state['completed_pairs'] + 1,
                         completed_clips=state['completed_clips'] + len(result['clips']),
                         reused=state['reused'] + counts.get('reused', 0) + counts.get('reused_completed_result_attempts', 0),
                         fresh=state['fresh'] + counts.get('fresh', 0))
            except Paused: raise
            except OSError as e:
                raise Paused('Transient filesystem failure; resume checkpoint: ' + repr(e)) from e
            except Exception as e:
                torch = sys.modules.get('torch')
                if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    progress(stage='gpu_memory_backoff', pair=pair)
                    stopped.wait(3)
                    continue
                import traceback
                traceback.print_exc(); queue.finish(pair, repr(e))
                write(work / 'errors' / (pair + '.json'), dict(error=repr(e), node=args.node, slot=args.slot, at=time.time()))
                progress(stage='pair_failed', pair=None, failures=state['failures'] + 1)
            finally:
                handle.close(); handle = None
    except Paused as e: progress(stage='paused', reason=str(e))
    except BaseException as e:
        progress(stage='error', error=repr(e)); raise
    finally:
        finished.set()
        if handle: handle.close()
        slot_lock.close()


def controller(args):
    work = args.work.resolve(); m = bounded.verify(work)
    owner = lock(work / 'controller.lock', False)
    if owner is None: raise RuntimeError('Controller already running')
    execution = read(work / 'gpu_execution_node17_boost.json')
    for path, digest in execution['code_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Execution source changed: ' + path)
    previous = read(work/'control.json')
    if args.adopt:
        if previous.get('stop'): raise ValueError('Cannot adopt stopped session')
        session = previous['session']; began = previous['started']
        launches = read(work/'gpu_launches.json')
        if any(v['node'] != 'node17' for v in launches.values()): raise ValueError('Only node17 may be adopted')
    else:
        session = uuid.uuid4().hex[:12]; began = time.time(); launches = {}
    enabled = execution['workers_per_node']; events = []; points = []
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    full.SCRIPT = SCRIPT
    def control(stop=False):
        write(work / 'control.json', dict(session=session, started=began, heartbeat=time.time(),
              enabled=enabled, stop=stop, execution='node17-8gpu-fast-queue', training_guard=False))
    def parallel(fn, values):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn, n): n['name'] for n in values}
            for f in concurrent.futures.as_completed(futures):
                name = futures[f]
                try: yield name, f.result()
                except Exception as e:
                    events.append(dict(at=time.time(), node=name, error=repr(e)))
                    print(name, repr(e), flush=True)
    def old_cleanup(n):
        old = {p.name.removesuffix('.launch.json'):read(p) for p in (work/'workers').glob(n['name']+'.w*.launch.json')}
        # These old workers observe the new session and stop at their next checkpoint.
        return len(old)
    control()
    write(work/'gpu_controller.launch.json',dict(pid=os.getpid(),session=session,started=began,hostname=socket.gethostname(),script=str(SCRIPT),
          start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
    base = read(work/'queue.json')['completed']
    try:
        # Small launch waves avoid simultaneous model-import and filesystem spikes.
        for offset in range(0,max(enabled.values()),8):
            if stopped.is_set(): break
            control()
            def start(n):
                return full.start_slots(work,m,n,[i for i in range(offset,min(offset+8,enabled[n['name']])) if key(n['name'],i) not in launches],session)
            for name, value in parallel(start,m['nodes']): launches.update(value)
            write(work/'gpu_launches.json',launches)
            print(json.dumps(dict(event='launch_wave',offset=offset,launched=len(launches))),flush=True)
        last_check = 0
        while not stopped.is_set():
            now=time.time()
            stop_path=work/'stop.request.json'
            if stop_path.exists() and read(stop_path)['at']>=began: break
            new_targets = read(work/'gpu_execution_node17_boost.json')['workers_per_node']
            if any(v < 0 or v > 160 for v in new_targets.values()): raise ValueError('Invalid concurrency')
            enabled.update(new_targets)
            control(); q=read(work/'queue.json')
            points.append((now,q['completed'])); points=[p for p in points if p[0]>=now-600]
            dt=now-points[0][0]
            rate=(q['completed']-points[0][1])/dt if dt>30 else None
            remaining=len(m['items'])-q['completed']-len(q['failed'])
            active={n['name']:0 for n in m['nodes']}; stages={}; fresh=reused=0; gpu_active=set()
            for k,v in launches.items():
                path=work/'workers'/(k+'.status.json')
                if not path.exists(): continue
                try: s=read(path)
                except (OSError, ValueError): continue
                if s.get('session')!=session: continue
                fresh+=s.get('fresh',0); reused+=s.get('reused',0)
                stage=s.get('stage'); stages[stage]=stages.get(stage,0)+1
                if now-s.get('heartbeat',0)<90 and stage not in ('paused','error'):
                    active[v['node']]+=1
                    if s.get('device')=='cuda:0': gpu_active.add((v['node'],s['gpu']))
            status=dict(session=session,started=began,at=now,state='running',total=len(m['items']),
                completed=q['completed'],failed=len(q['failed']),remaining=remaining,inflight=len(q['inflight']),
                enabled=enabled,active=active,stages=stages,gpu_count=len(gpu_active),configured_gpus=sum(len(execution['gpu_ids'][n]) for n in enabled if enabled[n]),
                sources_per_second=rate,eta=now+remaining/rate if rate else None,deadline=m['deadline'],
                deadline_at_risk=(remaining/rate>m['deadline']-now) if rate else None,
                completed_this_session=q['completed']-base,fresh_qc=fresh,reused_qc=reused,
                execution='node17-8gpu-fast-queue',training_guard=False,events=events[-30:])
            write(work/'status.json',status)
            with (work/'gpu_telemetry.jsonl').open('a') as f: f.write(json.dumps(status)+'\n')
            if remaining<=0:
                status['state']='completed_with_failures' if q['failed'] else 'completed'
                status['ended']=time.time(); write(work/'status.json',status); break
            if now-last_check>=60:
                code="""import json,sys
from pathlib import Path
p=json.load(sys.stdin);out={}
for k,v in p.items():
 try:
  r=Path('/proc')/str(v['pid']);fields=(r/'stat').read_text().rsplit(')',1)[1].split();args=(r/'cmdline').read_bytes().decode().split('\\0')
  out[k]=fields[0]!='Z' and fields[19]==v['start_ticks'] and v['session'] in args
 except FileNotFoundError:out[k]=False
print(json.dumps(out))"""
                def inspect(n):
                    if enabled[n['name']] == 0: return {}
                    values={k:v for k,v in launches.items() if v['node']==n['name']}
                    alive=bounded.remote(n,m['config'],code,values)
                    slots=[i for i in range(enabled[n['name']]) if not alive.get(key(n['name'],i),False)]
                    return full.start_slots(work,m,n,slots,session)
                for name,value in parallel(inspect,m['nodes']): launches.update(value)
                write(work/'gpu_launches.json',launches); last_check=now
            stopped.wait(15)
    finally:
        control(stop=True); owner.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['controller','worker'])
    parser.add_argument('--work',type=Path,required=True); parser.add_argument('--node')
    parser.add_argument('--slot',type=int); parser.add_argument('--session')
    parser.add_argument('--adopt', action='store_true')
    args=parser.parse_args()
    if args.command=='worker':
        worker(args)
    else:
        while True:
            try:
                controller(args); break
            except (OSError, ValueError) as e:
                import traceback
                traceback.print_exc(); args.adopt=False; time.sleep(5)
