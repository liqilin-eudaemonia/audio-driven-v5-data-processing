"""One bounded production-window check; rollback if the benefit is unclear."""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[1]
ROOT = W.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.j_annotation import read, write, acquire

NODE = 'node8'
cfg = read(W / 'config.json')
folder = W / 'performance_check/node8_cpu_balance'
owner = acquire(folder / 'run.lock', False)
if owner is None:
    raise SystemExit('CPU balance check already running')
if (folder / 'complete.json').exists():
    raise SystemExit('Completed check already exists; inspect instead of repeating')
old = Path(cfg['nodes'][NODE]['old_work'])
assert read(old / 'config.json')['nod_workers'] == 80
ssh = ['ssh', '-i', '/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ControlMaster=auto',
       '-o', 'ControlPersist=600', '-o', 'ControlPath=/tmp/jomni-guard-%C',
       'root@' + cfg['nodes'][NODE]['host']]


def measure(name):
    target = folder / name
    command = shlex.join(['python3', str(W / 'performance_check/observe.py'), NODE, str(target), '180'])
    subprocess.run(ssh + [command], check=True, timeout=240)
    return read(target / (NODE + '.json'))


def tune(workers):
    subprocess.run([sys.executable, str(W / 'performance_check/tune_nod_cpu.py'),
                    '--node', NODE, '--workers', str(workers)], check=True, timeout=45)


changed = False
try:
    before = measure('before')
    changed = True  # restore even if SSH disconnects after applying the change
    tune(48)
    code = f'ROOT={str(ROOT)!r}\nOLD={str(old)!r}\n' + '''
import sys,time,json
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts')
from listen_j_annotation_supervisor import alive
deadline=time.monotonic()+120
while time.monotonic()<deadline:
 s=json.loads((Path(OLD)/'supervisor.json').read_text())
 if not any(alive(s['children'].get(f'nod-{i}')) for i in range(48,80)):break
 time.sleep(5)
else:raise TimeoutError('CPU workers still draining; no forced kill')
'''
    subprocess.run(ssh + ['python3 -'], input=code, text=True, check=True, timeout=140)
    after = measure('after')
    gain = after['clips_per_second'] / before['clips_per_second'] - 1
    nod_ok = after['nod_clips_per_second'] >= 0.65 * before['nod_clips_per_second']
    keep = gain >= .10 and nod_ok
    if not keep:
        tune(80)
    result = dict(at=time.time(),node=NODE,before=before,after=after,
                  observed_relative_gain=gain,retained_workers=48 if keep else 80,
                  nod_rate_guard_passed=nod_ok,
                  limitation='Sequential production windows, not matched-input causal proof; continue monitoring.')
    for phase in ('before','after'):
        result[phase].pop('samples',None)
    write(folder / 'complete.json', result)
    print(json.dumps(result), flush=True)
except BaseException as exc:
    rollback_error = None
    if changed:
        try:
            tune(80)
        except Exception as restore:
            rollback_error = repr(restore)
    write(folder / 'error.json', dict(at=time.time(),error=repr(exc),rollback_error=rollback_error))
    raise
finally:
    owner.close()
