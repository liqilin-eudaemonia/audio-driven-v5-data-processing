"""Isolated additional ViCo baselines; sealed previous results remain read-only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'experiments/p2_vico_additional_baselines_20261107'
DATA = Path('/public/0_DATA/2_Avatar/ViCo/dataset')
SPLIT = ROOT / 'experiments/common_method_comparison_20261106/l2l_retrained_v2/split.json'
BFM = ROOT / 'experiments/p2_vico_motion8000_20261106'
DECA = ROOT / 'experiments/p2_compare_vico43_20261106/learning2listen'
GPU_UUID = 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def gpu_guard(free_gib=12):
    """Use only assigned GPU7; preserve existing processes and require headroom."""
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical=='7', 'Latest user allocation is node11 GPU7 only'
    row = subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert row[0].strip() == GPU_UUID, row
    required=max(32,free_gib)
    assert float(row[1]) >= required * 1024, row
    sharing=[]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    for line in apps:
        if ',' not in line:continue
        uid,pid=[s.strip() for s in line.split(',')]
        if uid!=row[0].strip() or int(pid)==os.getpid():continue
        proc=Path('/proc')/pid/'cmdline'
        if proc.exists():
            sharing.append(dict(pid=int(pid),owned_by_this_experiment=b'vico_more_' in proc.read_bytes()))
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(host=os.uname().nodename, uuid=row[0].strip(), free_mib=float(row[1]), pid=os.getpid(),
        required_free_gib=required,other_processes=sharing,policy='GPU7 only; preserve existing processes; capacity based admission')


def remote(command, background=False, log=None):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from listen_cluster import transport
    cfg = read(ROOT / 'experiments/p2_current500h_av_only31_20261107/cluster.json')
    node = next(n for n in cfg['nodes'] if n['name'] == 'node11')
    for attempt in range(6):
        result = subprocess.run(transport(cfg, node, command), capture_output=True, text=True)
        if result.returncode != 255:
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            return result.stdout
        time.sleep(2)
    raise RuntimeError(result.stderr)


def launch_node11(name, python, script, args, gpu=7):
    """Detached supervisor records the actual child exit, rather than SSH status."""
    directory = WORK / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=True)
    command = ['env', 'CUDA_VISIBLE_DEVICES='+str(gpu), 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
        'TOKENIZERS_PARALLELISM=false', 'PYTHONUNBUFFERED=1', str(python), str(script), *args]
    runner = ('import subprocess,json,os,time,fcntl; from pathlib import Path; '
        f'd=Path({str(directory)!r}); lock=open(d/"run.lock","a"); '
        'fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); '
        'f=open(d/"run.log","a"); '
        f'p=subprocess.Popen({command!r},cwd={str(ROOT)!r},stdout=f,stderr=subprocess.STDOUT); '
        '(d/"running.json").write_text(json.dumps(dict(pid=p.pid,supervisor_pid=os.getpid(),started=time.time()))); '
        'code=p.wait(); (d/"exit.json").write_text(json.dumps(dict(returncode=code,ended=time.time())))')
    launcher = ('import subprocess; '
        f'p=subprocess.Popen(["python","-c",{runner!r}],cwd={str(ROOT)!r},'
        f'stdout=open({str(directory / "supervisor.log")!r},"a"),stderr=subprocess.STDOUT,start_new_session=True); '
        'print(p.pid)')
    return remote(['python', '-c', launcher])


def test_samples():
    return read(BFM / 'protocol.json')['samples']


def source_tree(path):
    # Only explicit small vendor tree, never scan the experiments root.
    return {str(p): sha(p) for p in sorted(Path(path).rglob('*.py')) if '.git' not in p.parts}


def prepare():
    split = read(SPLIT)
    assert len(split['train']) == 351 and len(split['validation']) == 46
    p = dict(schema='vico-additional-baselines-v1', seed=137,
        delivery_utc='2026-11-08T04:00:00Z', model_freeze_utc='2026-11-08T00:00:00Z',
        split=str(SPLIT), split_sha256=sha(SPLIT), train=351, validation=46, test=43,
        repositories=read(WORK / 'REPOSITORIES.json'), downloads=read(WORK / 'DOWNLOADS.json'),
        source={k: source_tree(WORK / 'vendor' / k) for k in ['dualtalk', 'rlhg', 'listenformer', 'pch']},
        constraints=['No OVI or GLARE this round', 'Existing 31-GPU training unchanged',
            'Only node11 GPU7 until actual release', 'Train-only normalization; val-only checkpoint selection',
            'Source audio/motion and listener t0 only; no future listener input',
            'Incomplete training is diagnostic, never a formal baseline'])
    dest = WORK / 'protocol.json'
    if dest.exists():
        assert read(dest) == p, 'Protocol changed; use a new experiment root'
    else:
        save(dest, p)
    return p
