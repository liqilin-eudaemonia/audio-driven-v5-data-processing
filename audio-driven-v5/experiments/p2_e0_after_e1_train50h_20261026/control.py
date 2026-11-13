#!/usr/bin/env python3
"""Local control helpers for the explicitly authorized, paired E0 successor."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TRAIN=ROOT/'experiments/p2_e1_train50h_lora_tb_20261026'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
SEED=20261023
BUDGET=1268
E1=TRAIN/f'runs/{SEED}/E1'
E0=TRAIN/f'runs/{SEED}/E0'
HOSTS=('172.29.252.11','172.29.15.91','172.29.225.51')
HOSTNAMES=('dedicated-developjob-lq-dev-machine-1nkpf','dedicated-developjob-lq-dev-machine-2-taaft','dedicated-developjob-lq-dev-machine-3-zhw7q')
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'


def now():return datetime.now(timezone.utc).isoformat()

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def verify_release():
    release=read(OUT/'release.json')
    if release['arm']!='E0' or release['seed']!=SEED or release['steps']!=BUDGET:raise ValueError('Unexpected successor')
    for name,expected in release['files'].items():
        if sha(name)!=expected:raise ValueError('Frozen launch input changed: '+name)
    c=read(TRAIN/'config.json')
    if c['training']['feature_loss_enabled'] is not False:raise ValueError('Feature loss must be disabled')
    if c['training']['global_batch']!=24 or c['training']['steps']!=BUDGET:raise ValueError('Paired budget differs')
    if c['training']['text_lora']['enabled'] is not True:raise ValueError('Paired LoRA must remain enabled')
    if Path(c['output']).resolve()!=TRAIN:raise ValueError('Must reuse exact paired config/output')
    return release


def latest_step(run):
    p=Path(run)/'metrics.jsonl'
    if not p.exists():return 0
    with p.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-20000));lines=f.read().decode().splitlines()
    for line in reversed(lines):
        try:return int(json.loads(line)['step'])
        except (ValueError,KeyError):continue
    return 0


def completed_gate(run,nodes,budget=BUDGET):
    """Pure state gate; a checkpoint marker alone does not authorize succession."""
    if any(n.get('state')=='failed' or n.get('returncode',0)!=0 for n in nodes):
        raise RuntimeError('A training node failed')
    if len(nodes)!=3 or not all(n.get('state')=='finished' and n.get('returncode')==0 for n in nodes):return False
    if run is None:raise RuntimeError('All nodes exited but final training status is missing')
    if run.get('step')!=budget or run.get('budget')!=budget or run.get('stopped') is not False:
        raise RuntimeError('Training did not complete its entire planned budget')
    return True


def verify_checkpoint(run):
    checkpoint=Path(run)/f'step_{BUDGET:06d}.pt'
    marker=read(checkpoint.with_suffix('.complete.json'))
    if marker['step']!=BUDGET or sha(checkpoint)!=marker['sha256']:raise ValueError('Final checkpoint incomplete/corrupt')
    return dict(path=str(checkpoint),sha256=marker['sha256'],bytes=checkpoint.stat().st_size)


def verify_pair():
    first=read(E1/'run.json');second=read(E0/'run.json')
    if first['arm']!='E1' or second['arm']!='E0':raise ValueError('Wrong experiment arms')
    differences=[k for k in set(first)|set(second) if k!='arm' and first.get(k)!=second.get(k)]
    if differences:raise ValueError('E0/E1 run identities differ: '+', '.join(differences))
    return dict(status='passed',identical_except_arm=True,seed=SEED,budget=BUDGET,world=second['world'],
                initialization=second['initialization'],observations=second['observations'],
                targets=second['targets'],feature_loss_enabled=second['feature_loss_enabled'],checked_at=now())
