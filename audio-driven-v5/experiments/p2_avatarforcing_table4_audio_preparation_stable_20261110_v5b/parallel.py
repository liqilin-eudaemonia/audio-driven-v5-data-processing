"""Shard the unchanged single-video IIANet helper; validate exact ROI/wave parity."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save,sha


ALIASES=['python_deps','vendor','released_checkpoints','issue6_bundle','IIANET_SOURCE_MANIFEST.json','IIANET_MODEL_ASSEMBLY_AUDIT.json']


def helper():
    spec=importlib.util.spec_from_file_location('fixed_iianet_helper',WORK/'separate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_shard(name,samples):
    folder=WORK/'execution'/name;folder.mkdir(parents=True,exist_ok=True)
    for alias in ALIASES:
        target=folder/alias
        if not target.exists():target.symlink_to(WORK/alias,target_is_directory=(WORK/alias).is_dir())
    p=read(WORK/'PROTOCOL.json');files={}
    for key,value in p['source_and_weights'].items():
        q=Path(key)
        if any(q.is_relative_to(WORK/a) for a in ['python_deps','released_checkpoints','issue6_bundle']):q=folder/q.relative_to(WORK)
        files[str(q)]=value
    child=dict(p,samples=samples,source_and_weights=files,parent_protocol_sha256=sha(WORK/'PROTOCOL.json'),execution_shard=name,
        mathematical_helper_unchanged=True,executor_sha256=sha(__file__))
    dest=folder/'PROTOCOL.json'
    if dest.exists():assert read(dest)==child
    else:save(dest,child)
    return folder


def prepare():
    parent=read(WORK/'PROTOCOL.json');assert read(WORK/'IIANET_MODEL_ASSEMBLY_AUDIT.json')['passed']
    plan_path=WORK/'PARALLEL_PROTOCOL.json'
    plan=dict(schema='same-IIANet-fixed-frontend-independent-video-shards-v2',script_sha256=sha(__file__),
        helper_sha256=sha(WORK/'separate.py'),guard_sha256=sha(ROOT/'scripts/p2_avatarforcing_table4_generate.py'),
        parent_protocol_sha256=sha(WORK/'PROTOCOL.json'),workers=4,allocator_fraction=.05,context_budget_mib=1024,reserve_mib=16384,
        parity_gate='same canonical publishedCNN speaker: ROI array exact, raw waveform max_abs<=1e-6; no tolerance relaxation',
        partition='All43 pairs, both roles; deterministic duration balancing; no reuse of previous autotuned mouth/audio outputs',
        no_network_math_batching_or_weight_changes=True)
    if plan_path.exists():assert read(plan_path)==plan
    else:save(plan_path,plan)
    first=next(s for s in parent['samples'] if s['id']==parent['canary_ids'][0]);one=dict(first,roles=[r for r in first['roles'] if r['role']=='speaker'])
    make_shard('pilot_a',[one]);make_shard('pilot_b',[one])
    groups=[[] for _ in range(4)];loads=[0]*4;cached=set()
    for sample in sorted(parent['samples'],key=lambda s:(-s['samples'],s['id'])):
        i=min(range(4),key=lambda j:(loads[j],j));groups[i].append(sample)
        loads[i]+=0 if sample['id'] in cached else sample['samples']*2
    assert sum(map(len,groups))==43 and len({s['id'] for g in groups for s in g})==43
    for i,samples in enumerate(groups):
        samples=sorted(samples,key=lambda s:next(j for j,v in enumerate(parent['samples']) if v['id']==s['id']))
        folder=make_shard(f'worker_{i}',samples)
        for s in samples:
            if s['id'] not in cached:continue
            for role in s['roles']:
                source=WORK/'separated'/s['id']/role['role']/'complete.json';r=read(source)
                assert r['protocol_sha256']==sha(WORK/'PROTOCOL.json')
                dest=folder/'separated'/s['id']/role['role'];dest.mkdir(parents=True,exist_ok=True)
                receipt=dest/'complete.json'
                if receipt.exists():continue
                reuse=dict(r,computed_protocol_sha256=r['protocol_sha256'],protocol_sha256=sha(folder/'PROTOCOL.json'),
                    reused_without_recomputation=True,parent_receipt=str(source),parent_receipt_sha256=sha(source))
                for filename in ['target.wav','estimate.npy']:
                    if not (dest/filename).exists():(dest/filename).symlink_to(source.parent/filename)
                history=folder/'runtime_history';history.mkdir(exist_ok=True)
                (history/(r['runtime_sha256']+'.json')).write_bytes((WORK/'runtime_history'/(r['runtime_sha256']+'.json')).read_bytes())
                save(receipt,reuse)
    save(WORK/'PARALLEL_ASSIGNMENT.json',dict(groups=[[s['id'] for s in g] for g in groups],estimated_remaining_samples=loads,all_pairs=43,all_streams=86,reused_parent_streams=2*len(cached)))
    return plan


def execute(name):
    module=helper();module.WORK=WORK/'execution'/name
    module.run('all')


def parity_check():
    import numpy as np
    parent=read(WORK/'PROTOCOL.json');uid=parent['canary_ids'][0]
    base=WORK/'execution/pilot_a/separated'/uid/'speaker';trial=WORK/'execution/pilot_b/separated'/uid/'speaker'
    r0=read(base/'complete.json');r1=read(trial/'complete.json')
    assert sha(WORK/'separate.py')==read(WORK/'PARALLEL_PROTOCOL.json')['helper_sha256']
    for r in [r0,r1]:
        for path,digest in r['files'].items():assert sha(path)==digest
    a=np.load(base/'estimate.npy');b=np.load(trial/'estimate.npy');error=float(np.abs(a-b).max())
    mouth_a=np.load(base/'mouthroi/speaker1.npz')['data'];mouth_b=np.load(trial/'mouthroi/speaker1.npz')['data'];same=bool(np.array_equal(mouth_a,mouth_b))
    assert error<=1e-6 and same,(error,same)
    save(WORK/'PARALLEL_PARITY.json',dict(passed=True,id=uid,role='speaker',raw_audio_max_abs=error,mouth_ROI_exact=same,
        source_receipt_sha256=sha(base/'complete.json'),parallel_receipt_sha256=sha(trial/'complete.json'),executor_protocol_sha256=sha(WORK/'PARALLEL_PROTOCOL.json')))
    print('IIANet shard parity',error,same,flush=True)


def gpu():
    row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
    text=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True);used={}
    for line in text.splitlines():
        values=[x.strip() for x in line.split(',')]
        if len(values)==2 and all(x.isdigit() for x in values):used[int(values[0])]=int(values[1])
    return int(row[1]),int(row[2]),used


def collect():
    import numpy as np
    import soundfile as sf
    measured=helper()
    parent=read(WORK/'PROTOCOL.json');records=[]
    for name in [f'worker_{i}' for i in range(4)]:
        folder=WORK/'execution'/name;p=read(folder/'PROTOCOL.json');marker=read(folder/'SEPARATION_COMPLETE.json');assert marker['passed']
        assert marker['protocol_sha256']==sha(folder/'PROTOCOL.json')
        for sample in p['samples']:
            for role in sample['roles']:
                receipt=folder/'separated'/sample['id']/role['role']/'complete.json';r=read(receipt)
                assert r['id']==sample['id'] and r['role']==role['role']
                assert r['mix_sha256']==sample['mix_sha256'] and sha(sample['mix'])==sample['mix_sha256']
                assert r['video_sha256']==role['video_sha256'] and sha(role['video'])==role['video_sha256']
                assert r['protocol_sha256']==sha(folder/'PROTOCOL.json')
                for path,digest in r['files'].items():assert sha(path)==digest
                assert r['raw_audio_metrics']['samples']==r['saved_audio_metrics']['samples']==sample['samples']
                raw_path=next(k for k in r['files'] if k.endswith('/estimate.npy'))
                wav_path=next(k for k in r['files'] if k.endswith('/target.wav'))
                raw=np.load(raw_path);wav,sr=sf.read(wav_path,dtype='float32');mix,msr=sf.read(sample['mix'],dtype='float32')
                assert sr==msr==16000 and raw.shape==wav.shape==mix.shape==(sample['samples'],)
                assert np.isfinite(raw).all() and np.isfinite(wav).all()
                for name,values in [('raw_audio_metrics',raw),('saved_audio_metrics',wav)]:
                    independently_measured=measured.metrics(mix,values)
                    for key,value in independently_measured.items():
                        expected=r[name][key]
                        assert (value is None and expected is None) or (value is not None and expected is not None and abs(value-expected)<1e-8),(sample['id'],role['role'],key)
                assert sha(folder/'runtime_history'/(r['runtime_sha256']+'.json'))==r['runtime_sha256']
                if r.get('reused_without_recomputation'):assert sha(r['parent_receipt'])==r['parent_receipt_sha256']
                records.append(dict(id=r['id'],role=r['role'],receipt=str(receipt),receipt_sha256=sha(receipt),
                    target_wav=next(k for k in r['files'] if k.endswith('/target.wav')),raw_audio_metrics=r['raw_audio_metrics'],saved_audio_metrics=r['saved_audio_metrics']))
    assert len(records)==86 and len({(r['id'],r['role']) for r in records})==86
    assert {r['id'] for r in records}=={s['id'] for s in parent['samples']}
    save(WORK/'FULL_SEPARATION_COMPLETE.json',dict(passed=True,pairs=43,streams=86,records=records,
        parent_protocol_sha256=sha(WORK/'PROTOCOL.json'),parallel_protocol_sha256=sha(WORK/'PARALLEL_PROTOCOL.json'),parity_sha256=sha(WORK/'PARALLEL_PARITY.json')))


def manage():
    plan=prepare();assert read(WORK/'PARALLEL_PARITY.json')['passed']
    children=[];launched=set()
    with (WORK/'parallel_manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                for child in children:
                    if child.poll() not in [None,0]:raise RuntimeError(f'Shard{child.pid} exited{child.returncode}; preserve output and inspect log')
                done=sum((WORK/'execution'/f'worker_{i}'/'SEPARATION_COMPLETE.json').exists() for i in range(4))
                live=[c for c in children if c.poll() is None]
                if done==4 and not live:collect();save(WORK/'PARALLEL_STATUS.json',dict(stage='complete',pairs=43,streams=86,pid=os.getpid(),updated_unix=time.time()));return
                free,total,used=gpu();budget=math.ceil(total*.05)+1024;required=16384+sum(max(0,budget-used.get(c.pid,0)) for c in live)+budget
                if len(launched)<4 and free>=max(32768,required):
                    i=next(j for j in range(4) if j not in launched);name=f'worker_{i}'
                    with (WORK/'execution'/name/'run.log').open('a') as log:child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'execute','--name',name],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                    children.append(child);launched.add(i);live.append(child)
                completed=len(list((WORK/'execution').glob('worker_*/separated/*/*/complete.json')))
                save(WORK/'PARALLEL_STATUS.json',dict(stage='separating_parallel',pid=os.getpid(),completed=completed,total=86,live_pids=[c.pid for c in live],free_mib=free,required_to_add_mib=max(32768,required),updated_unix=time.time()))
                time.sleep(5)
        except BaseException:
            for child in children:
                if child.poll() is None:child.terminate()
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute','parity-check','manage','collect']);p.add_argument('--name');a=p.parse_args()
    if a.action=='execute':execute(a.name)
    else:{'prepare':prepare,'parity-check':parity_check,'manage':manage,'collect':collect}[a.action]()
