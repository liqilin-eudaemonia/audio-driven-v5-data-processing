"""Reserved-GPU7 queue: fixed nine extra draws, EMOCA, DIM best-of-ten readout.

Reuses the sealed SFT2752 conditions and original sampler/export math. No training,
no other GPUs, and no modification of original generation or feature directories.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
SOURCE=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
EMOCA_SOURCE=ROOT/'experiments/p2_vico_literature_protocol_20261108'
WORK=ROOT/'experiments/p2_vico_dim_best10_20261108'
SELF=Path(__file__).resolve()
NATIVE=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
METRIC=ROOT/'compare/.envs/vico_more/bin/python'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)


def link(source,dest):
    source=Path(source).resolve();dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve()!=source:raise ValueError('Wrong shared input: '+str(dest))
    elif dest.exists():raise ValueError('Refuse overwriting input: '+str(dest))
    else:dest.symlink_to(source,target_is_directory=source.is_dir())


def verify_cache(folder):
    marker=folder/'complete.json'
    if not marker.exists():raise ValueError('Missing committed input: '+str(folder))
    for rel,info in read(marker)['files'].items():
        if sha(folder/rel)!=info['sha256']:raise ValueError('Cached condition changed')


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    source=read(SOURCE/'protocol.json');ep=read(EMOCA_SOURCE/'protocol.json')
    ids=ep['ids'];items=read(SOURCE/'shards/00.json')
    assert source['seed']==137 and len(ids)==len(set(ids))==43
    assert {x['id'] for x in items}==set(ids)
    for path,h in source['execution_code'].items():assert sha(path)==h,path
    for item in items:
        folder=SOURCE/'conditions'/item['key']
        for part in ['qwen','text']:verify_cache(folder/part)
        verify_cache(SOURCE/'whisper'/item['key'])
    verify_cache(SOURCE/'negative/00')
    files=[SELF,ROOT/'scripts/p2_vico43_sft2752_gpu7.py',ROOT/'scripts/p2_vico43_combined8000_sft2752.py',
        ROOT/'scripts/p2_vico256_combined_eval.py',ROOT/'compare/deployment/vico_literature_emoca.py',
        ROOT/'compare/deployment/vico_literature_metrics.py']
    policy=dict(schema='vico-dim-best10-v1',model='combined8000_sft2752',ids=ids,seeds=list(range(137,147)),
        original_seed_reused=137,additional_outputs=387,
        source_complete_sha256=sha(SOURCE/'COMPLETE.json'),
        source_protocol_sha256=sha(SOURCE/'protocol.json'),
        emoca_source_complete_sha256=sha(EMOCA_SOURCE/'COMPLETE.json'),
        code={str(p):sha(p) for p in files},generation=source['generation'],
        selection='per_clip minimum raw full56 GT Frechet; one coherent candidate for every metric',
        score_frames='listener1:T vs speaker0:T-1; same valid mask across all10 candidates',
        conditioning='reuse exact sealed speaker-only conditions and listener first frame; no new planner samples',
        resource=dict(node='node11',physical_gpu=7,uuid=UUID,reserve_mib=16384,
            latent_free_mib=88000,latent_retry_free_mib=110000,decode_free_mib=60000,emoca_free_mib=32768),
        label='Oracle best-of10 under released DIM selection; not an inference-time deployable selection policy',
        exact_published_table_equivalence=False,author_contact_required=False)
    dest=WORK/'protocol.json'
    if dest.exists():
        if read(dest)!=policy:raise ValueError('Frozen best10 plan changed; use a new directory')
    else:save(dest,policy)
    # Each seed has an isolated output tree. Only immutable source caches are linked.
    for seed in policy['seeds'][1:]:
        work=WORK/'draws'/str(seed);work.mkdir(parents=True,exist_ok=True)
        for name in ['conditions','whisper','negative/00']:link(SOURCE/name,work/name)
        for name in ['INPUT_PREFLIGHT.json','shards/00.json']:
            target=work/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==sha(SOURCE/name)
            else:target.write_bytes((SOURCE/name).read_bytes())
        p=copy.deepcopy(source);p.update(seed=seed,render_gpus=[7],render_shards=1,
            authorized_scope='node11 physicalGPU7 only; fixed DIM best10 supplement',
            best10_protocol_sha256=sha(dest))
        if (work/'protocol.json').exists():assert read(work/'protocol.json')==p
        else:save(work/'protocol.json',p)
    save(WORK/'PREPARED.json',dict(passed=True,existing_outputs=43,new_outputs=387,
        total_candidates=430,seeds=policy['seeds'],condition_hashes_verified=True,at=time.time()))
    return policy


def verify():
    p=read(WORK/'protocol.json')
    for path,h in p['code'].items():
        if sha(path)!=h:raise ValueError('Queued code changed: '+path)
    assert sha(SOURCE/'protocol.json')==p['source_protocol_sha256']
    assert sha(SOURCE/'COMPLETE.json')==p['source_complete_sha256']
    assert sha(EMOCA_SOURCE/'COMPLETE.json')==p['emoca_source_complete_sha256']
    return p


def gpu_capacity(required):
    row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if row[0].strip()!=UUID:raise ValueError('Not authorized GPU7')
    free=int(row[1]);return free,free>=required


def stage(seed,kind,fraction):
    verify()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise ValueError('Worker must use only physicalGPU7')
    if kind in ['latent','decode']:
        import p2_vico43_sft2752_gpu7 as adapter
        work=WORK/'draws'/str(seed)
        adapter.WORK=work;adapter.original.WORK=work
        adapter.worker(kind,fraction)
    else:
        import vico_literature_emoca as emoca
        work=WORK/'draws'/str(seed)
        ework=work/'emoca';ework.mkdir(exist_ok=True)
        link(EMOCA_SOURCE/'assets',ework/'assets')
        for name in ['LICENSE_CONFIRMATION.json']:
            target=ework/name
            if not target.exists():target.write_bytes((EMOCA_SOURCE/name).read_bytes())
        original=read(EMOCA_SOURCE/'protocol.json');tasks=[]
        byid={x['id']:x for x in read(SOURCE/'shards/00.json')}
        for uid in original['ids']:
            path=work/'scoring/combined8000_sft2752'/(uid+'.listener.mp4')
            tasks.append(dict(id=uid,role='joint_sft2752',video=str(path),video_sha256=sha(path),
                expected_frames=byid[uid]['geometry']['gt_frames']))
        ep=copy.deepcopy(original);ep.update(tasks=tasks,selected_methods=['joint_sft2752'],seed=seed,
            best10_protocol_sha256=sha(WORK/'protocol.json'))
        if (ework/'protocol.json').exists():assert read(ework/'protocol.json')==ep
        else:save(ework/'protocol.json',ep)
        emoca.WORK=ework;emoca.run()


def call_stage(seed,kind,fraction):
    python=METRIC if kind=='emoca' else NATIVE
    work=WORK/'draws'/str(seed);log=work/f'{kind}_{fraction:.2f}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    with log.open('ab') as output:
        proc=subprocess.Popen([str(python),str(SELF),'worker','--seed',str(seed),'--stage',kind,'--fraction',str(fraction)],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        save(WORK/'ACTIVE.json',dict(pid=proc.pid,seed=seed,stage=kind,log=str(log)))
        code=proc.wait()
    save(work/(kind+'_exit.json'),dict(returncode=code,log=str(log),at=time.time()))
    return code,log


def controller():
    p=verify()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in p['seeds'][1:]:
            work=WORK/'draws'/str(seed)
            for kind,marker,required,fraction in [
                ('latent',work/'LATENT_GPU7_COMPLETE.json',88000,.40),
                ('decode',work/'DECODE_GPU7_COMPLETE.json',60000,.25),
                ('emoca',work/'emoca/EXTRACTION_COMPLETE.json',32768,.0)]:
                if marker.exists():continue
                while True:
                    free,okay=gpu_capacity(required)
                    if okay:break
                    save(WORK/'status.json',dict(stage='waiting_gpu7_capacity',next_stage=kind,seed=seed,
                        free_mib=free,required_free_mib=required,pid=os.getpid(),at=time.time(),gpu=7))
                    time.sleep(30)
                    verify()
                save(WORK/'status.json',dict(stage=kind,seed=seed,pid=os.getpid(),at=time.time()))
                code,log=call_stage(seed,kind,fraction)
                if code and kind=='latent' and 'out of memory' in log.read_text(errors='replace').lower():
                    while True:
                        free,okay=gpu_capacity(110000)
                        if okay:break
                        save(WORK/'status.json',dict(stage='waiting_gpu7_capacity',next_stage='latent_retry',seed=seed,
                            free_mib=free,required_free_mib=110000,pid=os.getpid(),at=time.time(),gpu=7))
                        time.sleep(30);verify()
                    code,log=call_stage(seed,kind,.60)
                if code or not marker.exists():
                    save(WORK/'status.json',dict(stage='failed',seed=seed,worker_stage=kind,returncode=code,log=str(log)))
                    raise RuntimeError('Worker failed; preserve outputs and inspect '+str(log))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (WORK/'scoring.log').open('ab') as log:
            subprocess.run([str(METRIC),str(SELF),'score'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def choose_candidate(gt,candidates,author):
    import numpy as np
    if gt.ndim!=2 or gt.shape[1]!=56 or len(gt)<2:raise ValueError('Expected full56 GT sequence')
    scores=[]
    for a in candidates:
        if a.shape!=gt.shape or not np.isfinite(a).all() or not np.isfinite(gt).all():raise ValueError('Invalid candidate observations')
        value=float(author.calculate_frechet_distance(*author.calculate_activation_statistics(gt),*author.calculate_activation_statistics(a)))
        if not np.isfinite(value):raise ValueError('Nonfinite selection distance')
        scores.append(value)
    if not scores:raise ValueError('No candidates')
    index=min(range(len(scores)),key=lambda i:scores[i])
    return index,scores


def score():
    import numpy as np
    from vico_literature_metrics import evaluate,load_author
    p=verify();author=load_author();ids=p['ids'];seeds=p['seeds'];features={}
    for r in read(EMOCA_SOURCE/'EXTRACTION_COMPLETE.json')['records']:
        if r['role'] not in ['speaker','gt','joint_sft2752']:continue
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as d:features[r['id'],r['role']]=d['motion'].copy()
    for seed in seeds[1:]:
        ex=read(WORK/'draws'/str(seed)/'emoca/EXTRACTION_COMPLETE.json')
        assert ex['success'] and ex['tasks']==43
        for r in ex['records']:
            assert sha(r['path'])==r['sha256']
            with np.load(r['path']) as d:features[r['id'],seed]=d['motion'].copy()
    gt=[];speaker=[];single=[];best=[];selection=[]
    for uid in ids:
        g=features[uid,'gt'][1:];x=features[uid,'speaker'][:-1]
        candidates=[features[uid,'joint_sft2752'][1:],*[features[uid,s][1:] for s in seeds[1:]]]
        if any(a.shape!=g.shape for a in [x,*candidates]):raise ValueError('Different candidate time grids')
        mask=np.isfinite(g).all(1)&np.isfinite(x).all(1)
        for a in candidates:mask &= np.isfinite(a).all(1)
        if mask.sum()<2:raise ValueError('Insufficient shared observations')
        g=g[mask];x=x[mask];candidates=[a[mask] for a in candidates]
        index,distances=choose_candidate(g,candidates,author)
        gt.append(g);speaker.append(x);single.append(candidates[0]);best.append(candidates[index])
        selection.append(dict(id=uid,seed=seeds[index],candidate_seeds=seeds,full56_fd=distances,
            listener_frame_indices=(np.where(mask)[0]+1).tolist(),shared_frames=int(mask.sum())))
    values=dict(single_shared_coverage=evaluate(gt,single,speaker),best10_shared_coverage=evaluate(gt,best,speaker))
    for r in selection:assert min(r['full56_fd'])<=r['full56_fd'][0]
    result=dict(schema='vico-dim-oracle-best10-results-v1',model=p['model'],pairs=43,
        seeds=seeds,selection=selection,methods=values,policy=p,
        strict_published_table_equivalence=False,author_contact_required=False,
        all_metrics_use_same_selected_candidate=True,primary_single_sample_preserved=True,
        sampling_budget_comparison='one fixed output vs oracle best-of10; both share all10-valid frame mask')
    save(WORK/'RESULTS.json',result)
    lines=['# DIM发布代码的10次择优补测','',
        '固定seed137–146，每条按完整56维GT Fréchet选择一次，所有指标使用同一候选。该选择使用GT，属于oracle，不代表实际部署可用的选择器。',
        '43条保留；两行使用全部10候选的共同有效帧，原有单次封存成绩不覆盖。','',
        '| 设置 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp | rPCC Pose |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in values.items():
        vals=[row['metrics'][k][g] for k in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]
        lines.append('| '+name+' | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines+=['','这是依据DIM公开tester重建的采样补充；发表主表是否采用该路径仍未被证明。RGB反提取与原生系数、GT统计等差异仍需披露。','']
    (WORK/'RESULTS.md').write_text('\n'.join(lines))
    save(WORK/'status.json',dict(stage='complete',pairs=43,candidates=430,at=time.time()))
    save(WORK/'COMPLETE.json',dict(success=True,results_sha256=sha(WORK/'RESULTS.json'),
        protocol_sha256=sha(WORK/'protocol.json'),strict_published_table_equivalence=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','controller','worker','score'])
    parser.add_argument('--seed',type=int);parser.add_argument('--stage',choices=['latent','decode','emoca'])
    parser.add_argument('--fraction',type=float,default=.4);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='controller':controller()
    elif args.action=='score':score()
    else:
        if args.seed not in range(138,147):raise ValueError('Seed outside fixed supplement')
        if args.stage is None:raise ValueError('Explicit worker stage required')
        stage(args.seed,args.stage,args.fraction)
