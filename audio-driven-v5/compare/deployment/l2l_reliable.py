"""Auditable L2L restart. Only node11 GPU7; fail closed before test release."""
import argparse
import gc
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types
import numpy as np
import torch
from l2l_reliable_core import *
from l2l_reliable_data import ROOT,SOURCE,DEFAULT_WORK,UPSTREAM,prepare,load,freeze

STOP=False
POLICY=dict(seeds=[137,271,811],max_steps=200000,min_steps=150000,val_every=1000,patience=20,
            warmup=100000,batch=32,vq_mean_improvement=.05,predictor_unigram_improvement=.01,
            rollout_variance_ratio=[.01,10],single_candidate=True)


def interrupted(*args):
    global STOP
    STOP=True


def event(work,value):
    value=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value)
    with (work/'events.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
    save(work/'status.json',value);print(json.dumps(value),flush=True)


def stage_run(work,repo,data,stats,seed,stage,tokens=None):
    if STOP or (work/'STOP').exists():raise SystemExit(0)
    folder=work/('seed_'+str(seed))/stage;folder.mkdir(parents=True,exist_ok=True)
    identity=dict(stage=stage,seed=seed,data=sha(work/'DATA_COMPLETE.json'),code=sha(work/'CODE_FROZEN.json'),policy=POLICY,
                  tokens_vq=sha(work/('seed_'+str(seed))/'vq/best.pth') if stage!='vq' else None)
    if (folder/'COMPLETE.json').exists():
        receipt=read(folder/'COMPLETE.json');assert receipt['passed']
        assert receipt['identity']==identity
        assert sha(folder/'best.pth')==receipt['checkpoint_sha256'];return receipt
    model,opt,_,_=factories(repo,stage,seed);loss=Losses(stage,data,tokens)
    hook=loss.observe_quantizer(model) if stage=='vq' else None
    n=len(data['train']['vq' if stage=='vq' else 'listener']);sampler=EpochBatches(n,32,seed)
    unigram=None
    if stage!='vq':
        unigram,meta=train_unigram(tokens);save(folder/'UNIGRAM.json',meta)
    if (folder/'last.pth').exists():
        # A crash may occur after writing best but before last. All these are
        # full recoverable states; resume the newest complete atomic checkpoint.
        candidates=[p for p in [folder/'last.pth',folder/'best.pth',folder/'best_reconstruction.pth'] if p.exists()]
        metadata=[(torch.load(p,map_location='cpu',weights_only=False)['step'],p) for p in candidates]
        resume_path=max(metadata,key=lambda item:item[0])[1]
        state=restore(resume_path,model,opt,sampler,identity);step=state['step'];selection=state['selection']
        event(work,dict(stage=stage,seed=seed,action='resumed',step=step))
    else:
        initial=evaluate(model,loss,stats=stats,unigram=unigram);step=0
        selection=dict(initial=initial,best=initial,best_step=0,best_reconstruction=initial.get('reconstruction'),stale=0,history=[])
        for name in ['initial.pth','best.pth','last.pth']:checkpoint(folder/name,model,opt,sampler,0,stage,identity,selection)
        event(work,dict(stage=stage,seed=seed,action='initialized',step=0,validation=initial,
                        effective_training_rows=sampler.effective_n,omitted_training_rows=n-sampler.effective_n))
    started=time.monotonic();start_step=step;train_sum={};grad_sum=0.;count=0
    termination='max_steps'
    while step<POLICY['max_steps']:
        # Recovery after the plateau checkpoint must not add another interval.
        if step>=POLICY['min_steps'] and selection['stale']>=POLICY['patience']:
            termination='validation_plateau';break
        if STOP or (work/'STOP').exists():
            checkpoint(folder/'last.pth',model,opt,sampler,step,stage,identity,selection)
            event(work,dict(stage=stage,seed=seed,action='stopped_safely',step=step));raise SystemExit(0)
        parts,grad=train_step(model,opt,loss,sampler.next());step+=1;count+=1;grad_sum+=grad
        for k,v in parts.items():train_sum[k]=train_sum.get(k,0.)+v
        # Early points are diagnostics only; plateau counter uses regular checks.
        validate=step in [1,100] or step%POLICY['val_every']==0 or step==POLICY['max_steps']
        if not validate:continue
        val=evaluate(model,loss,stats=stats,unigram=unigram)
        improved=val['total']<selection['best']['total']
        if improved:selection.update(best=val,best_step=step,stale=0)
        elif step%POLICY['val_every']==0:selection['stale']+=1
        selection['history'].append(dict(step=step,total=val['total']))
        if improved:checkpoint(folder/'best.pth',model,opt,sampler,step,stage,identity,selection)
        if stage=='vq' and val['reconstruction']<selection['best_reconstruction']:
            selection['best_reconstruction']=val['reconstruction']
            checkpoint(folder/'best_reconstruction.pth',model,opt,sampler,step,stage,identity,selection)
        checkpoint(folder/'last.pth',model,opt,sampler,step,stage,identity,selection)
        if step%10000==0:checkpoint(folder/('recovery_'+str(step)+'.pth'),model,opt,sampler,step,stage,identity,selection)
        elapsed=time.monotonic()-started;speed=(step-start_step)/elapsed
        event(work,dict(stage=stage,seed=seed,action='validation',step=step,best_step=selection['best_step'],
             learning_rate=opt._optimizer.param_groups[0]['lr'],train={k:v/count for k,v in train_sum.items()},
             gradient_norm=grad_sum/count,validation=val,updates_per_second=speed,
             stage_max_remaining_seconds=(POLICY['max_steps']-step)/speed))
        train_sum={};count=0;grad_sum=0.
        if step>=POLICY['min_steps'] and selection['stale']>=POLICY['patience']:
            termination='validation_plateau';break
    # Persist actual final optimizer/model BEFORE inspecting best.
    checkpoint(folder/'last.pth',model,opt,sampler,step,stage,identity,selection)
    best=selection['best'];initial=selection['initial']
    best_state=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
    assert best_state['identity']==identity and best_state['step']==selection['best_step']
    model.load_state_dict(best_state['state_dict'],strict=True)
    assert_equal(evaluate(model,loss,stats=stats,unigram=unigram),best,'selected_validation_recheck')
    del best_state
    checks=dict(selected_after_initialization=selection['best_step']>0,
                improved_from_initial=best['total']<initial['total'])
    if stage=='vq':
        checks.update(beats_mean_by_five_percent=best['reconstruction']<=best['train_mean_baseline']*.95,
                      multiple_codes=best['unique_codes']>=2)
    else:checks['beats_unigram_by_one_percent']=best['total']<=best['unigram_baseline']*.99
    receipt=dict(passed=all(checks.values()),checks=checks,seed=seed,stage=stage,best_step=selection['best_step'],
                 final_step=step,termination=termination,validation=best,initial=initial,
                 checkpoint_sha256=sha(folder/'best.pth'),last_sha256=sha(folder/'last.pth'),identity=identity)
    save(folder/('COMPLETE.json' if receipt['passed'] else 'FAILED_GATE.json'),receipt)
    if hook:hook.remove()
    del model,opt;gc.collect();torch.cuda.empty_cache()
    if not receipt['passed']:raise RuntimeError('Learning gate failed: '+str(folder))
    return receipt


def load_best(work,repo,seed,stage):
    model,opt,cfg,_=factories(repo,stage,seed)
    path=work/('seed_'+str(seed))/stage/'best.pth'
    assert sha(path)==read(path.parent/'COMPLETE.json')['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'],strict=True);model.eval()
    del opt;return model,cfg


def rollout(work,repo,stats,seed,items,destination):
    """items contain source paths only. Listener/GT files are inaccessible here."""
    from utils.load_utils import bilateral_filter
    from test_vq_decoder import generate_prediction
    vq,_=load_best(work,repo,seed,'vq');model,cfg=load_best(work,repo,seed,'predictor')
    padded=(max(s['frames'] for s in items)+31)//32*32;b=len(items)
    # Extra32history ensures the final40-frame condition window is complete.
    x=np.zeros((b,padded+32,56),np.float32);a=np.zeros((b,(padded+32)*4,128),np.float32)
    sources={}
    for i,s in enumerate(items):
        f=np.load(s['source']);speaker=f['speaker'] if 'speaker' in f else f['motion']
        mel=f['audio'] if 'audio' in f else np.load(s['audio']);n=s['frames']
        assert len(speaker)==n and len(mel)==4*n
        sx=(bilateral_filter(speaker[None])[0]-stats['speaker_mean'].reshape(56))/stats['speaker_std'].reshape(56)
        am=(mel-stats['audio_mean'].reshape(128))/stats['audio_std'].reshape(128)
        x[i,32:32+n]=sx;x[i,32+n:]=sx[-1];a[i,128:128+4*n]=am
        a[i,128+4*n:]=(0-stats['audio_mean'].reshape(128))/stats['audio_std'].reshape(128)
        sources[s['id']]=dict(source_sha256=sha(s['source']),frames=n)
    seed_all(seed);history=np.zeros((b,32,56),np.float32);token=None;shape=None
    with torch.no_grad():
        for t in range(0,padded,8):
            past=history if token is None else token[:,-4:].cpu().numpy()
            pred,_,inputs,shape=generate_prediction(cfg,types.SimpleNamespace(sample_idx=None),vq,model,
                   x[:,t:t+40],past,a[:,4*t:4*(t+40)],32,8,t//8,4,btc=shape)
            if token is None:token=inputs['listener_past']
            token=torch.cat((token,pred[:,0]),dim=1)
        codes=token[:,4:];decoded=[]
        for start in range(0,codes.shape[1],4):decoded.append(vq.module.decode_to_img(codes[:,start:start+4],shape).cpu().numpy())
    output=np.concatenate(decoded,axis=1)*stats['listener_std']+stats['listener_mean'];assert np.isfinite(output).all()
    destination.mkdir(parents=True,exist_ok=True);files={}
    for i,s in enumerate(items):
        path=destination/(s['id']+'.npz');np.savez_compressed(path,motion=output[i,:s['frames']].astype(np.float32));files[path.name]=sha(path)
    receipt=dict(seed=seed,clips=b,source_only=True,source_inputs=sources,outputs=files,sampling_candidates=1,gt_selected=False)
    save(destination/'COMPLETE.json',receipt)
    del vq,model;gc.collect();torch.cuda.empty_cache();return receipt


def validation_rollout(work,repo,stats,seed):
    folder=work/('seed_'+str(seed));gate=folder/'ROLLOUT_GATE.json'
    identity=dict(code=sha(work/'CODE_FROZEN.json'),data=sha(work/'DATA_COMPLETE.json'),
                  vq=sha(folder/'vq/best.pth'),predictor=sha(folder/'predictor/best.pth'),policy=POLICY)
    if gate.exists():assert read(gate)['passed'] and read(gate)['identity']==identity;return
    if STOP or (work/'STOP').exists():raise SystemExit(0)
    sources=read(work/'source_features.json')
    items=[]
    for row in read(work/'split.json')['validation']:
        uid=row['uuid'];path=SOURCE/'features/validation'/(uid+'.npz')
        assert sha(path)==sources[str(path)],path
        with np.load(path) as f:n=len(f['speaker'])
        items.append(dict(id=uid,frames=n,source=str(path)))
    rollout(work,repo,stats,seed,items,folder/'validation_rollout')
    rows=[]
    for s in items:
        pred=np.load(folder/'validation_rollout'/(s['id']+'.npz'))['motion'].astype(np.float64)
        gt=np.load(s['source'])['listener'].astype(np.float64)
        rows.append(dict(id=s['id'],frames=s['frames'],pred_variance={k:float(np.var(pred[:,sl],axis=0).mean()) for k,sl in GROUPS.items()},
                         gt_variance={k:float(np.var(gt[:,sl],axis=0).mean()) for k,sl in GROUPS.items()}))
    ratios={};collapsed={}
    for k in GROUPS:
        p=np.mean([r['pred_variance'][k] for r in rows]);g=np.mean([r['gt_variance'][k] for r in rows]);ratios[k]=float(p/g) if g>1e-12 else None
        collapsed[k]=[r['id'] for r in rows if r['gt_variance'][k]>1e-12 and r['pred_variance'][k]<=1e-12]
    checks={k:ratios[k] is None or .01<=ratios[k]<=10 for k in ['exp','head_rotation']}
    save(gate,dict(passed=all(checks.values()),identity=identity,checks=checks,variance_ratio=ratios,constant_clip_ids=collapsed,per_clip=rows,
                   limitation='Aggregate variance gate; per-clip constant outputs explicitly reported',seed=seed,test_opened=False))
    assert all(checks.values()),'Validation rollout variance gate failed'


def test_and_score(work,repo,stats):
    if STOP or (work/'STOP').exists():raise SystemExit(0)
    # All3 seeds must pass before opening any test media or calculating scores.
    release={str(seed):{name:read(work/('seed_'+str(seed))/name) for name in ['vq/COMPLETE.json','predictor/COMPLETE.json','ROLLOUT_GATE.json']} for seed in POLICY['seeds']}
    assert all(r['passed'] for entries in release.values() for r in entries.values())
    save(work/'TEST_RELEASE.json',dict(policy=POLICY,seeds=release,selected_on_test=False))
    base=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen';samples=read(base/'protocol.json')['samples']
    assert {s['id'] for s in samples}==set(read(work/'split.json')['held_out_test_ids'])
    from vico_common_deca import clip_metrics,GROUPS as SCORE_GROUPS
    results=[]
    for seed in POLICY['seeds']:
        folder=work/('seed_'+str(seed));items=[dict(id=s['id'],frames=s['frames'],source=str(base/'features/speaker'/(s['id']+'.npz')),
                    audio=str(base/'audio_features'/(s['id']+'.npy'))) for s in samples]
        rollout(work,repo,stats,seed,items,folder/'test_generated');rows=[]
        for s in samples:
            uid=s['id'];src,gt=[np.load(base/'features'/role/(uid+'.npz'))['motion'].astype(np.float64) for role in ['speaker','gt']]
            path=folder/'test_generated'/(uid+'.npz');pred=np.load(path)['motion'].astype(np.float64)
            assert pred.shape==gt.shape
            rows.append(dict(id=uid,frames=len(gt),metrics=clip_metrics(src,gt,pred),prediction_sha256=sha(path)))
        metrics={k:{g:float(np.mean([r['metrics'][k][g] for r in rows if r['metrics'][k][g] is not None])) for g in SCORE_GROUPS} for k in rows[0]['metrics']}
        coverage={k:{g:sum(r['metrics'][k][g] is not None for r in rows) for g in SCORE_GROUPS} for k in metrics}
        result=dict(seed=seed,clips=len(rows),metrics=metrics,coverage=coverage,per_clip=rows,strict_DIM_reproduction=False,
                    photorealistic_metrics=None,space='DECA50+3+3',test_release_sha256=sha(work/'TEST_RELEASE.json'))
        save(folder/'RESULTS.json',result);results.append(result)
    aggregate={k:{g:dict(mean=float(np.mean([r['metrics'][k][g] for r in results])),std=float(np.std([r['metrics'][k][g] for r in results],ddof=1))) for g in SCORE_GROUPS} for k in results[0]['metrics']}
    save(work/'RESULTS.json',dict(seeds=POLICY['seeds'],per_seed=results,aggregate=aggregate,primary_seed=137,strict_DIM_reproduction=False))
    save(work/'COMPUTE_COMPLETE.json',dict(passed=True,publication_pending=True,gpu7_released_on_exit=True,result_sha256=sha(work/'RESULTS.json')))


def run(work):
    lock=(work/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=open('/tmp/l2l_reliable_node11_gpu7.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend();assert read(work/'preflight/COMPLETE.json')['passed']
    assert read(work/'boundary_checks/COMPLETE.json')['passed']
    prepare(work)  # Verify every frozen data/statistics/split digest on resume.
    frozen=read(work/'CODE_FROZEN.json')
    for path,expected in frozen['files'].items():assert sha(work/path)==expected,path
    repo=work/'code_snapshot/l2l_src';data,stats=load(work)
    save(work/'POLICY.json',POLICY)
    for seed in POLICY['seeds']:
        if STOP or (work/'STOP').exists():raise SystemExit(0)
        stage_run(work,repo,data,stats,seed,'vq')
        if STOP or (work/'STOP').exists():raise SystemExit(0)
        vq,_=load_best(work,repo,seed,'vq');tokens=tokenize(vq,data);del vq;gc.collect();torch.cuda.empty_cache()
        stage_run(work,repo,data,stats,seed,'predictor',tokens);del tokens
        if STOP or (work/'STOP').exists():raise SystemExit(0)
        validation_rollout(work,repo,stats,seed)
    if STOP or (work/'STOP').exists():raise SystemExit(0)
    test_and_score(work,repo,stats)
    event(work,dict(action='compute_complete',publication_pending=True,gpu7='returned_to_inference'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--work',type=Path,default=DEFAULT_WORK);args=p.parse_args()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.command=='freeze':
        assert read(args.work/'preflight/COMPLETE.json')['passed'];freeze(args.work)
    else:
        try:run(args.work)
        except Exception:
            save(args.work/'FAILED.json',dict(traceback=traceback.format_exc(),gpu7_released_on_exit=True))
            raise
