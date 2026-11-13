"""Bounded diagnostic only. Train-only codebook initialization, no test media."""
import os,sys,json,time,gc,copy
from pathlib import Path
import numpy as np,torch
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');W=ROOT/'experiments/p2_l2l_reliable_retrain_20261107';OUT=W/'codebook_initialization_spike'
OUT.mkdir(exist_ok=True);sys.path.insert(0,str(W/'code_snapshot'))
from l2l_reliable_core import *
from l2l_reliable_data import load
backend();data,stats=load(W);repo=W/'code_snapshot/l2l_src';model,opt,_,_=factories(repo,'vq',137)
loss=Losses('vq',data);hook=loss.observe_quantizer(model);sampler=EpochBatches(len(data['train']['vq']),32,137)
identity=dict(diagnostic_only=True,initialization='sample200 actual train encoder vectors',seed=137,max_steps=3000,upstream_schedule_unchanged=True,data_sha256=sha(W/'DATA_COMPLETE.json'),frozen_code_sha256=sha(W/'CODE_FROZEN.json'))
rng=np.random.RandomState(137);effective=sampler.effective_n;ids=rng.choice(effective,size=1024,replace=False);pool=[]
with validation_context(model):
 for start in range(0,len(ids),32):
  h=model.module.encoder(tensor(data['train']['vq'][ids[start:start+32]]))
  # EXACT upstream quantizer vector layout, including its permutation.
  pool.append(h.permute(0,2,1).contiguous().reshape(-1,256))
 pool=torch.cat(pool);chosen=rng.choice(len(pool),size=200,replace=False)
 codes=pool[torch.as_tensor(chosen,device='cuda:0')];model.module.quantize.embedding.weight.copy_(codes)
seed_all(137);initial=evaluate(model,loss,stats=stats);selection=dict(initial=initial,best=initial,best_step=0,history=[])
checkpoint(OUT/'initial.pth',model,opt,sampler,0,'vq',identity,selection);checkpoint(OUT/'best.pth',model,opt,sampler,0,'vq',identity,selection)
save(OUT/'INITIALIZATION.json',dict(**identity,training_window_ids=ids.tolist(),sampled_latent_indices=chosen.tolist(),encoder_pool_rms=float(pool.square().mean().sqrt()),codebook_rms=float(codes.square().mean().sqrt()),initial_validation=initial,test_opened=False))
del pool,codes;torch.cuda.empty_cache();started=time.monotonic();bad=0;reason='budget3000';step=0
for step in range(1,3001):
 train_step(model,opt,loss,sampler.next())
 if step not in [1,100,500,1000,2000,3000]:continue
 val=evaluate(model,loss,stats=stats);selection['history'].append(dict(step=step,validation=val))
 if val['total']<selection['best']['total']:
  selection.update(best=val,best_step=step);checkpoint(OUT/'best.pth',model,opt,sampler,step,'vq',identity,selection)
 checkpoint(OUT/'last.pth',model,opt,sampler,step,'vq',identity,selection)
 bad=bad+1 if val['quantization']>max(initial['quantization']*10,20) or val['total']>initial['total']*2 else 0
 status=dict(step=step,validation=val,best_step=selection['best_step'],elapsed_seconds=time.monotonic()-started,diagnostic_only=True)
 save(OUT/'status.json',status);print(json.dumps(status),flush=True)
 if bad>=2:reason='divergence_guard';break
best=selection['best'];checks=dict(improved_total=best['total']<initial['total'],reconstruction_beats_mean5pct=best['reconstruction']<best['train_mean_baseline']*.95,multiple_codes=best['unique_codes']>=2,quantization_does_not_dominate=best['quantization']<best['reconstruction'])
save(OUT/'RESULTS.json',dict(diagnostic_only=True,formal_baseline=False,test_opened=False,steps=step,termination=reason,checks=checks,passed=all(checks.values()),initial=initial,best=best,best_step=selection['best_step'],history=selection['history'],identity=identity))
