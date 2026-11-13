"""Quick next-loss and actual stage-controller boundary checks."""
import copy
import gc
import numpy as np
import torch
from l2l_reliable_core import *
from l2l_reliable_data import load,DEFAULT_WORK,UPSTREAM
import l2l_reliable as runner

if __name__=='__main__':
    backend();work=DEFAULT_WORK;data,stats=load(work)
    folder=work/'boundary_checks';folder.mkdir(exist_ok=True)
    vq,_,_,_=factories(UPSTREAM,'vq',137);vq.eval();tokens=tokenize(vq,data);del vq;gc.collect();torch.cuda.empty_cache()
    results={}
    for stage in ['vq','predictor']:
        model,opt,_,_=factories(UPSTREAM,stage,137);sampler=EpochBatches(64,32,137);loss=Losses(stage,data,tokens)
        for _ in range(2):train_step(model,opt,loss,sampler.next())
        checkpoint(folder/'state.pth',model,opt,sampler,2,stage,{})
        expected=train_step(model,opt,loss,sampler.next())
        restore(folder/'state.pth',model,opt,sampler,{})
        actual=train_step(model,opt,loss,sampler.next());assert_equal(expected,actual,'next_loss_'+stage)
        results[stage]=dict(next_loss_and_gradient_norm_exact=True)
        del model,opt;gc.collect();torch.cuda.empty_cache()
    (folder/'state.pth').unlink()
    # Exercise the real final-last/best-recheck path under an explicitly tiny budget.
    for name in ['DATA_COMPLETE.json','CODE_FROZEN.json']:
        target=folder/name
        if not target.exists():target.symlink_to(work/name)
    runner.POLICY=dict(runner.POLICY,max_steps=2,min_steps=2,val_every=1)
    try:runner.stage_run(folder,UPSTREAM,data,stats,137,'vq')
    except RuntimeError as error:
        assert 'Learning gate failed' in str(error)
    final=torch.load(folder/'seed_137/vq/last.pth',weights_only=False,map_location='cpu')
    assert final['step']==2 and final['scheduler']['n_steps']==2
    results.update(stage_final_last_step=2,test_media_opened=False,passed=True)
    save(folder/'COMPLETE.json',results)
