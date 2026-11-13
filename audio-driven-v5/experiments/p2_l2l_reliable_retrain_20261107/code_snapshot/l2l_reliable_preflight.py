"""Real upstream parity, recovery and small-data learning checks; no test input."""
import copy
import gc
from pathlib import Path
import sys
import types
import numpy as np
import torch
from l2l_reliable_core import *


def run(repo,work,data,stats):
    folder=Path(work)/'preflight';folder.mkdir(exist_ok=True)
    sys.path.insert(0,str(repo));sys.path.insert(0,str(Path(repo)/'vqgan'))
    from vqgan.train_vq_transformer import generator_train_step as official_vq
    from train_vq_decoder import generator_train_step as official_predictor
    from utils.load_utils import create_data_vq
    result={};vq,_,_,_=factories(repo,'vq',137);vq.eval()
    tokens=tokenize(vq,data)
    # Independently compare cached token inputs with upstream data preparation.
    with torch.no_grad():
        d=data['train'];inputs,future,_,_=create_data_vq(vq,np.array(d['speaker'][:32]),np.array(d['listener'][:32]),np.array(d['audio'][:32]),32,patch_size=8)
        assert_equal(inputs['listener_past'].cpu().numpy(),tokens['train']['past'][:32])
        assert_equal(future.cpu().numpy(),tokens['train']['future'][:32])
        assert_equal(inputs['speaker_full'].cpu().numpy(),np.array(d['speaker'][:32,:40]))
        assert_equal(inputs['audio_full'].cpu().numpy(),np.array(d['audio'][:32,:160]))
    writer=types.SimpleNamespace(add_scalar=lambda *a,**kw:None)
    for stage in ['vq','predictor']:
        model,opt,cfg,vqcfg=factories(repo,stage,137)
        loss=Losses(stage,data,tokens);hook=loss.observe_quantizer(model) if stage=='vq' else None
        sampler=EpochBatches(32,32,137);identity={'preflight':stage}
        # Step1001 makes a wrong update observable, unlike an almost-zero firstLR.
        opt.n_steps=1000
        checkpoint(folder/'initial.pth',model,opt,sampler,1000,stage,identity)
        ref_rng=np.random.RandomState(137)
        if stage=='vq':
            official_vq(vqcfg,0,model,opt,np.array(data['train']['vq'][:32]),ref_rng,writer)
        else:
            d=data['train'];official_predictor(cfg,0,model,opt,vq,np.array(d['speaker'][:32]),np.array(d['listener'][:32]),np.array(d['audio'][:32]),ref_rng,writer,8,32)
        reference=copy.deepcopy(dict(model=model.state_dict(),adam=opt._optimizer.state_dict(),scheduler=scheduler_state(opt),rng=rng_state(model)))
        restore(folder/'initial.pth',model,opt,sampler,identity)
        train_step(model,opt,loss,sampler.next())
        actual=dict(model=model.state_dict(),adam=opt._optimizer.state_dict(),scheduler=scheduler_state(opt),rng=rng_state(model))
        assert_equal(reference,actual,'official_update_'+stage);del reference,actual
        # Two updates, save, two updates; compare with a NEW model restored at midpoint.
        train_step(model,opt,loss,sampler.next())
        checkpoint(folder/'midpoint.pth',model,opt,sampler,1002,stage,identity)
        for _ in range(2):train_step(model,opt,loss,sampler.next())
        checkpoint(folder/'uninterrupted.pth',model,opt,sampler,1004,stage,identity)
        if hook:hook.remove()
        del model,opt;gc.collect();torch.cuda.empty_cache()
        model,opt,_,_=factories(repo,stage,999);sampler=EpochBatches(32,32,999)
        restore(folder/'midpoint.pth',model,opt,sampler,identity)
        loss=Losses(stage,data,tokens);hook=loss.observe_quantizer(model) if stage=='vq' else None
        # Validation may not consume train RNG or mutate BatchNorm buffers.
        before_rng=rng_state(model);before_model=copy.deepcopy(model.state_dict())
        evaluate(model,loss,batch=32,stats=stats,unigram=train_unigram(tokens)[0])
        assert_equal(before_rng,rng_state(model),'validation_rng');assert_equal(before_model,model.state_dict(),'validation_buffers')
        del before_model,before_rng
        for _ in range(2):train_step(model,opt,loss,sampler.next())
        checkpoint(folder/'resumed.pth',model,opt,sampler,1004,stage,identity)
        assert_equal(torch.load(folder/'uninterrupted.pth',weights_only=False,map_location='cpu'),torch.load(folder/'resumed.pth',weights_only=False,map_location='cpu'),'resume_'+stage)
        # Diagnostic ONLY: shorten warmup to100 for a repeated8-window batch.
        if hook:hook.remove()
        del model,opt;gc.collect();torch.cuda.empty_cache()
        model,opt,_,_=factories(repo,stage,137);opt.n_warmup_steps=100
        # Preserve a1e-4 VQ peak in the SHORT diagnostic; base2 with100warmup
        # would peak at.0125, which is not a meaningful learning smoke test.
        if stage=='vq':opt.init_lr=.016
        loss=Losses(stage,data,tokens);hook=loss.observe_quantizer(model) if stage=='vq' else None
        ids=np.arange(8)
        def diagnostic_loss():
            with validation_context(model):return float(loss(model,'train',ids,False)[0])
        initial=diagnostic_loss()
        for _ in range(400):train_step(model,opt,loss,ids)
        final=diagnostic_loss();assert final<initial*.95,(stage,initial,final)
        result[stage]=dict(official_step_parity=True,step_for_parity=1001,resume_exact=True,validation_preserves_state=True,
            overfit_initial=initial,overfit_final=final,diagnostic_steps=400,diagnostic_warmup=100,
            diagnostic_base_lr=opt.init_lr,formal_weights_reused=False)
        if hook:hook.remove()
        del model,opt;gc.collect();torch.cuda.empty_cache()
        save(folder/'PROGRESS.json',result)
    del vq;gc.collect();torch.cuda.empty_cache()
    for p in folder.glob('*.pth'):p.unlink()
    result.update(passed=True,test_media_opened=False,upstream_token_parity=True)
    save(folder/'COMPLETE.json',result);return result
