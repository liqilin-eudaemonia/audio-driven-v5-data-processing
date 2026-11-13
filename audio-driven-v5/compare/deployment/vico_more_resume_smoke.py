"""Real ListenFormer optimizer/RNG restart and long/short native decode audit."""
import copy
import random

import numpy as np
import torch
from vico_more_common import WORK, gpu_guard, save, sha
from vico_more_train import Windows,model_for,optimize_training,call,inference,atomic_torch


def main():
    gpu_guard(12)
    out=WORK/'diagnostics/listenformer_resume';out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(137);np.random.seed(137);random.seed(137)
    data=Windows('train',limit=4)
    model,_=model_for('listenformer');model=model.cuda().train();optimize_training(model,'listenformer')
    opt=torch.optim.AdamW(model.parameters(),lr=.002,betas=(.9,.999),weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,30,.5);scheduler.step()
    batch=data.batch(np.r_[0:6])
    def step(m,o):
        o.zero_grad(set_to_none=True)
        loss,_,pred=call(m,'listenformer',batch,0)
        loss.backward();o.step()
        return loss.detach().clone(),pred.detach().clone()
    for _ in range(3):step(model,opt)
    ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        epoch=0,offset=3,steps=3,best=1.,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
        rng_python=random.getstate(),rng_numpy=np.random.get_state(),diagnostic_only=True)
    atomic_torch(out/'last.pt',ck)
    expected_loss,expected_pred=step(model,opt)
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    fresh,_=model_for('listenformer');fresh.cuda().train();optimize_training(fresh,'listenformer')
    fresh_opt=torch.optim.AdamW(fresh.parameters(),lr=.002,betas=(.9,.999),weight_decay=.05)
    fresh_scheduler=torch.optim.lr_scheduler.StepLR(fresh_opt,30,.5)
    state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    fresh.load_state_dict(state['model'],strict=True);fresh_opt.load_state_dict(state['optimizer'])
    fresh_scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    np.random.set_state(state['rng_numpy']);random.setstate(state['rng_python'])
    loss,pred=step(fresh,fresh_opt)
    errors=dict(loss=float((loss-expected_loss).abs()),prediction=float((pred-expected_pred).abs().max()),
        parameters=max(float((v-expected[k]).abs().max()) for k,v in fresh.state_dict().items()))
    assert max(errors.values())==0,errors
    fresh.eval();audits=[]
    for n in [30,90,91,170,len(data.host['audio'][0])]:
        a=data.host['audio'][0][:n].cuda();s=data.host['speaker'][0][:n].cuda();init=data.host['listener'][0][0].cuda()
        with torch.inference_mode():p=inference(fresh,'listenformer',a,s,init)
        assert p.shape==(n,73) and torch.isfinite(p).all()
        assert torch.equal(p[0],init)
        audits.append(dict(frames=n,finite=True,initial_exact=True))
    save(out/'AUDIT.json',dict(passed=True,real_optimizer_restart=True,max_abs=errors,
        native_decode_checks=audits,checkpoint_sha256=sha(out/'last.pt'),diagnostic_only=True,
        formal_training_started=False))


if __name__=='__main__':main()
