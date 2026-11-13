"""Upstream L2L training operations with complete deterministic recovery."""
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import torch

GPU_UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
GROUPS={'exp':slice(0,50),'head_rotation':slice(50,53),'jaw_rotation':slice(53,56)}
MASKS=[(0,1/21),(1,1/21),(2,1/21),(3,1/21),(4,17/21)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def backend():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7','Only node11 physicalGPU7 authorized'
    actual=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual==GPU_UUID,actual
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in [':4096:8',':16:8']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    limit=os.environ.get('L2L_GPU_MEMORY_LIMIT_GIB')
    if limit:
        total=torch.cuda.get_device_properties(0).total_memory
        fraction=float(limit)*1024**3/total
        assert 0<fraction<=1
        torch.cuda.set_per_process_memory_fraction(fraction,device=0)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def factories(repo,stage,seed):
    sys.path.insert(0,str(repo))
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    from modules.fact_model import setup_model
    cfg=read(Path(repo)/'configs/vq/delta_v6.json');vqcfg=read(Path(repo)/cfg['l_vqconfig'])
    seed_all(seed)
    if stage=='vq':model,opt,_=setup_vq_transformer(types.SimpleNamespace(),vqcfg,load_path=None)
    else:model,opt,_=setup_model(cfg,vqcfg,load_path=None)
    assert opt.n_warmup_steps==100000
    assert opt.init_lr==(2. if stage=='vq' else .01)
    assert opt.d_model==(256 if stage=='vq' else 200)
    return model,opt,cfg,vqcfg


class EpochBatches:
    """Fresh shuffled full-batch indices each epoch, as in upstream trainers."""
    def __init__(self,n,batch,seed):
        self.n=n;self.batch=batch;self.rng=np.random.RandomState(seed);self.epoch=-1;self.cursor=0;self.order=[]
        assert n>=batch

    @property
    def effective_n(self):return self.n//self.batch*self.batch

    def next(self):
        if self.cursor>=len(self.order):
            self.epoch+=1;self.order=np.arange(self.n//self.batch);self.rng.shuffle(self.order);self.cursor=0
        batch_id=int(self.order[self.cursor]);self.cursor+=1
        return np.arange(batch_id*self.batch,(batch_id+1)*self.batch)

    def state(self):
        return dict(n=self.n,batch=self.batch,rng=self.rng.get_state(),epoch=self.epoch,cursor=self.cursor,order=np.asarray(self.order).copy())

    def restore(self,state):
        assert (state['n'],state['batch'])==(self.n,self.batch)
        self.rng.set_state(state['rng']);self.epoch=state['epoch'];self.cursor=state['cursor'];self.order=np.array(state['order']).copy()


def model_random_states(model):
    result={}
    for module_name,module in model.named_modules():
        for name,value in vars(module).items():
            if isinstance(value,np.random.RandomState):result[(module_name,name)]=value.get_state()
    return result


def restore_model_random_states(model,state):
    modules=dict(model.named_modules())
    assert set(state)==set(model_random_states(model)), 'Model-local RNG layout changed'
    for (module_name,name),value in state.items():getattr(modules[module_name],name).set_state(value)


def rng_state(model):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),model=model_random_states(model))


def restore_rng(model,state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda']);restore_model_random_states(model,state['model'])


@contextlib.contextmanager
def validation_context(model):
    state=rng_state(model);training=model.training;model.eval()
    try:
        with torch.no_grad():yield
    finally:
        model.train(training);restore_rng(model,state)


def scheduler_state(opt):
    return {k:getattr(opt,k) for k in ['init_lr','d_model','n_warmup_steps','n_steps']}


def checkpoint(path,model,opt,sampler,step,stage,identity,selection=None):
    assert opt.n_steps==step
    value=dict(schema='l2l-reliable-full-checkpoint-v1',state_dict=model.state_dict(),
        optimizer=opt._optimizer.state_dict(),scheduler=scheduler_state(opt),sampler=sampler.state(),
        rng=rng_state(model),step=step,stage=stage,identity=identity,selection=copy.deepcopy(selection or {}))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def restore(path,model,opt,sampler,identity):
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['schema']=='l2l-reliable-full-checkpoint-v1' and value['identity']==identity
    model.load_state_dict(value['state_dict'],strict=True);opt._optimizer.load_state_dict(value['optimizer'])
    for key,item in value['scheduler'].items():setattr(opt,key,item)
    sampler.restore(value['sampler']);restore_rng(model,value['rng'])
    assert opt.n_steps==value['step']
    return value


def tensor(value,dtype=torch.float32):
    # Own the selected rows so mmap/read-only buffers never reach a write path.
    return torch.as_tensor(np.array(value,copy=True),dtype=dtype,device='cuda:0')


class Losses:
    def __init__(self,stage,data,tokens=None):self.stage=stage;self.data=data;self.tokens=tokens;self.last_codes=None;self.last_quant_parts=None

    def observe_quantizer(self,model):
        def hook(module,inputs,outputs):
            with torch.no_grad():
                z=inputs[0].detach().permute(0,2,1).contiguous().reshape(-1,module.e_dim)
                indices=outputs[2][2].detach().reshape(-1);q=module.embedding(indices).detach()
                self.last_codes=indices
                self.last_quant_parts=dict(commitment=module.beta*(q-z).square().mean(),embedding=(q-z).square().mean())
        return model.module.quantize.register_forward_hook(hook)

    def __call__(self,model,split,indices,training):
        if self.stage=='vq':
            from vqgan.vqmodules.gan_models import calc_vq_loss
            y=self.data[split]['vq'][indices];target=tensor(y);prediction,quant=model(target,None)
            parts={key:(prediction[:,:,sl]-target[:,:,sl]).abs().mean() for key,sl in GROUPS.items()}
            parts['reconstruction']=parts['exp']+parts['head_rotation']+parts['jaw_rotation']
            parts['quantization']=quant.mean();parts['total']=calc_vq_loss(prediction,target,quant)
            if self.last_quant_parts:parts.update(self.last_quant_parts)
            return parts['total'],parts,prediction,target
        from modules.fact_model import calc_logit_loss
        d=self.data[split];t=self.tokens[split]
        # Preserve upstream allocation/strides as well as values for update parity.
        inputs=dict(speaker_full=tensor(d['speaker'][indices])[:,:40],audio_full=tensor(d['audio'][indices])[:,:160],
                    listener_past=tensor(t['past'][indices],torch.long))
        target=tensor(t['future'][indices],torch.long)
        if training:
            pred=model(inputs,4,-1);loss=calc_logit_loss(pred[:,:4],target)
            return loss,dict(total=loss),pred,target
        parts={}
        for mask,weight in MASKS:
            pred=model(inputs,4,mask);parts['mask'+str(mask)]=calc_logit_loss(pred[:,:4],target)
        loss=sum(parts['mask'+str(mask)]*weight for mask,weight in MASKS);parts['total']=loss
        return loss,parts,None,target


def train_step(model,opt,losses,indices):
    model.train();opt.zero_grad();loss,parts,_,_=losses(model,'train',indices,True)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    norm=torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
    opt.step_and_update_lr()  # KEEP the official optimizer and its real warmup.
    return {k:float(v.detach()) for k,v in parts.items()},float(norm)


def evaluate(model,losses,batch=32,stats=None,unigram=None):
    n=len(losses.data['validation']['vq' if losses.stage=='vq' else 'listener']);sums={};codes=np.zeros(200,np.int64)
    raw_err={k:0. for k in GROUPS};pred_var={k:0. for k in GROUPS};gt_var={k:0. for k in GROUPS};window_count=0
    with validation_context(model):
        for start in range(0,n,batch):
            ids=np.arange(start,min(start+batch,n));_,parts,pred,target=losses(model,'validation',ids,False)
            for key,value in parts.items():sums[key]=sums.get(key,0.)+float(value)*len(ids)
            if losses.stage=='vq':
                assert losses.last_codes is not None
                codes+=np.bincount(losses.last_codes.cpu().numpy(),minlength=200)
                z=target.cpu().numpy();baseline=sum(float(np.abs(z[:,:,sl]).mean()) for sl in GROUPS.values())
                sums['train_mean_baseline']=sums.get('train_mean_baseline',0.)+baseline*len(ids)
                if stats is not None:
                    pr=pred.cpu().numpy().astype(np.float64)*stats['listener_std']+stats['listener_mean']
                    gt=z.astype(np.float64)*stats['listener_std']+stats['listener_mean']
                    for key,sl in GROUPS.items():
                        raw_err[key]+=float(np.abs(pr[:,:,sl]-gt[:,:,sl]).mean())*len(ids)
                        pred_var[key]+=float(np.var(pr[:,:,sl],axis=1).mean())*len(ids)
                        gt_var[key]+=float(np.var(gt[:,:,sl],axis=1).mean())*len(ids)
                    window_count+=len(ids)
            elif unigram is not None:
                value=float(-np.log(unigram[target.cpu().numpy()]).mean())
                sums['unigram_baseline']=sums.get('unigram_baseline',0.)+value*len(ids)
    values={k:v/n for k,v in sums.items()}
    assert all(math.isfinite(v) for v in values.values())
    values['samples']=n
    if losses.stage=='vq':
        values.update(unique_codes=int(np.count_nonzero(codes)),code_histogram=codes.tolist())
        if window_count:
            values['raw_l1']={k:v/window_count for k,v in raw_err.items()}
            values['variance_ratio']={k:pred_var[k]/gt_var[k] if gt_var[k]>1e-12 else None for k in GROUPS}
    else:values['perplexity']=math.exp(min(values['total'],700))
    return values


def tokenize(vq,data,batch=128):
    result={}
    with validation_context(vq):
        for split,d in data.items():
            past=[];future=[]
            for start in range(0,len(d['listener']),batch):
                y=tensor(d['listener'][start:start+batch]);_,p=vq.module.get_quant(y[:,:32]);_,f=vq.module.get_quant(y[:,32:])
                past.append(p.reshape(len(y),4).cpu().numpy());future.append(f.reshape(len(y),4).cpu().numpy())
            result[split]=dict(past=np.concatenate(past),future=np.concatenate(future))
    return result


def train_unigram(tokens,batch=32):
    effective=len(tokens['train']['future'])//batch*batch
    counts=np.bincount(tokens['train']['future'][:effective].reshape(-1),minlength=200)
    probabilities=(counts+1)/(counts.sum()+200)
    return probabilities,dict(alpha=1,classes=200,effective_training_rows=effective,
        omitted_training_rows=len(tokens['train']['future'])-effective,training_target_tokens=int(counts.sum()),counts=counts.tolist())


def assert_equal(a,b,path='value'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path
        for key in a:assert_equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        assert isinstance(b,type(a)) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_equal(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)
