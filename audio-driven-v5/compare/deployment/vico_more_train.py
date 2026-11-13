"""Native RLHG/ListenFormer architectures, full official epoch budgets, resumable.

Local adaptation: single GPU retains official effective batch (16/128), exact
window enumeration, AdamW and pre-epoch StepLR. GPU-resident raw clips replace
duplicated eager window dictionaries. It is a local ViCo retraining, not an
author checkpoint reproduction. Incomplete budgets remain diagnostic.
"""
import argparse
import ast
import copy
import importlib
import math
import os
from pathlib import Path
from types import MethodType
import random
import sys
import time

import numpy as np
import torch
from vico_more_common import ROOT, WORK, BFM, read, save, sha, gpu_guard, test_samples
from vico_more_features import OUT as FEATURES, dynamic

FREEZE = 1794096000  # 2026-11-08 00:00 UTC


def model_for(method):
    import yaml
    from box import Box
    vendor = WORK / 'vendor' / method
    base = vendor / 'vico' if method == 'rlhg' else vendor
    sys.path.insert(0, str(base))
    config = Box(yaml.safe_load((base / 'configs/configs.yaml').read_text()))
    module = importlib.import_module('networks.listener_generator')
    if method == 'listenformer':
        # Upstream decode_period_sl_initlast_cat references an undefined global.
        # Match its shipped run.sh change_epoch=100; inference epoch is fixed500.
        module.change_epoch = 100
    cls = module.ListenerGenerator if method == 'rlhg' else module.ListenerGenerator_trans_ca
    return cls(config), config


class Windows:
    def __init__(self, split, stats=None, limit=None):
        tasks = [r for r in read(FEATURES / 'protocol.json')['tasks'] if r['split'] == split]
        if limit is not None:
            tasks = [r for r in tasks if all((FEATURES / split / r['id'] / (k+'.json')).exists()
                     for k in ['audio', 'speaker', 'listener'])][:limit]
        assert tasks, f'No completed {split} feature clips'
        self.tasks = tasks
        chunks = {'audio': [], 'speaker': [], 'listener': []}
        for r in tasks:
            folder = FEATURES / split / r['id']
            for name in ['audio','speaker','listener']:
                path=folder/(name+('.npy' if name=='audio' else '.npz'))
                assert sha(path)==read(path.with_suffix('.json'))['sha256'],path
            chunks['audio'].append(np.load(folder / 'audio.npy'))
            for role in ['speaker', 'listener']:
                chunks[role].append(np.load(folder / (role+'.npz'))['dynamic'])
        if stats is None:
            assert limit is not None, 'Partial statistics only allowed for diagnostics'
            stats = {}
            for key in ['audio', 'dynamic']:
                a = torch.from_numpy(np.concatenate(chunks['audio'] if key == 'audio' else chunks['speaker']+chunks['listener']))
                stats[key+'_mean'] = a.mean(0).numpy()
                std = a.std(0).numpy()
                stats[key+'_std'] = np.where(std == 0, 1, std)
        self.stats = stats
        normalized = {}
        for name, parts in chunks.items():
            key = 'audio' if name == 'audio' else 'dynamic'
            mean, std = [torch.from_numpy(stats[key+'_'+x]) for x in ['mean', 'std']]
            normalized[name] = [((torch.from_numpy(a)-mean)/std).float() for a in parts]
        self.host = normalized
        self.flat = {k: torch.cat(v).cuda() for k, v in normalized.items()}
        starts, lens = [], []
        offset = 0
        self.offsets = []
        for a in normalized['audio']:
            n = len(a)
            self.offsets.append(offset)
            # Exact official enumeration, including trailing windows of length30.
            count = n - 90 // 3 + 1 if n > 90 else 1
            starts.extend(offset + np.arange(count))
            lens.extend(np.minimum(90, n - np.arange(count)))
            offset += n
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lens, dtype=np.int64)
        self.steps = torch.arange(90, device='cuda')

    def __len__(self):
        return len(self.starts)

    def batch(self, indices):
        lengths = self.lengths[indices]
        t = int(lengths.max())
        lens = torch.from_numpy(lengths).cuda()
        starts = torch.from_numpy(self.starts[indices]).cuda()
        offsets = self.steps[:t][None]
        gather = starts[:, None] + torch.minimum(offsets, lens[:, None]-1)
        valid = offsets < lens[:, None]
        audio, speaker, listener = [self.flat[k][gather] * valid[:, :, None] for k in ['audio', 'speaker', 'listener']]
        return audio, speaker, listener[:, 0], listener[:, 1:].contiguous(), torch.from_numpy(lengths)


def optimize_training(model, method, compile_loss=False):
    """Remove log-only device syncs and vectorize the identical padding mask."""
    import inspect
    import textwrap
    source=inspect.getsource(type(model).get_loss)
    node=ast.parse(textwrap.dedent(source)).body[0]
    class LogsOnly(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='item' and not n.args:
                n.func.attr='detach'
            return n
    node=ast.fix_missing_locations(LogsOnly().visit(node))
    module=sys.modules[type(model).__module__]
    env=dict(module.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<log-sync-only-adapter>','exec'),env)
    model.get_loss=MethodType(env['get_loss'],model)
    if compile_loss:
        model.loss_weights=dict(model.loss_weights)
        model.get_loss=torch.compile(model.get_loss,fullgraph=True,dynamic=False)
    original=model.forward
    def forward(self,audio,driven,init,lengths,*args):
        if method=='rlhg':
            target=args[0] if args else None
            if target is None:return original(audio,driven,init,lengths)
            pred=self.predict(audio,driven,init,lengths)
        else:
            epoch,change=args[:2];target=args[2] if len(args)>2 else None
            if target is None:return original(audio,driven,init,lengths,epoch,change)
            pred=self.predict(audio,driven,init,lengths,epoch,change,target)
        mask=torch.arange(pred.shape[1],device=pred.device)[None]<(lengths.to(pred.device)[:,None]-1)
        pred=pred.masked_fill(~mask[:,:,None],0.)
        target_expand=torch.zeros_like(pred);target_expand[:,:-1]=target
        loss,logs=self.get_loss(pred,target_expand)
        return loss,logs,pred
    model.forward=MethodType(forward,model)


def optimization_parity(model,method,data,out,compile_loss=False):
    state=copy.deepcopy(model.state_dict())
    # Include a short tail in the same batch to check masking and loss weighting.
    tails=np.where(data.lengths<90)[0]
    ids=np.r_[0,1,tails[-2:] if len(tails)>1 else [len(data)-1]]
    batch=data.batch(ids)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    loss,_,prediction=call(model,method,batch,0);loss.backward()
    expected=prediction.detach().clone();expected_loss=loss.detach().clone()
    gradients={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    optimize_training(model,method,compile_loss)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    loss,_,prediction=call(model,method,batch,0);loss.backward()
    diffs=dict(prediction=float((prediction-expected).abs().max()),loss=float((loss-expected_loss).abs()),
        gradient=max(float((p.grad-gradients[k]).abs().max()) for k,p in model.named_parameters() if k in gradients))
    assert max(diffs.values())<1e-5,diffs
    model.load_state_dict(state);model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    save(out/('TRAINING_ADAPTER_PARITY_COMPILED.json' if compile_loss else 'TRAINING_ADAPTER_PARITY.json'),dict(passed=True,max_abs=diffs,gradient_checked=True,
        changes=['Log-only item replaced detach','Vectorized identical >=length-1 zero mask',
                 'Lengths retained on CPU for native packed LSTM'],native_architecture_unchanged=True))


def call(model, method, batch, epoch):
    audio, speaker, initial, target, lengths = batch
    if method == 'rlhg':
        return model(audio, speaker, initial, lengths, target)
    return model(audio, speaker, initial, lengths.cuda(), epoch, 100, target)


def loader_parity(data, out):
    """Compare GPU gather/padding to independent original per-item slicing."""
    candidates = np.unique(np.r_[0, 1, len(data)//2, len(data)-1,
        np.where(data.lengths < 90)[0][:3]])
    batch = data.batch(candidates)
    expected = [[], [], [], []]
    for index in candidates:
        start, n = int(data.starts[index]), int(data.lengths[index])
        flat = {k: torch.cat(v) for k, v in data.host.items()}
        expected[0].append(flat['audio'][start:start+n])
        expected[1].append(flat['speaker'][start:start+n])
        expected[2].append(flat['listener'][start])
        expected[3].append(flat['listener'][start+1:start+n])
    padded = [torch.nn.utils.rnn.pad_sequence(v, batch_first=True) if i != 2 else torch.stack(v)
              for i, v in enumerate(expected)]
    diffs = [float((a.cpu()-b).abs().max()) for a, b in zip(batch[:4], padded)]
    assert max(diffs) == 0, diffs
    save(out / 'LOADER_PARITY.json', dict(passed=True, max_abs=diffs,
        windows=len(data), sampled_indices=candidates.tolist(),
        definition='Independent official-style slicing/pad_sequence, first/rest target split; exact tail windows'))


def inference(model, method, audio, speaker, initial):
    lengths = torch.tensor([len(audio)], dtype=torch.long, device='cuda')
    if method == 'rlhg':
        pred = model(audio[None], speaker[None], initial[None], lengths)
    else:
        pred = model(audio[None], speaker[None], initial[None], lengths, None, None)
    # Official eval removes the last unused step and prepends listener t0.
    return torch.cat([initial[None], pred[0, :len(audio)-1]], 0)


def validation(model, method, data):
    model.eval()
    values = []
    with torch.inference_mode():
        for i, task in enumerate(data.tasks):
            a, s, g = [data.host[k][i].cuda() for k in ['audio', 'speaker', 'listener']]
            pred = inference(model, method, a, s, g[0])
            values.append(float((pred[1:]-g[1:]).abs().mean()))
    model.train()
    return float(np.mean(values))


def atomic_torch(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def train(method, benchmark=False,compile_loss=False):
    out = WORK / method
    out.mkdir(parents=True, exist_ok=True)
    device=gpu_guard(18)
    save(out/('BENCHMARK_DEVICE.json' if benchmark else 'DEVICE.json'),device)
    torch.manual_seed(137)
    np.random.seed(137)
    random.seed(137)
    if benchmark:
        data = Windows('train',dict(np.load(FEATURES/'train_stats.npz'))) if (FEATURES/'COMPLETE.json').exists() else Windows('train', limit=4)
        stats = data.stats
    else:
        complete=read(FEATURES / 'COMPLETE.json')
        assert complete['success'] and sha(FEATURES/'train_stats.npz')==complete['stats_sha256']
        stats = dict(np.load(FEATURES / 'train_stats.npz'))
        data = Windows('train', stats)
    loader_parity(data, out)
    model, config = model_for(method)
    model = model.cuda().train()
    optimization_parity(model,method,data,out,compile_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9, .999), weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.5)
    epochs = 500 if method == 'rlhg' else 200
    batch_size = 16 if method == 'rlhg' else 128
    batches = math.ceil(len(data) / batch_size)
    protocol = dict(method=method, seed=137, max_epochs=epochs, global_batch=batch_size,
        time_size=90, window_stride=1, train_windows=len(data), batches_per_epoch=batches,
        total_steps=batches*epochs, optimizer='AdamW .002 betas .9,.999 wd .05',
        schedule='StepLR gamma .5 every30; step before each epoch as upstream',
        phase_change=100 if method == 'listenformer' else None,
        precision='FP32; no AMP/TF32', torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version(),
        distributed_adaptation='single GPU, same effective global batch; final partial batch has no DistributedSampler duplicate padding',
        teacher_forcing='epoch_index<=100 (zero-based), exactly the upstream predicate' if method=='listenformer' else None,
        dataset='train351 / validation46; one original train row quarantined; test43 held out',
        checkpoint_selection='minimum native free-generation normalized L1 over validation46; every10epochs plus1/final',
        source=str(WORK/'vendor'/method), source_protocol_sha256=sha(WORK/'protocol.json'),
        features_complete_sha256=None if benchmark else sha(FEATURES/'COMPLETE.json'),
        benchmark=benchmark, compiled_loss=compile_loss, implementation_sha256=sha(__file__),
        compatibility=['train-only stats; exact-zero std becomes1',
            'Numerically audited log-sync removal and padding-mask vectorization',
            'ListenFormer missing inference global change_epoch set100' if method == 'listenformer' else 'No model code edits'])
    if benchmark:
        # Uses real clip windows; full-schedule extrapolation uses all train rows.
        full = sum((r['frames']-29 if r['frames'] > 90 else 1)
                   for r in read(FEATURES/'protocol.json')['tasks'] if r['split']=='train')
        durations, losses = [], []
        for step in range(220):
            idx = np.random.default_rng(137+step).integers(0,len(data),size=batch_size)
            torch.cuda.synchronize()
            start = time.time()
            batch = data.batch(idx)
            loss, _, pred = call(model, method, batch, 0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 20:
                durations.append(time.time()-start)
                losses.append(float(loss))
        # Check forward, loss, gradient are finite and inference never accepts GT targets.
        model.eval()
        a, s, initial, target, lengths = data.batch(np.arange(min(batch_size,len(data))))
        n = int(lengths[0])
        with torch.inference_mode():
            p1 = inference(model, method, a[0,:n], s[0,:n], initial[0])
            target.uniform_(-100,100)
            p2 = inference(model, method, a[0,:n], s[0,:n], initial[0])
        assert torch.equal(p1,p2) and torch.isfinite(p1).all()
        save(out/('BENCHMARK_COMPILED.json' if compile_loss else 'BENCHMARK.json'), dict(success=True, diagnostic_only=True, real_clips=len(data.tasks),
            measured_steps=200, median_step_s=float(np.median(durations)), p95_step_s=float(np.percentile(durations,95)),
            mean_step_s=float(np.mean(durations)),
            full_train_windows=full, full_steps=math.ceil(full/batch_size)*epochs,
            estimated_training_hours=float(np.mean(durations)*math.ceil(full/batch_size)*epochs/3600),
            first_loss=losses[0], last_loss=losses[-1], finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
            future_target_perturbation_max_abs=float((p1-p2).abs().max()), protocol=protocol))
        return
    dest = out/'TRAIN_PROTOCOL.json'
    if dest.exists():
        assert read(dest)==protocol, 'Frozen training setup differs'
    else:
        save(dest,protocol)
    val = Windows('validation',stats)
    epoch, offset, total_steps, best = 0,0,0,float('inf')
    if (out/'last.pt').exists():
        ck = torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==sha(dest)
        model.load_state_dict(ck['model'],strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        epoch, offset, total_steps, best = [ck[k] for k in ['epoch','offset','steps','best']]
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy'])
    def checkpoint(e, off):
        atomic_torch(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=e,offset=off,steps=total_steps,best=best,
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_python=random.getstate(),
            rng_numpy=np.random.get_state(),protocol_sha256=sha(dest)))
    begin = time.time()
    last_save = time.time()
    durations=[]
    for ep in range(epoch,epochs):
        if not offset:
            scheduler.step()
        generator=torch.Generator().manual_seed(ep)
        order=torch.randperm(len(data),generator=generator).numpy()
        model.train()
        for batch_index in range(offset,batches):
            t=time.time()
            ids=order[batch_index*batch_size:(batch_index+1)*batch_size]
            loss,_,_=call(model,method,data.batch(ids),ep)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total_steps+=1
            durations.append(time.time()-t)
            if total_steps%100==0:
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss epoch{ep} step{batch_index}')
                median=float(np.median(durations[-100:]))
                save(out/'status.json',dict(stage='training',epoch=ep+1,epochs=epochs,
                    steps=total_steps,total_steps=batches*epochs,loss=float(loss),median_step_s=median,
                    estimated_remaining_h=median*(batches*epochs-total_steps)/3600,elapsed_s=time.time()-begin))
                if time.time()>=FREEZE:
                    checkpoint(ep,batch_index+1)
                    save(out/'INCOMPLETE.json',dict(reason='model freeze deadline',steps=total_steps,full_steps=batches*epochs,formal_result=False))
                    return
            if time.time()-last_save>600:
                checkpoint(ep,batch_index+1);last_save=time.time()
        offset=0
        if ep==0 or (ep+1)%10==0 or ep+1==epochs:
            score=validation(model,method,val)
            with (out/'validation.jsonl').open('a') as f:
                import json
                f.write(json.dumps(dict(epoch=ep+1,score=score,steps=total_steps))+'\n')
            if score<best:
                best=score
                atomic_torch(out/'best.pt',dict(model=model.state_dict(),epoch=ep+1,validation_score=score,
                    steps=total_steps,protocol_sha256=sha(dest)))
        checkpoint(ep+1,0)
        print(method,'epoch',ep+1,'steps',total_steps,'best',best,flush=True)
    save(out/'TRAIN_COMPLETE.json',dict(success=True,epochs=epochs,steps=total_steps,
        best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt'),
        protocol_sha256=sha(dest),formal_training_budget_complete=True))


def generate(method):
    out=WORK/method
    complete=read(out/'TRAIN_COMPLETE.json')
    assert complete['success'] and complete['formal_training_budget_complete']
    assert complete['epochs']==(500 if method=='rlhg' else 200)
    gpu_guard(12)
    stats=dict(np.load(FEATURES/'train_stats.npz'))
    assert sha(FEATURES/'train_stats.npz')==read(FEATURES/'COMPLETE.json')['stats_sha256']
    model,_=model_for(method)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    checkpoint_sha256=sha(out/'best.pt')
    assert ck['protocol_sha256']==sha(out/'TRAIN_PROTOCOL.json')
    model.load_state_dict(ck['model'],strict=True);model.cuda().eval()
    def norm(a,key):
        return torch.from_numpy(((a-stats[key+'_mean'])/stats[key+'_std']).astype(np.float32)).cuda()
    records=[]
    test_input_records={r['id']:r for r in read(FEATURES/'TEST_INPUTS.json')['records']}
    for sample in test_samples():
        uid=sample['id'];dest=out/'predictions'/(uid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        source=test_input_records[uid]
        assert sha(source['speaker'])==source['speaker_sha256']
        assert sha(source['initial'])==source['initial_sha256']
        speaker=np.load(source['speaker'])
        # This file physically contains only t0; no future GT sequence is opened.
        initial=dict(np.load(source['initial']))
        audio=np.load(FEATURES/'test'/uid/'audio.npy')
        n=len(audio)
        s=norm(dynamic(speaker),'dynamic');a=norm(audio,'audio');init=norm(dynamic(initial),'dynamic')[0]
        with torch.inference_mode():pred=inference(model,method,a,s,init).cpu().numpy()
        pred=pred*stats['dynamic_std']+stats['dynamic_mean']
        assert pred.shape==(n,73) and np.isfinite(pred).all()
        motion=pred[:,np.r_[3:67,0:3,67:70]]
        np.savez_compressed(dest,dynamic=pred,motion=motion,initial_coeff=initial['coeff'],
            initial_transform_params=initial['transform_params'])
        rec=dict(id=uid,path=str(dest),sha256=sha(dest),frames=n,
            listener_frames_used=1,future_listener_dynamic=False,future_listener_crop=False,
            checkpoint_sha256=checkpoint_sha256)
        save(dest.with_suffix('.json'),rec);records.append(rec)
        print(method,'generated',len(records),uid,flush=True)
    save(out/'GENERATION_COMPLETE.json',dict(success=True,outputs=43,records=records,
        train_complete_sha256=sha(out/'TRAIN_COMPLETE.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['rlhg','listenformer'])
    p.add_argument('stage',choices=['benchmark','train','generate']);p.add_argument('--compile-loss',action='store_true');a=p.parse_args()
    if a.stage=='generate':generate(a.method)
    else:train(a.method,a.stage=='benchmark',a.compile_loss)
