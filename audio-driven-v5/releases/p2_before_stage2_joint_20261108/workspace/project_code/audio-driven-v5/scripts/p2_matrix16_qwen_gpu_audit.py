#!/usr/bin/env python3
"""Real Qwen online-gradient smoke; surrogate AV objective, not real LTX S0."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    import torch
    from p2_matrix16.config import load,data_config
    from p2_listen.config import extractor_config
    from p2_plan_cache.reader import PlanCacheReader
    from p2_matrix16.data import metadata_targets
    from p2_matrix16.model import MatrixModel
    from p2_matrix16.teacher import identity
    from p2_listen.cache import load_part
    from p2_listen.contracts import atomic_json,read,EVENTS
    from p2_matrix16.training import gradient_audit,objective
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--plan-cache',required=True);p.add_argument('--sample-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--real-ltx',action='store_true');p.add_argument('--memory-limit-gib',type=float,default=48)
    a=p.parse_args();c=load(a.config);d=data_config(c);cfg=extractor_config(d)
    from omni_av_p2.data import load_omni
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(a.memory_limit_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    reader=PlanCacheReader(a.plan_cache);item=read(reader.work/'samples'/(a.sample_id+'.json'));obs=item['observation']
    target=metadata_targets(d,[obs])[a.sample_id]
    b=reader.training_sample(a.sample_id,target,'A2','cuda:0')
    teacher=load_part(Path(c['teacher_cache'])/'cache'/a.sample_id,identity(d,obs,target,c['teacher_cache']),'cuda:0')
    b.update(teacher={k:v for k,v in teacher.items() if k in ('video','audio','video_valid','audio_valid')},
             condition=None,observation=obs,plan=item['guidance']['prompt'])
    runtime,processor=load_omni(cfg,'cuda:0')
    renderer=None
    if a.real_ltx:
        from omni_av_p2.ltx import load_renderer,append_conditions
        from omni_av.ltx_runtime import with_text
        from p2_listen.performance import configure_renderer
        from p2_listen.runtime import av_loss
        renderer,strategy,sampler=load_renderer(cfg,'cuda:0');configure_renderer(renderer,'all')
    torch.manual_seed(31);model=MatrixModel(c,cfg,renderer,runtime,processor).to('cuda:0').train()
    # Seed nonzero outputs to simulate a trained Stage1 parent for path auditing.
    with torch.no_grad():
        for module in (model.connector,model.decoders):
            for p in module.parameters():
                if p.numel()>1 and not p.any():p.normal_(std=.001)
    frozen_versions={n:p._version for n,p in runtime.thinker.named_parameters() if not p.requires_grad}
    def surrogate(u,*args):return (u.video-.2).square().mean()+(u.audio+.1).square().mean(),{}
    def real_av(u,pack,text):
        prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
        return av_loss(renderer,append_conditions(prepared,u),pack['support'],return_components=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        v=model(b,{k:1. for k in EVENTS},torch.Generator(device='cuda:0').manual_seed(61),real_av if a.real_ltx else surrogate)
        loss,_,_=objective(v,c,v['event_counts'],1,1)
    paths=gradient_audit(model,v,c,v['event_counts'])
    for key in ('av','event','feature'):
        if paths['qwen_lora'][key]<=0:raise AssertionError('Missing '+key+' Qwen gradient')
    loss.backward();optimizer=torch.optim.AdamW(model.groups(),weight_decay=0);optimizer.step()
    for n,p in runtime.thinker.named_parameters():
        if n in frozen_versions and (p._version!=frozen_versions[n] or p.grad is not None):raise AssertionError('Base updated')
    report=dict(success=True,real_qwen=True,real_postprocessed_teacher=True,real_ltx=a.real_ltx,
        av_objective='native AV flow matching' if a.real_ltx else 'surrogate U MSE only',sample_id=a.sample_id,paths=paths,
        losses=dict(av=float(v['av'].detach()),features={k:float(x.detach()) for k,x in v['features'].items()}),
        qwen_targets=len(model.qwen_targets),frozen_base_unchanged=True,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    atomic_json(a.output,report);print(json.dumps(report))

if __name__=='__main__':main()
