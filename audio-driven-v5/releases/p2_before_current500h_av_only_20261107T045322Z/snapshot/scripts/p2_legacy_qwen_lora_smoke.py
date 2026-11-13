#!/usr/bin/env python3
"""Real longest-clip AV/Event/Feature backward from an actual legacy parent."""
from pathlib import Path
import argparse,sys,json,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    import torch
    from p2_listen.config import extractor_config
    from p2_listen.contracts import atomic_json,EVENTS
    from p2_matrix16.config import load
    from p2_matrix16.model import MatrixModel
    from p2_matrix16.training import objective,gradient_audit
    from p2_legacy_qwen_lora import backend
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.performance import BucketGradientAccumulator,configure_renderer
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=load(a.config);d=backend.data_config(c);cfg=extractor_config(d);configure_training_numerics()
    from omni_av_p2.ltx import load_renderer,append_conditions
    from omni_av_p2.data import load_omni
    from omni_av.ltx_runtime import with_text
    from p2_listen.runtime import av_loss
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(64*2**30/torch.cuda.get_device_properties(0).total_memory)
    dataset=backend.Dataset(c);index=max(range(len(dataset.rows)),key=lambda i:dataset.rows[i]['requested_duration_s'])
    b=dataset.sample(index,'cuda:0');renderer,strategy,sampler=load_renderer(cfg,'cuda:0');configure_renderer(renderer,'all')
    runtime,processor=load_omni(cfg,'cuda:0');torch.manual_seed(c['training']['seed'])
    model=MatrixModel(c,cfg,renderer,runtime,processor).to('cuda:0');parent=backend.initialize_stage2(Path(c['legacy_parent'])/'step_001268.pt',model,c,backend.data_identity(c));model.train()
    opt=torch.optim.AdamW(model.groups(),weight_decay=0);acc=BucketGradientAccumulator(model.parameters(),32)
    def av(u,pack,text):return av_loss(renderer,append_conditions(strategy.prepare_training_inputs(with_text(pack['av'],text),sampler),u),pack['support'],return_components=True)
    start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        values=model(b,{k:dataset.parent['statistics'][k]['pos_weight'] for k in EVENTS},torch.Generator(device='cuda:0').manual_seed(91),av)
        loss,_,_=objective(values,c,values['event_counts'],1,1)
    paths=gradient_audit(model,values,c,values['event_counts'])
    for key in ('av','event','feature'):
        if paths['qwen_lora'].get(key,0)<=0:raise AssertionError('Missing Qwen path '+key)
    loss.backward();acc.add();acc.finish(1);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();model.assert_ownership()
    result=dict(success=True,sample_id=dataset.rows[index]['sample_id'],duration_s=dataset.rows[index]['requested_duration_s'],parent=parent,paths=paths,
                loss=float(loss.detach()),seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,original_audio=True,formal_training=False)
    atomic_json(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
