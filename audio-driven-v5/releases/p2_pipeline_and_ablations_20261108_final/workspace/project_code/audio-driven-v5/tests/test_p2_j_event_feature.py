from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from p2_listen.contracts import ROOT,read
from p2_matrix16.config import load
from p2_matrix16.training import objective,gradient_audit
from p2_j_event_feature_launch import verify_metrics
from test_p2_matrix16 import setup,forward


def config():return load(ROOT/'configs/p2_j_base_h_av_event_feature_28gpu_20261103.json',formal=True)


def test_comparison_contract_and_weights():
    c=config();base=read(ROOT/'experiments/p2_j_guided_av_feature_base025_20261102/duration_memorysafe_save500/config.json')
    assert c['training']['global_batch']==28 and c['training']['steps']==10202
    assert c['duration_sampling']['start_step']==0 and c['duration_sampling']['window']==448
    for k in ['seed','connector_lr','feature_lr','event_lr','save_every','gradient_checkpointing']:
        assert c['training'][k]==base['training'][k]
    assert c['qwen_hidden_source']=='base_pre_sft' and c['j_cache']==c['plan_cache']
    assert c['event_head_training'] and c['loss']=='av_event_feature'
    a=torch.tensor(2.,requires_grad=True);e=torch.tensor(.2,requires_grad=True)
    v=torch.tensor(4.,requires_grad=True);u=torch.tensor(6.,requires_grad=True)
    loss,_,_=objective(dict(av=a,features=dict(video=v,audio=u),event_sums=dict(x=e)),c,dict(x=torch.tensor(1.)),1,1)
    assert loss.item()==pytest.approx(2.26)
    loss.backward()
    assert a.grad.item()==1 and e.grad.item()==pytest.approx(.05)
    assert v.grad.item()==u.grad.item()==pytest.approx(.025)


def test_all_three_losses_reach_expected_groups():
    c=config();c.pop('event_supervision');c['connector']=dict(fusion_width=16,fusion_depth=2,fusion_heads=4)
    model,bundle,pref,av=setup(c);values=forward(model,bundle,pref,av)
    norms=gradient_audit(model,values,c,values['event_counts'])
    for k in ['av','event','feature']:assert norms['connector'][k]>0
    assert norms['text_lora']['av']>0
    assert norms['text_lora']['event']==norms['text_lora']['feature']==0
    loss,_,_=objective(values,c,values['event_counts'],1,1);loss.backward()
    for module in [model.connector,model.event_heads,model.decoders,model.text_lora]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_launch_gate_rejects_missing_branch():
    rows=[dict(step=i,av=1.,event=.2,feature_video=2.,feature_audio=2.,weighted_total=1.11,
               seconds=4.,peak_reserved_gib=90.,grad_norm=dict(connector=.1,events=.1,features=.1,text_lora=.1)) for i in range(1,17)]
    assert verify_metrics(rows,16)['success']
    for r in rows:r['grad_norm']['events']=0
    with pytest.raises(ValueError):verify_metrics(rows,16)


def test_heterogeneous_commands_never_use_busy_cards():
    from p2_j_event_feature_launch import commands
    cluster=read(ROOT/'configs/p2_j_base_h_av_event_feature_cluster28_20261103.json')
    cmds=commands(cluster,ROOT/'configs/p2_j_base_h_av_event_feature_28gpu_20261103.json')
    assert [next(x for x in c if x.startswith('--nproc_per_node=')) for c in cmds]==[
        '--nproc_per_node=8','--nproc_per_node=8','--nproc_per_node=8','--nproc_per_node=4']
    assert 'CUDA_VISIBLE_DEVICES=0,1,2,3' in cmds[-1]
    assert all('TORCH_ELASTIC_WORKER_IDENTICAL=0' in c for c in cmds)
    assert cluster['network']['NCCL_IB_HCA']=='=mlx5_0:1'
