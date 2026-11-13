"""One online generation condition; separate planning-prompt CE on the same LoRA."""
import torch
from torch import nn
from p2_ltx_events.heads import objective as generation_objective


class JointModel(nn.Module):
    def __init__(self,generator,qwen,qwen_lr=1e-6,generation_lr_scale=1/3):
        super().__init__();self.generator=generator;self.qwen=qwen
        self.qwen_lr=qwen_lr;self.generation_lr_scale=generation_lr_scale
        self.assert_ownership()

    def assert_ownership(self):
        self.generator.assert_ownership();self.qwen.assert_ownership()
        a={id(p)for p in self.generator.parameters()if p.requires_grad}
        b={id(p)for p in self.qwen.parameters()if p.requires_grad}
        if a&b:raise ValueError('Shared optimizer ownership between Qwen and generator')

    def groups(self):
        groups=[]
        for g in self.generator.groups():groups.append(dict(g,lr=g['lr']*self.generation_lr_scale))
        groups.append(dict(name='qwen',params=list(self.qwen.adapters),lr=self.qwen_lr))
        return groups

    def forward(self,bundle,pos_weights,rng,strategy,sampler):
        if bundle.get('condition')is not None:raise ValueError('Stage2 forbids frozen cached Qwen hidden')
        condition=self.qwen.condition(bundle['qwen_generation_inputs'],bundle['generation_prompt'],bundle['layout'])
        generation=self.generator(dict(bundle,condition=condition),pos_weights,rng,strategy,sampler)
        if bundle.get('planning_valid',True):
            total,tokens=self.qwen.planning_loss(bundle['qwen_planning_inputs'],bundle['planning_target'])
        else:
            total=sum(p.sum()*0 for p in self.qwen.adapters);tokens=0
        generation.update(planning_sum=total,planning_tokens=tokens)
        return generation


def objective(values,c,event_counts,global_planning_tokens,accumulation=1,world_size=1):
    total,parts,event=generation_objective(values,c,event_counts,accumulation,world_size)
    # Native training SUM-reduces gradients; CE uses the GLOBAL answer-token denominator.
    if global_planning_tokens<0:raise ValueError('Invalid global planning denominator')
    plan=values['planning_sum']/max(global_planning_tokens,1)
    return total+c['planning_weight']*plan,dict(parts,planning=plan),event
