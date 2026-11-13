"""Checkpoint restoration and online conditions for external evaluation runners.

This module never reads targets, annotations, recipes or training hidden caches.
Planning/text encoding is supplied by the caller and explicitly attributed.
"""
import torch
from .config import data_config,source_identity
from .checkpoint import checked
from .model import MatrixModel


def restore(path,renderer,runtime,processor,device='cuda:0',*,backend=None):
    from p2_listen.config import extractor_config
    from p2_listen.features import weights_identity
    saved=checked(path);identity=saved['identity'];c=identity['experiment']
    d=(backend.data_config if backend is not None else data_config)(c)
    if identity['source']!=(backend.source_identity if backend is not None else source_identity)():raise ValueError('Checkpoint source changed')
    if identity['assets']!={k:weights_identity(d,k) for k in ('omni','ltx','gemma')}:
        raise ValueError('Pretrained inference assets changed')
    model=MatrixModel(c,extractor_config(d),renderer,runtime if c['stage']==2 else None,processor).to(device)
    if model.qwen_targets!=identity['qwen_targets'] or model.text_targets!=identity['text_targets']:
        raise ValueError('Adapter target layers differ')
    model.load_state_dict(saved['model'],strict=True)
    model.eval().requires_grad_(False)
    renderer.eval().requires_grad_(False);runtime.thinker.eval().requires_grad_(False)
    return model


@torch.no_grad()
def conditions(model,runtime,processor,observation,text,*,plan=None,plan_source='none'):
    from p2_listen.contracts import observation as validate_observation,digest
    from .qwen import online_condition
    obs=validate_observation(observation)
    guided=model.spec['text_condition']=='guided'
    if guided:
        if not plan or plan_source not in ('predicted','annotation_oracle') or text['prompt']!=plan:
            raise ValueError('Qwen/Gemma require the exact same attributed plan')
    else:
        from p3_listen.prompts import GEMMA
        if plan is not None or plan_source!='none' or text['prompt']!=GEMMA:raise ValueError('Fixed arm requires its fixed text and no plan')
    if model.spec['stage']==2 and runtime is not model.qwen_runtime:raise ValueError('Use the restored Qwen adapters')
    if model.training:raise ValueError('Inference model must be in eval mode')
    x=online_condition(runtime,processor,obs,plan)
    return model.connector(x),dict(plan_source=plan_source,plan_sha256=digest(plan),
        target_annotation_access=plan_source=='annotation_oracle',qwen_lora=model.spec['stage']==2)
