"""Online differentiable text backbone; audio/image prefix stays frozen."""
from types import SimpleNamespace
import torch
from torch import nn


def inject_qwen_lora(runtime,spec):
    from omni_av_p21.lora import LowRankUpdate,AdaptedLinear
    model=runtime.thinker
    if any(isinstance(m,AdaptedLinear) for m in model.modules()):raise ValueError('Qwen already adapted')
    model.requires_grad_(False).eval()
    limit=max(runtime.config.hidden_layers)
    if limit>=len(model.model.layers):raise ValueError('Interior hidden layers required')
    targets=[]
    for i,layer in enumerate(model.model.layers[:limit]):
        for name in spec['targets']:
            base=getattr(layer.self_attn,name)
            if not isinstance(base,nn.Linear):raise ValueError('Unexpected Qwen projection')
            targets.append((f'model.layers.{i}.self_attn.{name}',base))
    adapters=nn.ModuleList()
    with torch.random.fork_rng(devices=[]):
        for name,base in targets:
            update=LowRankUpdate(base,SimpleNamespace(rank=spec['rank'],alpha=spec['alpha']))
            owner,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(owner),leaf,AdaptedLinear(base,update))
            adapters.append(update)
    if spec.get('gradient_checkpointing'):
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return adapters,[n for n,_ in targets]


def online_condition(runtime,processor,obs,prompt,maximum=32768):
    from p3_listen.features import prefix
    from p2_listen.features import make_observation,image_grid
    from p2_listen.layout import make_layout
    from p2_plan_cache.encoding import append_plan
    from omni_av_p2.connector import ConditionInputs
    pref=prefix(runtime,processor,make_observation(obs))
    model=runtime.thinker;cfg=runtime.config
    if prompt is not None:
        pref=append_plan(pref,processor.tokenizer,model.get_input_embeddings(),prompt,maximum,model.config.text_config.max_position_embeddings)
    if pref['prefix'].shape[1]>maximum:raise ValueError('Online Qwen context exceeds limit')
    out=model.model(inputs_embeds=pref['prefix'],position_ids=pref['prefix_positions'],
        attention_mask=pref['prefix_valid'].long(),output_hidden_states=True,use_cache=False,return_dict=True)
    if len(out.hidden_states)!=model.config.text_config.num_hidden_layers+1:raise ValueError('Native hidden tuple differs')
    # Deliberately no detach: selected block outputs must update the LoRA parameters.
    h=torch.stack([out.hidden_states[i] for i in cfg.hidden_layers],dim=1)
    _,_,grid=image_grid(obs,processor,cfg)
    layout=make_layout(obs['requested_duration_s'],grid,cfg.vision_fps,cfg.audio_hz)
    x=ConditionInputs(h,pref['prefix_valid'],layout['video_coordinates'].to(h.device),layout['audio_coordinates'].to(h.device),
        layout['video_valid'].to(h.device),layout['audio_valid'].to(h.device),torch.tensor(cfg.hidden_layers,device=h.device))
    x.validate()
    return x
