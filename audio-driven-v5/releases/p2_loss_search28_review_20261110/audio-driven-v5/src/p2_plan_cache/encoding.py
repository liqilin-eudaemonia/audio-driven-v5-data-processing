"""Append the exact Gemma prompt to the existing frozen multimodal prefix."""
import torch

from p2_listen.cache import cpu
from p2_listen.contracts import digest


def text_length(tokens, maximum=8192):
    buckets=(1024,2048,4096,8192)
    if maximum not in buckets or type(tokens) is not int or tokens<=0:raise ValueError('Invalid text length policy')
    if tokens>maximum:raise OverflowError('Full guidance exceeds configured Gemma context; no truncation')
    return next(n for n in buckets if tokens<=n)


def encode_text(encoder,prompt,maximum):
    wrapper=encoder.encoder.tokenizer
    ids=wrapper.tokenizer(prompt.strip(),truncation=False)['input_ids']
    length=text_length(len(ids),maximum)
    if length>encoder.encoder.model.config.text_config.max_position_embeddings:
        raise ValueError('Gemma positional support exceeded')
    old=wrapper.max_length
    try:
        wrapper.max_length=length
        pairs=wrapper.tokenize_with_weights(prompt)['gemma']
        if [int(t) for t,valid in pairs if int(valid)]!=ids or len(pairs)!=length:
            raise ValueError('Tokenizer changed or truncated guidance')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden,mask=encoder.encoder.encode(prompt)
            if int(mask.sum())!=len(ids):raise ValueError('Gemma did not receive all prompt tokens')
            result=encoder.processor.process_hidden_states(hidden,mask)
        return dict(video=result.video_encoding.detach(),audio=result.audio_encoding.detach(),
                    mask=result.attention_mask.bool(),prompt=prompt),dict(tokens=len(ids),sequence_length=length,truncated=False)
    finally:
        wrapper.max_length=old


def append_plan(pref, tokenizer, embed, prompt, maximum, model_maximum):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or set(ids) & set(tokenizer.all_special_ids):
        raise ValueError('Empty plan or model special token inside plan')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end is None or end == tokenizer.unk_token_id or end not in tokenizer.all_special_ids:
        raise ValueError('Missing native Qwen end-of-message token')
    prefix = pref['prefix']; valid = pref['prefix_valid']; positions = pref['prefix_positions']
    if prefix.shape[0] != 1 or valid.shape != prefix.shape[:2] or not valid.all():
        raise ValueError('Expected one unpadded native input prefix')
    if positions.shape != (3, 1, prefix.shape[1]):
        raise ValueError('Unexpected Qwen multimodal position shape')
    n = prefix.shape[1]; total = n+len(ids)+1
    next_position = int(positions.max())+1
    if total > min(maximum, model_maximum) or next_position+len(ids)+1 > model_maximum:
        raise OverflowError('Qwen input plus complete plan exceeds context; truncation is forbidden')
    token_ids = torch.tensor([ids+[end]], dtype=torch.long, device=prefix.device)
    suffix = embed(token_ids).to(prefix)
    tail = torch.arange(next_position, next_position+token_ids.shape[1], device=positions.device,
                        dtype=positions.dtype)[None, None].expand(3, 1, -1)
    full_valid = torch.cat([valid.bool(), torch.ones_like(token_ids, dtype=torch.bool)], dim=1)
    plan_mask = torch.zeros_like(full_valid); plan_mask[:, n:n+len(ids)] = True
    return dict(prefix=torch.cat([prefix, suffix], dim=1),
                prefix_valid=full_valid, prefix_positions=torch.cat([positions, tail], dim=2),
                plan_mask=plan_mask, plan_token_ids=token_ids[:, :-1],
                prefix_tokens=n, plan_tokens=len(ids), terminal_tokens=1)


@torch.no_grad()
def extract_planned_input(c, obs, prompt, runtime, processor, maximum, *, pref=None):
    from p3_listen.features import prefix
    from p2_listen.features import make_observation, image_grid
    from p2_listen.layout import make_layout
    from omni_av_p2.data import select_hidden_layers
    from omni_av_p2.connector import ConditionInputs
    model = runtime.thinker; cfg = runtime.config
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Qwen must be frozen and in eval mode')
    if pref is None:pref = prefix(runtime, processor, make_observation(obs))
    full = append_plan(pref, processor.tokenizer, model.get_input_embeddings(), prompt,
                       maximum, model.config.text_config.max_position_embeddings)
    out = model.model(inputs_embeds=full['prefix'], position_ids=full['prefix_positions'],
                      attention_mask=full['prefix_valid'].long(), output_hidden_states=True,
                      use_cache=False, return_dict=True)
    hidden = select_hidden_layers(out.hidden_states, cfg.hidden_layers,
                                  model.config.text_config.num_hidden_layers)
    _, _, grid = image_grid(obs, processor, cfg)
    layout = make_layout(obs['requested_duration_s'], grid, cfg.vision_fps, cfg.audio_hz)
    condition = dict(hidden=hidden, hidden_valid=full['prefix_valid'],
                     layer_ids=torch.tensor(cfg.hidden_layers, device=hidden.device))
    for mod in ('video', 'audio'):
        for key in ('_coordinates', '_valid'):
            condition[mod+key] = layout[mod+key].to(hidden.device)
    ConditionInputs(**condition).validate()
    if hidden.shape[2] != full['prefix_valid'].shape[1] or not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite or truncated Qwen hidden')
    return cpu(dict(condition=condition, layout=layout, prompt=prompt, prompt_sha256=digest(prompt),
                    plan_mask=full['plan_mask'], plan_token_ids=full['plan_token_ids'],
                    position_ids=full['prefix_positions'], prefix_tokens=full['prefix_tokens'],
                    plan_tokens=full['plan_tokens'], terminal_tokens=1,
                    qwen_frozen=True, truncated=False))
