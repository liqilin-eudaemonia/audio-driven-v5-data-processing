"""Strict model-produced schedules, rendered with the existing guidance template."""
import json
import math
from p2_listen.contracts import digest, safe_id
from p2_plan_cache.protocol import render_clean
from p2_guided_prompt_en.formatting import is_english_script

SYSTEM = '''You plan a listener's natural backchannel during the supplied speaker audio.
The image is the listener's reference first frame. The audio is ONLY the other
participant speaking. Plan the listener's facial/head responses and brief vocal
feedback, preserving listening rather than taking over the conversation.
Use only these inputs. Do not transcribe the speaker as the listener's speech.
You may choose not to vocalize. All descriptions and proposed vocal words must
be English. Timings are in seconds from the clip start, inside its duration.
Return only one JSON object: {"events": [{"modality": "visual" or "audio",
"start": number, "end": number, "description": string or null,
"text": string or null}]}. Visual events require a description and null text.
Audio events describe the listener's sound; text contains exact words to say
or null for nonverbal sounds. Explicit silence may be described with null text.
An empty events array is allowed. Do not output markdown, reasoning, labels,
sample IDs, speaker text, or keys other than the stated schema.'''
DEFAULTS = dict(seed=20261028, max_new_tokens=2048, max_attempts=3,
                max_events=64, max_qwen_tokens=32768, max_text_tokens=8192,
                do_sample=False, qwen_lora=False)


def instruction(duration, previous_error=None):
    text = f'Plan the listener response throughout this {duration!r}-second clip.'
    if previous_error:
        text += '\nThe previous attempt failed validation: '+previous_error[:500]+'. Return a corrected complete JSON schedule.'
    return text


def _pairs(items):
    d = {}
    for k, v in items:
        if k in d: raise ValueError('Duplicate JSON key: '+k)
        d[k] = v
    return d


def parse_schedule(raw, sid, duration, max_events=64):
    safe_id(sid)
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid input duration')
    if '<|' in raw or '|>' in raw: raise ValueError('Model control tokens forbidden')
    value = json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
    if not isinstance(value, dict) or set(value) != {'events'}:
        raise ValueError('Expected exactly an events object')
    if not isinstance(value['events'], list) or len(value['events']) > max_events:
        raise ValueError('Invalid event count')
    events = []
    for i, e in enumerate(value['events']):
        if not isinstance(e, dict) or set(e) != {'modality','start','end','description','text'}:
            raise ValueError('Unexpected event fields')
        if e['modality'] not in ('visual', 'audio'): raise ValueError('Invalid modality')
        a, b = e['start'], e['end']
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (a,b)) or not 0 <= a < b <= duration:
            raise ValueError('Event interval outside clip')
        for k in ('description','text'):
            v=e[k]
            if v is not None and (not isinstance(v,str) or not v.strip() or len(v)>4000 or not is_english_script(v)):
                raise ValueError('Expected nonempty English text or null: '+k)
        if e['modality']=='visual' and (not e['description'] or e['text'] is not None):
            raise ValueError('Visual event needs description and null text')
        if e['modality']=='audio' and not (e['description'] or e['text']):
            raise ValueError('Audio event needs content')
        events.append(dict(e, event_id=f'plan_{i:04d}'))
    # Schema compatibility is only for the formatter, never evidence of annotation access.
    record=dict(schema='listen-existing-result-fusion-verbatim-v2',sample_id=sid,duration_s=duration,events=events)
    guidance=render_clean(record)
    return dict(schema='p2-autonomous-plan-v1',sample_id=sid,raw=raw,schedule=value,
                guidance=guidance,prompt=guidance['prompt'],prompt_sha256=digest(guidance['prompt']),
                plan_source='qwen_autonomous',target_annotation_access=False,gt_media_access=False)


def checked_continuation(tokenizer, ids, prefix_tokens, eos):
    tail=ids[0,prefix_tokens:].tolist()
    if eos not in tail: raise ValueError('Planner reached token limit without EOS; no truncation allowed')
    # Reject premature control tokens rather than hide them during decode.
    tail=tail[:tail.index(eos)]
    if any(t in tokenizer.all_special_ids for t in tail): raise ValueError('Unexpected model control token')
    return tokenizer.decode(tail,skip_special_tokens=False).strip()


def generate_attempt(runtime, processor, obs, options, attempt, previous_error=None):
    import torch
    from PIL import Image
    from omni_av.data import read_audio
    from omni_av.omni import move_inputs
    model=runtime.thinker;cfg=runtime.config;device=next(model.parameters()).device
    if model.training or any(p.requires_grad for p in model.parameters()): raise ValueError('Planner must be frozen')
    wave=read_audio(obs['speaker_audio'],end_s=obs['requested_duration_s'])
    if len(wave)!=round(obs['requested_duration_s']*16000): raise ValueError('Speaker audio duration differs')
    with Image.open(obs['listener_image']) as im: image=im.convert('RGB')
    messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),
              dict(role='user',content=[dict(type='image'),dict(type='audio'),
                   dict(type='text',text=instruction(obs['requested_duration_s'],previous_error))])]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=move_inputs(dict(processor(text=[prompt],images=[image],audio=[wave],sampling_rate=16000,
        return_tensors='pt',padding=True,truncation=False,
        images_kwargs={'max_pixels':cfg.vision_max_pixels,'min_pixels':28*28*4})),device)
    n=inputs['input_ids'].shape[1]
    if n+options['max_new_tokens']>min(options['max_qwen_tokens'],model.config.text_config.max_position_embeddings):
        raise ValueError('Planning request exceeds context; no truncation allowed')
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos==processor.tokenizer.unk_token_id or eos not in processor.tokenizer.all_special_ids:
        raise ValueError('Missing Qwen end-of-message token')
    with torch.random.fork_rng(devices=[device]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        torch.manual_seed(options['seed']);model.rope_deltas=None
        generated=model.generate(**inputs,max_new_tokens=options['max_new_tokens'],do_sample=False,
            use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id,use_cache=True)
    raw=processor.tokenizer.decode(generated[0,n:],skip_special_tokens=False)
    audit=dict(attempt=attempt,request=messages,raw_with_special_tokens=raw,prefix_tokens=n,
               generated_token_ids=generated[0,n:].tolist(),planner_seed=options['seed'])
    try:
        clean=checked_continuation(processor.tokenizer,generated,n,eos)
        result=parse_schedule(clean,obs['sample_id'],obs['requested_duration_s'],options['max_events'])
        return result,dict(audit,status='accepted')
    except ValueError as e:
        return None,dict(audit,status='rejected',error=str(e))
