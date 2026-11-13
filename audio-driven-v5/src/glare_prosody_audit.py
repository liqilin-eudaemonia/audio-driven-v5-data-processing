"""Auditable GLARE prosody parsing; invalid replies never become negatives."""
import math
import re

SCHEMA='glare-prosody-audit-v2'
LINE=re.compile(r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*')
ACCEPTED={'events','explicit_null','below_threshold'}


def parse_reply(raw, duration, threshold=.5, truncated=False):
    if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid audio duration')
    if not 0<=threshold<=1:raise ValueError('Invalid intensity threshold')
    result=dict(schema=SCHEMA,raw_text=raw,status=None,accepted=False,normalized_text=None,
                events=[],below_threshold=[],errors=[],duration_s=duration,threshold=threshold,
                format_repairs=[],canonical_text=None)
    if truncated:
        result.update(status='truncated',errors=['generation_reached_token_limit_without_eos']);return result
    if not isinstance(raw,str) or not raw.strip():
        result.update(status='empty_output',errors=['empty_model_reply']);return result
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):
        lines=text.splitlines()
        if len(lines)>=3 and lines[0].strip() in ('```','```text','```csv') and lines[-1].strip()=='```':
            text='\n'.join(lines[1:-1]).strip();result['format_repairs'].append('outer_code_fence')
    if re.search(r'\\+n',text):
        text=re.sub(r'\\+n','\n',text);result['format_repairs'].append('escaped_newlines')
    if text.upper()=='NULL':
        result.update(status='explicit_null',accepted=True,normalized_text='NULL',canonical_text='NULL');return result
    matches=list(LINE.finditer(text))
    cursor=0;parts=[]
    for match in matches:
        if text[cursor:match.start()].strip(' \t\r\n;'):
            result['errors'].append(dict(reason='non_numeric_text_or_mixed_null',text=text[cursor:match.start()]))
        parts.append(', '.join(match.groups()));cursor=match.end()
    if not parts or text[cursor:].strip(' \t\r\n;'):
        result['errors'].append(dict(reason='invalid_format_or_non_numeric_suffix',text=text[cursor:]))
    if result['errors']:
        result['status']='parse_error';return result
    canonical='\n'.join(parts)
    if len(parts)!=len([line for line in text.splitlines() if line.strip()]):
        result['format_repairs'].append('numeric_triplets_reflowed')
    result['canonical_text']=canonical
    previous_end=-math.inf
    for number,line in enumerate(canonical.splitlines(),1):
        if not line.strip():continue
        match=LINE.fullmatch(line)
        if not match:
            result['errors'].append(dict(line=number,reason='invalid_format_or_mixed_null',text=line));continue
        start,end,score=map(float,match.groups())
        reason=None
        if not all(math.isfinite(v) for v in (start,end,score)):reason='nonfinite_value'
        elif not 0<=start<end<=duration+1e-6:reason='invalid_time_interval'
        elif not 0<=score<=1:reason='score_out_of_range'
        elif start<previous_end-1e-8:reason='overlap_or_unsorted_intervals'
        if reason:
            result['errors'].append(dict(line=number,reason=reason,text=line));continue
        previous_end=end
        event=dict(start=float(f'{start:.2f}'),end=float(f'{end:.2f}'),score=float(f'{score:.2f}'))
        if score<threshold:result['below_threshold'].append(event);continue
        if event['end']<=event['start']:
            result['errors'].append(dict(line=number,reason='interval_collapses_after_native_rounding',text=line));continue
        result['events'].append(event)
    if result['errors']:
        result['status']='parse_error';return result
    result['accepted']=True
    if result['events']:
        result['status']='events'
        result['normalized_text']='\n'.join(f"{e['start']:.2f}, {e['end']:.2f}, {e['score']:.2f}" for e in result['events'])
    else:result.update(status='below_threshold',normalized_text='NULL')
    return result


def frame_targets(result, frames, fps=25):
    """Same floor/ceil and score rounding as the native frame-vector method."""
    if not result.get('accepted') or result.get('status') not in ACCEPTED:
        raise ValueError('Unverified prosody must not become an all-zero target')
    if frames<=0 or fps<=0:raise ValueError('Invalid frame grid')
    import numpy as np
    scores=np.zeros((frames,1),dtype=np.float32)
    for event in result['events']:
        first=max(0,math.floor(event['start']*fps))
        end=min(frames,math.ceil(event['end']*fps))
        if end>first:scores[first:end,0]=event['score']
    return scores,np.ones((frames,1),dtype=bool)


def attach_audit(encoder):
    """Observe raw native generation without editing frozen upstream files."""
    import inspect
    from types import MethodType
    original_generate=encoder.model.generate
    def tower_hook(module,inputs,output):
        encoder.audit_generation['audio_tower_calls']=encoder.audit_generation.get('audio_tower_calls',0)+1
        encoder.audit_generation['audio_hidden_shape']=list(output.last_hidden_state.shape)
    encoder.audit_tower_hook=encoder.model.audio_tower.register_forward_hook(tower_hook)
    original_merge=encoder.model._merge_input_ids_with_audio_features
    def merge(*args,**kwargs):
        encoder.audit_generation['audio_merged_into_language_input']=True
        encoder.audit_generation['audio_output_lengths']=args[1].tolist()
        return original_merge(*args,**kwargs)
    encoder.model._merge_input_ids_with_audio_features=merge
    def generate(*args,**kwargs):
        features=kwargs.get('input_features');mask=kwargs.get('feature_attention_mask')
        assert features is not None and mask is not None,'Audio features were not supplied to generation'
        count=int((kwargs['input_ids']==encoder.model.config.audio_token_index).sum())
        assert count==1 and int(mask.sum())>0,'Missing audio token or empty audio mask'
        encoder.audit_generation=dict(audio_feature_shape=list(features.shape),valid_mel_frames=mask.sum(-1).tolist(),
            audio_token_count=count,audio_tower_calls=0,audio_merged_into_language_input=False)
        outputs=original_generate(*args,**kwargs)
        assert encoder.audit_generation['audio_tower_calls']>0 and encoder.audit_generation['audio_merged_into_language_input'],'Audio encoder or multimodal merge was bypassed'
        new_ids=outputs[0,kwargs['input_ids'].shape[1]:].tolist()
        eos=encoder.model.generation_config.eos_token_id
        if eos is None:eos=encoder.processor.tokenizer.eos_token_id
        eos=set(eos if isinstance(eos,(list,tuple)) else [eos])
        encoder.audit_generation.update(tokens=len(new_ids),last_token=new_ids[-1] if new_ids else None,
            max_new_tokens=kwargs['max_new_tokens'],
            truncated=len(new_ids)>=kwargs['max_new_tokens'] and (not new_ids or new_ids[-1] not in eos))
        return outputs
    encoder.model.generate=generate
    def normalize(self,text,max_time):
        self.audit_result=parse_reply(text,max_time,self.min_intensity_score,
            truncated=self.audit_generation.get('truncated',False))
        return self.audit_result['normalized_text'] or ''
    encoder.normalize_result=MethodType(normalize,encoder)
    if 'audios' in inspect.signature(encoder.processor.__call__).parameters:
        class ProcessorCompatibility:
            def __init__(self,original):self.original=original
            def __getattr__(self,name):return getattr(self.original,name)
            def __call__(self,**kwargs):
                if 'audio' in kwargs:kwargs['audios']=kwargs.pop('audio')
                return self.original(**kwargs)
        encoder.processor=ProcessorCompatibility(encoder.processor)
    encoder.audit_generation={};encoder.audit_result=None
    return encoder
