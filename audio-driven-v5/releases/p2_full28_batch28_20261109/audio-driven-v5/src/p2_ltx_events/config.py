"""An independent identity; old checkpoints cannot silently acquire a new loss."""
import math
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
from p2_whisper_event_feature.config import validate as original_validate
from . import VERSION

DEFAULT=ROOT/'configs/p2_ltx_hidden_clip_events_current500h_20261106.json'
TEMPORAL_CLASSES={'video':['head_nod','head_shake','smile','laugh','frown','surprise'],
                  'audio':['listener_vocalization']}
CURRENT_TEMPORAL_CLASSES={'video':TEMPORAL_CLASSES['video']+['brow_lowering','brows_raised'],
                          'audio':TEMPORAL_CLASSES['audio']}


def temporal_enabled(c):
    return c['hidden_event_supervision']['temporal_weight']>0


def validate(c,formal=True):
    original_validate(c,formal=formal)
    s=c.get('hidden_event_supervision',{})
    if c.get('hidden_event_experiment')!=VERSION or s.get('readout')!='last_block_before_output':
        raise ValueError('Expected explicit LTX hidden readout')
    for key in ('clip_weight','temporal_weight'):
        if isinstance(s.get(key),bool) or not isinstance(s.get(key),(float,int)) or not math.isfinite(s[key]) or s[key]<0:
            raise ValueError('Invalid hidden Event '+key)
    if s['clip_weight']+s['temporal_weight']<=0:
        raise ValueError('Invalid hidden Event weights')
    if temporal_enabled(c):
        bg=s.get('background_weight')
        if (s.get('classes') not in (TEMPORAL_CLASSES,CURRENT_TEMPORAL_CLASSES)
                or s.get('label_policy')!='historical-weak-intervals-v1'
                or not c.get('temporal_labels') or isinstance(bg,bool)
                or not isinstance(bg,(int,float)) or not math.isfinite(bg) or not 0<=bg<=1):
            raise ValueError('Frozen temporal labels, taxonomy and background policy required')
    if c.get('clip_numeric_labels') and not c.get('current_cache_index'):
        raise ValueError('Numeric clip labels require the current-cache dataset')
    if not c.get('hidden_event_parent_config'):
        raise ValueError('Parent configuration required')
    if Path(c['output']).resolve()==Path(read(c['hidden_event_parent_config'])['output']).resolve():
        raise ValueError('Hidden Event training requires a new output directory')
    from .resume31 import validate_checkpoint_schedule,validate_resume
    validate_checkpoint_schedule(c);validate_resume(c)
    return c


def validate_baseline(c,parent):
    validate(c)
    if c.get('resume_execution'):
        # The immediate, fully accepted batch32 parent is frozen field by field.
        from .resume31 import validate_resume
        validate_resume(c)
        return c
    added={'hidden_event_experiment','hidden_event_supervision','temporal_labels','hidden_event_parent_config','output'}
    if c.get('current_cache_index'):
        added|={'clip_numeric_labels','current_cache_index','current_source','j_cache','plan_cache','clip_labels','teacher_cache','whisper_cache','duration_sampling','training'}
        if {k:v for k,v in c['training'].items() if k not in ('global_batch','steps','save_every')}!={k:v for k,v in parent['training'].items() if k not in ('global_batch','steps','save_every')}:
            raise ValueError('Current data arm changed optimizer or training precision policy')
    if {k:v for k,v in c.items() if k not in added}!={k:v for k,v in parent.items() if k not in added}:
        raise ValueError('Hidden Event arm changed parent data, conditioning, optimizer or schedule')
    return c


def load(path=DEFAULT):return validate(read(path))


def source_identity():
    from p2_whisper_event_feature.config import source_identity as original
    result=original()
    for p in sorted(Path(__file__).parent.glob('*.py')):result[str(p)]=sha(p)
    p=ROOT/'scripts/p2_ltx_events.py';result[str(p)]=sha(p)
    for name in ('qwen_hidden','gemma_cache','av_cache','whisper_cache','teacher_cache'):
        p=ROOT/('scripts/p2_j_current_'+name+'.py');result[str(p)]=sha(p)
    for name in ('scripts/check_glare_headshake_direction.py','scripts/check_glare_expression_gate.py','src/listen_benchmark/caption_guard.py'):
        p=ROOT/name;result[str(p)]=sha(p)
    for name in ('scripts/p2_ltx_resume31.py','scripts/p2_ltx_resume24.py','scripts/p2_whisper_event_feature_train31.py'):
        p=ROOT/name;result[str(p)]=sha(p)
    return result
