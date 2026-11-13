import copy
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
from p2_matrix16.config import validate as matrix_validate,feature_enabled,event_enabled
from p2_whisper_av.config import validate as whisper_validate
from p2_whisper_event_feature.config import pure_av_view
from . import VERSION

REFERENCE=ROOT/'experiments/p2_j_whisper_event_feature_train31_20261105/config.json'


def validate(c,formal=True):
    matrix_validate(c,formal=formal)
    if (c.get('whisper_ablation_experiment')!=VERSION or c['loss'] not in ('av_event','av_feature')
        or c['text_condition']!='guided' or c['stage']!=1 or c.get('qwen_hidden_source')!='base_pre_sft'
        or c.get('event_head_training')!=event_enabled(c) or bool(c.get('teacher_cache'))!=feature_enabled(c)):
        raise ValueError('Expected one auxiliary loss plus frozen base H and speaker Whisper')
    if c['loss_weights']!=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025):raise ValueError('Loss coefficients differ')
    if c['training']['global_batch']!=31 or c['training']['save_every']!=500:raise ValueError('Expected batch31 and save500')
    if c['training']['steps']!=8000:raise ValueError('Single-auxiliary ablations stop at8000')
    if c['parent_experiment'] is not None or c['qwen_lora']['enabled']:raise ValueError('Fresh frozen-H stage1 only')
    whisper_validate(pure_av_view(c),formal=formal)
    from p2_whisper_event_feature.execution import settings
    settings(c)
    return c


def validate_baseline(c,reference):
    validate(c)
    meta=c['ablation_reference']
    if sha(meta['config'])!=meta['sha256'] or read(meta['config'])!=reference:raise ValueError('Frozen comparison config changed')
    expected=copy.deepcopy(reference)
    for key in ['whisper_aux_experiment','world31_execution']:expected.pop(key,None)
    expected.update(id='guided_'+c['loss']+'_s1',loss=c['loss'],output=c['output'],
        teacher_cache=reference['teacher_cache'] if feature_enabled(c) else None,
        event_head_training=event_enabled(c),baseline_config=str(REFERENCE),
        whisper_ablation_experiment=VERSION,ablation_reference=meta)
    expected['training']['steps']=8000
    if c!=expected or Path(c['output']).resolve()==Path(reference['output']).resolve():raise ValueError('Unapproved comparison change')
    return c


def load(path):return validate(read(path))


def source_identity():
    saved=read(REFERENCE.parent/'run.json')['source']
    for name,h in saved.items():
        if sha(name)!=h:raise ValueError('Historical dependency changed: '+name)
    result=dict(saved)
    for p in [*Path(__file__).parent.glob('*.py'),ROOT/'scripts/p2_whisper_ablation.py']:
        result[str(p)]=sha(p)
    return result


class Schedule:
    """Exact prefix of the old full-epoch plan, including the last duration block."""
    def __init__(self,c):
        from p2_j_duration_execution import Schedule as Native
        reference=read(c['ablation_reference']['config']);validate_baseline(c,reference)
        self.native=Native(reference);self.steps=c['training']['steps']
        self.pairs=self.native.pairs[:self.steps*c['training']['global_batch']]
    def indices(self,size,step,batch,seed,rank=0,world=1):
        if not 1<=step<=self.steps:raise ValueError('Step outside ablation budget')
        return self.native.indices(size,step,batch,seed,rank,world)
