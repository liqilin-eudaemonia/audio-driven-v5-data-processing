import math
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json
from . import VERSION

LOSSES=('av','av_event','av_feature','av_event_feature')

def validate_connector(value):
    keys={'fusion_width','fusion_depth','fusion_heads'}
    if (not isinstance(value,dict) or set(value)!=keys
            or any(type(value[k]) is not int or value[k]<1 for k in keys)):
        raise ValueError('Connector requires positive integer fusion_width/depth/heads')
    if value['fusion_width']%value['fusion_heads']:
        raise ValueError('Connector width must be divisible by attention heads')
    return value

def connector_config(c,cfg):
    """Override trainable fusion only; frozen extraction/layout config stays intact."""
    from dataclasses import replace
    if 'connector' not in c:return cfg
    return replace(cfg,**validate_connector(c['connector']))

def experiment_id(text,loss,stage):
    return f'{text}_{loss}_s{stage}'

def feature_enabled(c):return 'feature' in c['loss']
def event_enabled(c):return 'event' in c['loss']

def validate(c,formal=False):
    if 'connector' in c:validate_connector(c['connector'])
    if 'event_supervision' in c:
        from p2_clip_events.model import validate_spec
        validate_spec(c['event_supervision'])
    if c['schema']!=VERSION or c['text_condition'] not in ('fixed','guided') or c['loss'] not in LOSSES or c['stage'] not in (1,2):raise ValueError('Invalid experiment matrix member')
    if c['id']!=experiment_id(c['text_condition'],c['loss'],c['stage']):raise ValueError('Experiment ID mismatch')
    if c['qwen_lora']['enabled']!=(c['stage']==2):raise ValueError('Qwen LoRA stage mismatch')
    if c['loss_weights']!={'av':1.,'event':.05,'video_feature':.1,'audio_feature':.1}:raise ValueError('Paired loss weights differ')
    if c['qwen_lora']['targets']!=['q_proj','k_proj','v_proj','o_proj']:raise ValueError('Unsupported Qwen targets')
    for key in ('qwen_lora','text_lora'):
        if type(c[key]['rank']) is not int or c[key]['rank']<1 or any(not math.isfinite(c[key][k]) or c[key][k]<=0 for k in ('alpha','learning_rate')):raise ValueError('Invalid LoRA setting')
    t=c['training']
    if c['text_lora']['enabled'] is not True or type(t['global_batch']) is not int or t['global_batch']<1 or t['save_every']!=100:raise ValueError('Invalid common training setup')
    if type(t['warmup_steps']) is not int or t['warmup_steps']<1 or t['weight_decay']!=0:raise ValueError('Invalid optimizer schedule')
    for key in ('connector_lr','feature_lr','event_lr','clip_norm'):
        if not math.isfinite(t[key]) or t[key]<=0:raise ValueError('Invalid '+key)
    if t['gradient_checkpointing'] not in ('all','alternating','three_quarters','none'):raise ValueError('Invalid checkpointing')
    expected_parent=experiment_id(c['text_condition'],c['loss'],1) if c['stage']==2 else None
    if c['parent_experiment']!=expected_parent:raise ValueError('Invalid stage parent')
    if formal and (type(t['steps']) is not int or t['steps']<1):raise ValueError('Set an explicit training step budget; no inherited 1000-step default')
    return c

def load(path,formal=False):return validate(read(path),formal)

def data_config(c):
    from p2_listen.config import load
    d=load(c['data_config'])
    if d['data']['size']!=512 or not d['data'].get('audio_postprocess') or d['training']['feature_loss_enabled']:
        raise ValueError('Expected immutable 512px postprocessed data view (feature targets live separately)')
    return d

def source_identity():
    from p2_listen.config import source_identity as native,load as native_load,DEFAULT
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_matrix16.py']
    paths+=list((ROOT/'src/p2_plan_cache').glob('*.py'))+list((ROOT/'src/listen_audio_postprocess').glob('*.py'))
    paths+=list((ROOT/'src/p2_clip_events').glob('*.py'))
    base=native_load(DEFAULT)
    base['training']['text_lora']=dict(enabled=True,rank=8,alpha=8.,learning_rate=1e-5)
    paths+=map(Path,native(base))
    return {str(p):sha(p) for p in sorted(paths)}

def emit(folder):
    folder=Path(folder);rows=[]
    for text in ('fixed','guided'):
        for loss in LOSSES:
            for stage in (1,2):
                ident=experiment_id(text,loss,stage)
                c=dict(schema=VERSION,id=ident,text_condition=text,loss=loss,stage=stage,
                    data_config=str(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json'),
                    plan_cache=str(ROOT/'experiments/p2_plan_conditions_train50h_verbatim_20261027'),
                    teacher_cache=str(ROOT/'experiments/p2_matrix16_teacher50h_20261027'),
                    output=str(ROOT/'experiments/p2_matrix16_training_20261027'/ident),
                    parent_experiment=experiment_id(text,loss,1) if stage==2 else None,
                    loss_weights=dict(av=1.,event=.05,video_feature=.1,audio_feature=.1),
                    qwen_lora=dict(enabled=stage==2,rank=8,alpha=8,learning_rate=1e-5,targets=['q_proj','k_proj','v_proj','o_proj'],gradient_checkpointing=True),
                    text_lora=dict(enabled=True,rank=8,alpha=8,learning_rate=1e-5),
                    training=dict(seed=20261023,global_batch=24,steps=None,save_every=100,warmup_steps=50,
                        connector_lr=3e-5,feature_lr=3e-5,event_lr=1e-4,clip_norm=1.,weight_decay=0.,
                        gradient_checkpointing='all',tensorboard=True))
                validate(c);atomic_json(folder/(ident+'.json'),c,immutable=True)
                rows.append(dict(id=ident,text=text,loss=loss,stage=stage,parent=c['parent_experiment'],config=ident+'.json'))
    atomic_json(folder/'matrix.json',dict(schema=VERSION,experiments=rows,training_started=False),immutable=True)
    return rows
