"""An independent experiment; existing Whisper checkpoints keep their identity."""
import copy
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
from p2_matrix16.config import validate as matrix_validate
from p2_whisper_av.config import validate as whisper_validate
from p2_whisper_av import VERSION as WHISPER_VERSION
from . import VERSION

DEFAULT=ROOT/'configs/p2_j_qwen_whisper_event_feature_20261105.json'


def pure_av_view(c):
    """Use the unchanged Whisper constructor, then attach the audited auxiliaries."""
    base=copy.deepcopy(c)
    base.update(id='guided_av_s1',loss='av',teacher_cache=None,event_head_training=False,
                whisper_experiment=WHISPER_VERSION)
    return base


def validate(c,formal=True):
    matrix_validate(c,formal=formal)
    if (c.get('whisper_aux_experiment'),c['id'],c['loss'],c['text_condition'],c['stage'],
            c.get('event_head_training'),c.get('qwen_hidden_source')) != (
            VERSION,'guided_av_event_feature_s1','av_event_feature','guided',1,True,'base_pre_sft'):
        raise ValueError('Expected base frozen H + AV + Event + Feature + speaker Whisper')
    if not c.get('teacher_cache') or c['qwen_lora']['enabled'] or c['parent_experiment'] is not None:
        raise ValueError('Independent stage1 requires audited teacher and frozen Qwen')
    if c['loss_weights'] != dict(av=1.,event=.05,video_feature=.025,audio_feature=.025):
        raise ValueError('Keep AV1 + Event.05 + video/audio Feature.025 each')
    if 'event_supervision' not in c or c['event_supervision']['mode']!='clip_multilabel':
        raise ValueError('Reuse existing clip-level Event labels and class mask')
    if 'whisper_execution_migration' in c:
        raise ValueError('A fresh combined experiment cannot resume an old pure-AV migration')
    if type(c['training']['save_every']) is not int or c['training']['save_every']<1:
        raise ValueError('Checkpoint interval must be a positive integer')
    whisper_validate(pure_av_view(c),formal=formal)
    from .execution import settings
    settings(c)
    return c


def load(path=DEFAULT):return validate(read(path))


def validate_baseline(c,baseline):
    """Freeze conditioning/order/Whisper/init; add auxiliary supervision only."""
    validate(c);whisper_validate(baseline)
    same=['data_config','plan_cache','j_cache','clip_labels','qwen_lora','text_lora',
          'event_supervision','connector','duration_sampling','qwen_hidden_source',
          'whisper','whisper_cache','whisper_training_execution']
    if any(c.get(k)!=baseline.get(k) for k in same):
        raise ValueError('Combined arm changed baseline conditions, order, architecture or execution')
    if c['training']!=baseline['training']:
        raise ValueError('Combined arm changed baseline training budget/order/optimizer')
    if Path(c['output']).resolve()==Path(baseline['output']).resolve():
        raise ValueError('Never write combined training into the old Whisper output')
    return c


def source_identity():
    from p2_whisper_av.config import source_identity as original
    result=original()
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_j_whisper_event_feature.py']
    result.update({str(p):sha(p) for p in paths})
    return result
