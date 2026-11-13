"""A separate experiment identity; the existing AV control stays immutable."""
import math
from pathlib import Path

from p2_listen.contracts import ROOT, read, sha
from p2_matrix16.config import validate as matrix_validate
from . import VERSION

DEFAULT = ROOT / 'configs/p2_j_qwen_whisper_av_20261104.json'


def validate(c, formal=True):
    matrix_validate(c, formal=formal)
    if (c.get('whisper_experiment'), c['id'], c['text_condition'], c['loss'], c['stage'],
            c.get('event_head_training'), c.get('qwen_hidden_source')) != (
            VERSION, 'guided_av_s1', 'guided', 'av', 1, False, 'base_pre_sft'):
        raise ValueError('Expected frozen base Qwen H + guided text + pure AV Whisper experiment')
    if c.get('teacher_cache') is not None or c['qwen_lora']['enabled']:
        raise ValueError('Whisper AV forbids Feature teachers and trainable Qwen')
    if Path(c['plan_cache']).resolve() != Path(c['j_cache']).resolve():
        raise ValueError('Keep the original pre-SFT guided H cache')
    if Path(c['output']).resolve() == Path(c['baseline_output']).resolve():
        raise ValueError('Whisper must use a new output directory')
    if Path(c['whisper_cache']).resolve() == Path(c['j_cache']).resolve():
        raise ValueError('Whisper must use its own cache directory')
    w = c['whisper']
    if set(w) != {'model_path', 'input_dim', 'context_dim', 'temporal_rope', 'enable_video',
                  'enable_audio', 'learning_rate', 'encoder_dtype'}:
        raise ValueError('Unexpected Whisper configuration fields')
    if (type(w['input_dim']) is not int or type(w['context_dim']) is not int
            or w['input_dim'] < 1 or w['context_dim'] < 1
            or any(type(w[k]) is not bool for k in ('temporal_rope', 'enable_video', 'enable_audio'))
            or not w['enable_video'] or not w['enable_audio'] or w['encoder_dtype'] != 'float32'
            or isinstance(w['learning_rate'], bool) or not math.isfinite(w['learning_rate'])
            or w['learning_rate'] <= 0):
        raise ValueError('Invalid frozen Whisper / dual residual branch settings')
    return c


def load(path=DEFAULT):
    return validate(read(path))


def validate_baseline(c, reference):
    """Reject unannounced data, initialization, objective, or schedule changes."""
    keys = ('id', 'text_condition', 'loss', 'stage', 'data_config', 'plan_cache', 'teacher_cache',
            'parent_experiment', 'loss_weights', 'qwen_lora', 'text_lora', 'j_cache', 'clip_labels',
            'event_supervision', 'connector', 'duration_sampling', 'event_head_training')
    if any(c.get(k) != reference.get(k) for k in keys):
        raise ValueError('Whisper arm differs from its AV baseline beyond the extra condition')
    original = reference['training']
    if (set(c['training']) != set(original) or any(c['training'][k] != v for k, v in original.items()
            if k not in ('save_every', 'gradient_checkpointing'))):
        raise ValueError('Whisper training data order / budget / optimizer differs from AV baseline')
    return c


def source_identity():
    from p2_j_duration_execution import install_backend
    result = dict(install_backend().source_identity())
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'scripts/p2_j_whisper_av.py']
    paths += sorted((ROOT / 'external_sources/p2_whisper_av').rglob('*.py'))
    paths += [ROOT / 'external_sources/p2_whisper_av/UPSTREAM.json']
    result.update({str(p): sha(p) for p in paths})
    return result
