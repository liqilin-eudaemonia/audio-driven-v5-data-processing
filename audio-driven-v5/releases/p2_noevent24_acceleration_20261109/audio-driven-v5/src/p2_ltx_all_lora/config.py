"""New independent scope and output; no changes to historical training identity."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT, read, sha
from p2_ltx_events.config import validate as native_validate
from . import VERSION
from .lora import ATTENTION, FFN, WHISPER

FULL = ROOT / 'experiments/p2_ltx_clip_train32_after_gpu67_20261106/config.json'
MATCHED = ROOT / 'experiments/p2_current500h_av_whisper31_20261107/config.json'
DEFAULT_OUTPUT = ROOT / 'experiments/p2_ltx_all_attention_ffn_lora31_20261108'


def scope():
    return dict(attention=list(ATTENTION), ffn=list(FFN),
                optional_attention_gate=True, whisper_full=list(WHISPER),
                frozen_norms_and_nonblock_projections=True)


def expected(output):
    c = copy.deepcopy(read(FULL));matched = read(MATCHED)
    for key in ('training', 'duration_sampling', 'execution', 'sample_list'):
        c[key] = copy.deepcopy(matched[key])
    c['training']['gradient_checkpointing'] = 'all'
    c.update(output=str(Path(output).resolve()), ltx_all_lora_experiment=VERSION,
             ltx_lora_scope=scope(), ltx_all_lora_reference=str(FULL),
             ltx_all_lora_reference_sha256=sha(FULL),
             matched_reference=str(MATCHED), matched_reference_sha256=sha(MATCHED))
    c.pop('resume_execution', None)
    c.pop('checkpoint_schedule', None)
    return c


def validate(c, formal=True):
    native_validate(c, formal=formal)
    if c != expected(c['output']):
        raise ValueError('Expanded LoRA must preserve final500h, caption, losses and matched31 schedule')
    if Path(c['output']).resolve() in (Path(read(FULL)['output']).resolve(),
                                       Path(read(MATCHED)['output']).resolve()):
        raise ValueError('Use an independent expanded-LoRA output')
    return c


def load(path):
    return validate(read(path))


def source_identity():
    from p2_ltx_events.config import source_identity as original
    result = original()
    for p in [*Path(__file__).parent.glob('*.py'), ROOT/'scripts/p2_ltx_all_lora_train.py',
              ROOT/'scripts/p2_current500h_component_ablations.py']:
        result[str(p)] = sha(p)
    from p2_current_av_only import schedule
    result[str(Path(schedule.__file__))] = sha(schedule.__file__)
    return result

