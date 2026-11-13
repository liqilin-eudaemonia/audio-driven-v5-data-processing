"""Full-recipe continuation; reuse the validated native batch28 training loop."""
from pathlib import Path
import copy, importlib.util, math, sys

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_listen.contracts import read, sha

PARENT = W.parent/'p2_ltx_all_attention_ffn_lora31_20261108'
OUT = W/'train'
LOOP = W.parent/'p2_full28_batch28_20261109/training.py'
CANARY = W.parent/'p2_full28_batch28_20261109/canary.py'
START, TARGET, WORLD = 5500, 7000, 28


def normalized(c):
    base = read(PARENT/'config.json')
    base['output'] = c['output']
    return base


def validate(c):
    from p2_ltx_all_lora.config import validate as native
    base = normalized(c)
    native(base)
    expected = copy.deepcopy(base)
    expected['training'].update(global_batch=28, save_every=100,
                               steps=read(W/'PLAN.json')['new_end_step'])
    expected['batch28_continuation'] = read(W/'PLAN.json')
    assert c == expected and Path(c['output']) == OUT
    p = c['batch28_continuation']
    assert p['start_step'] == START and p['prefix_pairs'] == START*31
    assert p['new_end_step'] == START+math.ceil((p['total_pairs']-START*31)/28)
    assert sha(PARENT/'config.json') == p['parent_config_sha256']
    return c


def source():
    from p2_ltx_all_lora.config import source_identity
    result = source_identity()
    for p in [LOOP, CANARY, W/'run_runtime.py', W/'worker.py', W/'PLAN.json']:
        result[str(p)] = sha(p)
    return result


class Schedule:
    def __init__(self, c):
        from p2_ltx_all_lora.schedule import Schedule as Native
        validate(c)
        base = Native(normalized(c))
        self.pairs, self.size, self.seed = base.pairs, base.size, base.seed
        self.batch, self.plan = 28, c['batch28_continuation']
        assert len(self.pairs) == self.plan['total_pairs']

    def bounds(self, step):
        if not START < step <= self.plan['new_end_step']:
            raise ValueError('Outside continuation')
        begin = self.plan['prefix_pairs']+(step-START-1)*28
        return begin, min(begin+28, len(self.pairs))

    def count(self, step):
        a, b = self.bounds(step)
        return b-a

    def indices(self, size, step, batch, seed, rank=0, world=1):
        assert (size, batch, seed) == (self.size, 28, self.seed)
        assert world in (1, 28) and 0 <= rank < world
        a, b = self.bounds(step)
        return self.pairs[a:b][rank::world]


def load():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    spec = importlib.util.spec_from_file_location('p2_ltx_all_lora._resume5500_batch28', LOOP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.validate, m.source_identity, m.Schedule = validate, source, Schedule
    from p2_ltx_all_lora.model import Model
    from p2_ltx_all_lora.data import Dataset, data_identity
    def model(c, cfg, renderer):
        value = Model(normalized(c), cfg, renderer)
        value.spec = c
        return value
    m.QwenWhisperEventFeature = model
    m.Dataset = lambda c: Dataset(normalized(c))
    m.data_identity = lambda c: data_identity(normalized(c))
    spec = importlib.util.spec_from_file_location('full5500_canary', CANARY)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    m.split_canary = lambda **kw: check.longest_canary(training=m, arm='full', **kw)
    return m
