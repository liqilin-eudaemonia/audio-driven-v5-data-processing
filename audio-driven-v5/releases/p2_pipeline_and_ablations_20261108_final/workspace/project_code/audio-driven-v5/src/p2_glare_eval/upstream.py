"""Load pinned upstream functions without leaking their generic `common` import."""
import importlib.util
import sys
from functools import lru_cache
from .common import ROOT, read, sha

REACTIONS = ('smiling', 'laughing', 'frowning', 'surprised', 'nodding', 'head_shaking')
GLARE = ROOT / 'related/glare/eval'
DETECTOR = ROOT / 'related/visual_reaction_annotation'
LOCK = ROOT / 'configs/p2_glare_upstream.lock.json'


def verify_sources():
    lock = read(LOCK)
    for path, expected in lock['files'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'GLARE source/model changed: {path}')
    return lock


def module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return result


@lru_cache(None)
def metrics():
    verify_sources()
    shared = module('_p2_glare_upstream_common', GLARE / 'common.py')
    previous = sys.modules.get('common')
    sys.modules['common'] = shared
    try:
        result = {name: module('_p2_glare_' + name, GLARE / (name + '.py'))
                  for name in ('r_f1', 'r_tiou', 'r_atd', 'r_fid')}
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    return dict(common=shared, **result)


def reaction_module(name):
    if name not in REACTIONS:
        raise ValueError(name)
    filename = 'run_nodding_intensity_mt_v2.py' if name == 'nodding' else f'run_{name}_intensity_mt.py'
    return module('_p2_glare_detector_' + name, DETECTOR / 'reaction_detector' / name / filename)
