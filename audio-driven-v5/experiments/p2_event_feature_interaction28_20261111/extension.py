"""Two approved interaction candidates using the frozen training implementation."""
from pathlib import Path
import copy
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
ORIGINAL = ROOT / 'experiments/p2_loss_search28_20261110'
LEGACY = ROOT / 'experiments/p2_loss_search28_evaluation_20261110'
CONTROL = ROOT / 'experiments/p2_event4_dualseed_eval28_20261110'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(LEGACY)]
from p2_listen.contracts import read, sha

CANDIDATES = {'event020_feature010': (.20, .10), 'event100_feature010': (1., .10)}
SOURCE_FILES = ('extension.py', 'worker.py', 'evaluate28.py', 'driver.py',
                'rolling_driver.py', 'rolling_video.py', 'parallel_events.py')


def make(name):
    event, feature = CANDIDATES[name]
    config = copy.deepcopy(read(ORIGINAL / 'runs/baseline/config.json'))
    config['output'] = str(WORK / 'runs' / name)
    config['loss_search']['candidate'] = name
    config['loss_weights'] = dict(av=1., event=event, video_feature=feature, audio_feature=feature)
    return config


def validate(config):
    name = config.get('loss_search', {}).get('candidate')
    if name not in CANDIDATES or config != make(name):
        raise ValueError('Interaction candidate/config differs from the approved experiment')
    return config


def source_identity():
    original = read(ORIGINAL / 'PREFLIGHT_ACCEPTED.json')['source']
    for filename, expected in original.items():
        if sha(filename) != expected:
            raise ValueError('Frozen original training source changed: ' + filename)
    return {**original, **{str(WORK / n): sha(WORK / n) for n in SOURCE_FILES},
            str(WORK / 'protocol.json'): sha(WORK / 'protocol.json')}


def cluster_for(name):
    cluster = copy.deepcopy(read(ORIGINAL / 'runs/baseline/cluster.json'))
    cluster.update(master_port=30230 + list(CANDIDATES).index(name),
                   training_script=str(WORK / 'worker.py'),
                   training_config=str(WORK / 'runs' / name / 'config.json'))
    return cluster


def validate_cluster(config, cluster):
    validate(config)
    if cluster != cluster_for(config['loss_search']['candidate']):
        raise ValueError('Interaction allocation/network/worker changed')
    return cluster


def activate():
    # Each training/inference process installs the extension before importing
    # the native executor. Historical source files and checkpoints stay usable.
    import p2_loss_search28.config as config
    config.WORK = WORK
    config.CANDIDATES = CANDIDATES
    config.make = make
    config.validate = validate
    config.validate_cluster = validate_cluster
    config.source_identity = source_identity

