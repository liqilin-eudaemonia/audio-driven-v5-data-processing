"""Pinned ViCo inference inventory and CPU-only cache validation."""
from pathlib import Path
import subprocess
import os
from vico_more_common import ROOT, read, save, sha

BASE = ROOT / 'experiments/p2_compare_vico43_20261106'
NAMES = ['conan', 'fallon', 'stephen', 'trevor']
# Availability from the existing local audit, not a fresh online release claim.
ALIASES = {'rlhg': 'vico', 'pch': 'mm2022', 'l2l': 'learning2listen', 'lm_listener': 'lmlistener'}
UNAVAILABLE = {
    'elp': 'No locally runnable generator; reviewed paper/search did not locate usable official assets',
    'mfrnet': 'No locally runnable generator; reviewed paper/search did not locate usable official assets',
    'customlistener': 'Official project Code link points to paper; no usable generator obtained',
    'arig': 'Located official project-page repository only; no usable generator obtained',
    'echo': 'Paper located; no usable official generator obtained in this check',
    'latent_behavior_diffusion': 'Paper located; no usable official generator obtained in this check',
    'glare': 'Formal training not completed; separate cache/training workflow is owned by the existing task',
    'lmlistener': 'Official native pretrained model linked but not integrated locally; scorer optimized; our RGB model evaluation still needs verified media mapping',
    'dim': 'No usable core generator checkpoint in the audited local assets',
    'vico': 'ViCo/RLHG are the same baseline; no completed native checkpoint, only local training diagnostics',
    'mm2022': 'Core generator checkpoint unavailable',
    'reactface': 'Core checkpoint and renderer resources unavailable',
    'ditailistener': 'Runnable official generator assets unavailable',
    'difflistener': 'CVPR2025 official runnable generator unavailable',
    'infp': 'Official runnable generator unavailable',
    'omniresponse': 'Complete generator code/weights unavailable',
    'speakervid': 'Repository has curation/metrics, no runnable generator',
}


def node8_guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7', 'node8 physical GPU7 only'
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', '7',
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert uuid.strip() == 'GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc'
    assert float(free) >= 32 * 1024, 'Insufficient free GPU memory; preserve other jobs'


def assets_match(protocol):
    assets = dict(protocol.get('weights', {}))
    for model in protocol.get('models', {}).values():
        assets.update(model['files'])
    assets.update(protocol.get('code_sha256', {}))
    return bool(assets) and all(Path(p).is_file() and sha(p) == digest for p, digest in assets.items())


def dystream_records(work, protocol):
    records = []
    for item in protocol['samples']:
        uid = item['id']
        receipt = work / 'generated' / uid / 'complete.json'
        video = work / 'generated' / uid / 'generated.mp4'
        score = work / 'scoring/dystream' / (uid + '.listener.mp4')
        if not receipt.is_file() or not video.is_file() or not score.is_file():
            continue
        r = read(receipt)
        if r['id'] != uid or r['protocol_sha256'] != sha(work / 'protocol.json'):
            continue
        if sha(video) != r['native_sha256'] or sha(score) != r['score_sha256']:
            continue
        records.append(dict(id=uid, path=str(score), sha256=r['score_sha256'], receipt=str(receipt)))
    return records


def l2l_group(work, name, protocol):
    marker = work / 'generated' / name / 'COMPLETE.json'
    if not marker.is_file():return None
    complete=read(marker)
    if complete.get('model') != name or complete.get('samples') != len(protocol['samples']):return None
    records = []
    for sample in protocol['samples']:
        path = work / 'generated' / name / (sample['id'] + '.npz')
        receipt = path.with_suffix('.json')
        if not path.is_file() or not receipt.is_file():
            return None
        r = read(receipt)
        if r['id'] != sample['id'] or r['model'] != name or sha(path) != r['feature_sha256']:
            return None
        records.append(dict(id=sample['id'], model=name, path=str(path), sha256=r['feature_sha256']))
    return records


def reuse(method, out):
    archive = BASE / method
    expected = read(BASE / 'dystream/protocol.json')['samples']
    for directory in dict.fromkeys([out, archive]):
        if not (directory / 'protocol.json').exists():
            continue
        protocol = read(directory / 'protocol.json')
        if [s['id'] for s in protocol['samples']] != [s['id'] for s in expected]:
            continue
        if not assets_match(protocol):
            continue
        if method == 'dystream':
            records = dystream_records(directory, protocol)
            if len(records) == len(expected):
                return records, directory / 'protocol.json'
        else:
            if set(protocol.get('models', {})) != set(NAMES):continue
            groups = [l2l_group(directory, n, protocol) for n in NAMES]
            if all(g is not None for g in groups):
                return [r for group in groups for r in group], directory / 'protocol.json'
    return None, None
