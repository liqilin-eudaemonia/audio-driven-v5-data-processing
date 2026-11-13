"""Freeze current public source and measure the requested training pool on CPU."""
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
POOL = ROOT / 'experiments/listen_j_upright_av500h_train_dev_v3_20261105'
REPOS = ['lzk901372/glare', 'RobinWitch/DyStream', 'TaekyungKi/AvatarForcing',
         'liushenme/ListenFormer', 'Boese0601/Dyadic-Interaction-Modeling',
         'evonneng/learning2listen', 'dc3ea9f/vico_challenge_baseline']


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Codex-reproduction-audit'})
    return urllib.request.urlopen(request, timeout=40).read()


def upstream(repo):
    target = WORK / 'upstream' / repo.replace('/', '__')
    target.mkdir(parents=True, exist_ok=True)
    try:
        tree = json.loads(get('https://api.github.com/repos/' + repo + '/git/trees/HEAD?recursive=1'))
        (target / 'tree.json').write_text(json.dumps(tree, indent=2) + '\n')
        paths = [r['path'] for r in tree['tree'] if r['type'] == 'blob']
        readme = get(f'https://raw.githubusercontent.com/{repo}/{tree["sha"]}/README.md')
        (target / 'README.md').write_bytes(readme)
        return {'repo': repo, 'sha': tree['sha'], 'file_count': len(paths),
                'training_named_files': [p for p in paths if 'train' in p.lower()],
                'readme_sha256': hashlib.sha256(readme).hexdigest(),
                'tree_snapshot': str(target / 'tree.json')}
    except Exception as e:
        return {'repo': repo, 'error': str(e)}


def main():
    csv_path = POOL / 'selection/sample_list.csv'
    summary = json.loads((POOL / 'selection/SUMMARY.json').read_text())
    raw = csv_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary['files']['sample_list.csv']['sha256']
    splits = collections.Counter()
    ids = set()
    durations = []
    captions = collections.Counter()
    for row in csv.DictReader(raw.decode('utf-8-sig').splitlines()):
        assert row['sample_id'] not in ids
        ids.add(row['sample_id'])
        assert row['split'] == row['source_split'] and row['split'] in {'train', 'dev'}
        splits[row['split']] += 1
        durations.append(float(row['duration_s']))
        captions[row['caption_identity_version']] += 1
    assert len(ids) == summary['samples'] == 284218
    assert dict(splits) == summary['source_split_counts']
    assert abs(math.fsum(durations) / 3600 - summary['hours']) < 1e-8
    # Upstream reads full 60-frame chunks, then slices its requested window.
    windows = []
    for audio_s, refs in [(10, 125), (2, 125), (2, 25), (2, 1)]:
        requested_frames = int(audio_s * 25 + 10 + refs)
        full_chunks = math.ceil(requested_frames / 60)
        min_seconds = full_chunks * 60 / 25
        selected = [d for d in durations if math.floor(d * 25 + 1e-7) >= full_chunks * 60]
        windows.append({'audio_seconds': audio_s, 'reference_frames': refs, 'history_frames': 10,
                        'requested_frames': requested_frames, 'requested_seconds': requested_frames / 25,
                        'minimum_full_chunk_seconds': min_seconds, 'potential_eligible_segments': len(selected),
                        'potential_eligible_hours': math.fsum(selected) / 3600,
                        'assumption': '60-frame HDF5 chunks, no temporal padding; duration projection, not exported features',
                        'adopted_for_training': False})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        repos = list(executor.map(upstream, REPOS))
    report = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'pool': {'selected_manifest': str(POOL / 'selection/selected.jsonl'),
                       'selected_manifest_sha256_from_sealed_summary': summary['files']['selected.jsonl']['sha256'],
                       'csv': str(csv_path), 'csv_sha256_verified': hashlib.sha256(raw).hexdigest(),
                       'samples': len(ids), 'hours': math.fsum(durations) / 3600,
                       'source_splits': dict(splits), 'min_duration_s': min(durations),
                       'max_duration_s': max(durations), 'caption_identities': dict(captions),
                       'test_included': False,
                       'geometry': 'selected row source.video_transform plus upright normalization in current reader',
                       'caption': 'selected row caption with caption_file_sha256; do not rewrite into invented scores',
                       'ltx_cache_is_not_lia_or_emoca_cache': True},
              'glare_window_projection': windows, 'upstream': repos,
              'training_started': False, 'gpu_allocated': False,
              'requested_order': ['GLARE', 'DyStream and Avatar Forcing', 'L2L, DIM, ViCo, ListenFormer'],
              'blockers': {
                  'GLARE': ['LIA motion/audio/reaction/intensity HDF5 not exported for this selected pool',
                            'published 10s+125ref+10history window exceeds every selected segment',
                            'no project-local GLARE checkpoints directory; auxiliary model reuse needs validation',
                            'upstream random video split must not redefine the frozen benchmark test'],
                  'DyStream': ['current public repository contains inference but no training loop'],
                  'AvatarForcing': ['current public repository contains inference but no training loop'],
                  'L2L': ['existing stopped diagnostic trained ViCo397, not this 500h pool',
                          'need 500h motion/audio extraction and fixed paper-versus-code training protocol'],
                  'DIM': ['need paired EMOCA50+pose6 and HuBERT features for this 500h pool'],
                  'ViCo': ['need paired 3DMM features and train-only fitted normalization for this 500h pool'],
                  'ListenFormer': ['public training code exists; need paired features and local environment',
                                   'expired model link does not prevent training from scratch']}}
    (WORK / 'AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    pool_summary = {k: v for k, v in report['pool'].items() if k != 'caption_identities'}
    print(json.dumps({'pool': pool_summary, 'caption_identity_count': len(captions), 'windows': windows,
                      'repos': [{k: r[k] for k in ['repo', 'sha', 'error'] if k in r} for r in repos]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
