#!/usr/bin/env python3
"""Release the user-selected weak-label subset without changing supervision or shared caches."""
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'experiments/p2_train50h_backchannel_face_20261026'
sys.path.insert(0, str(ROOT / 'src'))
from p2_listen.config import load, activate, source_identity
from p2_listen.contracts import read, sha, digest, atomic_json, EVENTS
from p2_listen.annotations import validate, validate_contract, class_statistics
from p2_listen.features import identity, load_text
from p2_listen.cache import load_part


def main():
    import torch
    torch.set_num_threads(1)
    if (OUT / 'READY.json').exists():
        raise FileExistsError('Training view already prepared; do not overwrite')
    complete = read(SOURCE / 'COMPLETE.json')
    for name, expected in complete['files'].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f'Source file checksum mismatch: {name}')
    c = read(ROOT / 'configs/p2_listen_av_event_text_lora_h200_v1.json')
    c['output'] = str(OUT)
    c['training'].update(global_batch=24, steps=1268)
    atomic_json(OUT / 'config.json', c, immutable=True)
    c = load(OUT / 'config.json')
    activate(c)
    release = dict(
        schema='p2-user-authorized-training-release-v1', source=str(SOURCE),
        source_complete_sha256=sha(SOURCE / 'COMPLETE.json'),
        authorization='User requested E1 training on this 50-hour subset, explicitly 3 machines / 24 GPUs, and selected 1268 steps at global batch 24.',
        arm='E1', seed=20261023, steps=1268, global_batch=24, world=24,
        samples=30415, sample_positions=30432, human_verified=False,
        supervision_policy='Preserve all original events, descriptions, timestamps, coverage, unknown intervals and weak-background assumptions. No new negatives or human verification.',
        cache_policy='Reuse input/VAE/text. Events have new provenance-bound identities and byte-identical immutable payload links. No teacher cache.',
        source_identity=source_identity(c), config_sha256=sha(OUT / 'config.json'))
    atomic_json(OUT / 'training_release.json', release, immutable=True)
    for name in ('observations.json', 'targets.json'):
        dest = OUT / 'manifests' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(SOURCE / 'manifests' / name)
    (OUT / 'cache').mkdir(exist_ok=True)
    for part in ('input', 'vae', 'text'):
        dest = OUT / 'cache' / part
        if not dest.exists():
            dest.symlink_to((SOURCE / 'cache' / part).resolve())
    observations = read(SOURCE / 'manifests/observations.json')
    targets = {r['sample_id']: r for r in read(SOURCE / 'manifests/targets.json')}
    rows = [json.loads(line) for line in (SOURCE / 'dataset.jsonl').open()]
    ids = (SOURCE / 'sample_ids.txt').read_text().splitlines()
    assert len(ids) == len(set(ids)) == 30415
    assert ids == [r['sample_id'] for r in rows] == [r['sample_id'] for r in observations]
    assert set(targets) == set(ids)
    load_text(c)
    reference = str(OUT / 'training_release.json')
    release_sha = sha(reference)

    def prepare(item):
        row, obs = item
        sid = row['sample_id']
        assert obs['experiment_split'] == obs['official_split'] == 'train'
        label_path = SOURCE / 'labels/canonical' / (sid + '.json')
        original_sha = sha(label_path)
        assert original_sha == row['canonical_label_sha256'], sid
        original = read(label_path)
        validate(original, obs)
        validate_contract(original, c)
        p = original['provenance']
        assert original['training_admitted'] is True, sid
        assert p['debug_only'] is False and p['formal_training_admitted'] is False, sid
        assert p['feature_preparation_only'] is True and p['human_verified'] is False, sid
        assert p['raw_completion_status'] == 'complete_candidates', sid
        assert p['media_hashes'] == targets[sid]['media_hashes'], sid
        # Match every reused per-sample cache contract against the current production code.
        for part in ('input', 'vae', 'events'):
            prior = read(Path(row['cache'][part]['path']) / 'complete.json')
            expected = identity(c, part, obs, targets[sid] if part == 'vae' else None,
                                label=original if part == 'events' else None)
            assert prior['identity'] == digest(expected) == row['cache'][part]['identity'], (sid, part)
            assert prior['contract'] == expected, (sid, part)
            assert prior['payload_sha256'] == row['cache'][part]['payload_sha256'], (sid, part)
            assert (Path(row['cache'][part]['path']) / 'payload.pt').is_file(), (sid, part)
        label = copy.deepcopy(original)
        label['provenance'].update(formal_training_admitted=True, feature_preparation_only=False,
            training_release=dict(reference=reference, sha256=release_sha,
                                  original_label_sha256=original_sha, human_verified=False))
        # rasterize() reads only these unchanged top-level fields, never provenance.
        assert {k: v for k, v in label.items() if k != 'provenance'} == {
            k: v for k, v in original.items() if k != 'provenance'}
        validate(label, obs)
        validate_contract(label, c)
        atomic_json(OUT / 'labels/canonical' / (sid + '.json'), label, immutable=True)
        old = Path(row['cache']['events']['path'])
        prior = read(old / 'complete.json')
        # Validate bytes and deserialize using the unchanged old production identity.
        payload = load_part(old, identity(c, 'events', obs, label=original))
        dest = OUT / 'cache/events' / sid
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'payload.pt').exists():
            (dest / 'payload.pt').symlink_to(old / 'payload.pt')
        new_identity = identity(c, 'events', obs, label=label)
        atomic_json(dest / 'complete.json', dict(identity=digest(new_identity), contract=new_identity,
                    payload_sha256=prior['payload_sha256']), immutable=True)
        stats = class_statistics([payload], c['events']['pos_weight_cap'], c['events']['K_train'])
        return dict(sample_id=sid, original_label_sha256=original_sha,
                    training_label_sha256=sha(OUT / 'labels/canonical' / (sid + '.json')),
                    events_payload_sha256=prior['payload_sha256'], events_identity=digest(new_identity),
                    supervision_unchanged=True), stats

    begin = time.monotonic()
    totals = {k: dict(positive=0., count=0) for k in EVENTS}
    with ThreadPoolExecutor(max_workers=16) as pool, (OUT / 'release_samples.jsonl').open('w') as f:
        for n, (record, stats) in enumerate(pool.map(prepare, zip(rows, observations)), 1):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            for k in EVENTS:
                totals[k]['positive'] += stats[k]['positive']
                totals[k]['count'] += stats[k]['count']
            if n % 500 == 0 or n == len(rows):
                progress = dict(prepared=n, total=len(rows), elapsed_seconds=time.monotonic()-begin)
                atomic_json(OUT / 'preparation_progress.json', progress)
                print(json.dumps(progress), flush=True)
    for k, stat in totals.items():
        stat['negative'] = stat['count'] - stat['positive']
        stat['enabled'] = 0 < stat['positive'] < stat['count']
        stat['pos_weight'] = min(stat['negative'] / max(stat['positive'], 1.), c['events']['pos_weight_cap'])
        assert stat['enabled'], (k, stat)
    atomic_json(OUT / 'READY.json', dict(samples=len(rows), config_sha256=sha(OUT / 'config.json'),
        release_sha256=release_sha, release_samples_sha256=sha(OUT / 'release_samples.jsonl'),
        statistics=totals, full_input_vae_payload_hash_check='Production trainer checks every payload before model allocation on each rank',
        event_payloads_sha256_verified=True, labels_and_event_supervision_unchanged=True), immutable=True)
    print('READY', flush=True)


if __name__ == '__main__':
    main()
