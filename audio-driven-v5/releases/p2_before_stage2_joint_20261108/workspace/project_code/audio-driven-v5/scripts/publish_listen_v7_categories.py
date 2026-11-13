"""Extend the existing V7 preview with all six audio-QC/pool categories."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
import time

import publish_listen_v7_sample as base

OUT = base.OUT
SEED = 202610221210
CATEGORIES = {
    'keep_primary': ('自动通过 · 主池', 24),
    'keep_onset_reserve': ('自动通过 · 开头反馈备用池', 12),
    'review_primary': ('待复核 · 主池', 16),
    'review_onset_reserve': ('待复核 · 开头反馈备用池', 12),
    'exclude_primary': ('建议排除 · 主池', 8),
    'exclude_onset_reserve': ('建议排除 · 开头反馈备用池', 8),
}


def select():
    archive = OUT / 'primary_samples_original.json'
    if not archive.exists():
        old = json.loads((OUT / 'samples.json').read_text())
        assert len(old['samples']) == 24
        base.write(archive, old)
    original = json.loads(archive.read_text())['samples']
    rows = [dict(r, review_category='keep_primary') for r in original]
    used_groups = {r['group_id'] for r in rows}
    used_pairs = {r['pair'] for r in rows}
    rng = random.Random(SEED)
    index = json.loads((base.CORPUS / 'manifests/index.json').read_text())
    stats = {}
    # Select the smallest excluded pool first, before common groups are consumed.
    for key in ['exclude_onset_reserve', 'exclude_primary', 'keep_onset_reserve',
                'review_onset_reserve', 'review_primary']:
        path = base.CORPUS / 'manifests' / f'train_{key}.jsonl'
        pools = {(short, voice): [] for short in (True, False) for voice in (True, False)}
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                pools[r['duration_s'] < 6, bool(r['listener_human_evidence'])].append(r)
        for pool in pools.values():
            rng.shuffle(pool)
        selected = []
        while len(selected) < CATEGORIES[key][1]:
            added = False
            for pool in pools.values():
                while pool:
                    r = pool.pop()
                    if r['group_id'] in used_groups or r['pair'] in used_pairs:
                        continue
                    selected.append(dict(r, review_category=key,
                        preview=f"media/{r['sample_id']}/"))
                    used_groups.add(r['group_id']); used_pairs.add(r['pair'])
                    added = True
                    break
                if len(selected) == CATEGORIES[key][1]:
                    break
            if not added:
                raise RuntimeError('Insufficient distinct groups: ' + key)
        rows.extend(selected)
        stats[key] = dict(manifest=str(path), sha256=base.sha(path))
    rows.sort(key=lambda r: list(CATEGORIES).index(r['review_category']))
    for key, (label, _) in CATEGORIES.items():
        stats.setdefault(key, {})
        stats[key].update(label=label, population_count=index['counts']['train_' + key],
            population_hours=index['unique_hours']['train_' + key],
            sample_count=sum(r['review_category'] == key for r in rows))
    stats['keep_primary'].update(manifest=str(base.MANIFEST),sha256=base.sha(base.MANIFEST))
    base.write(OUT / 'category_selection.json', dict(seed=SEED, samples=rows, categories=stats,
        note='Original 24 primary controls retained. Additional clips sampled by duration (<6/>=6) and model human evidence, distinct groups; sparse cells redistribute. No manual cherry-picking.'))
    return rows, stats


def evidence(row):
    result = json.loads(Path(row['quality_evidence_file']).read_text())
    detail = next(c for c in result['clips'] if c['sample_id'] == row['sample_id'])
    assert detail['effective_decision'] == row['effective_decision']
    tracks = {}
    for role in ('speaker', 'listener'):
        q = detail['qc3'][role]
        flags = []
        for kind in ('events', 'background'):
            for e in q[kind]:
                if e['decision']['status'] == 'keep':
                    continue
                flags.append(dict(kind=kind, start=e['start'], end=e['end'],
                    decision=e['decision'], panns=e.get('panns'), paired=e.get('paired')))
        tracks[role] = dict(status=q['status'], reasons=q['reasons'], flagged_intervals=flags)
    return dict(tracks=tracks,
        deferred_reasons=detail['listener_screen'].get('deferred_reasons', []),
        human_decisions=detail.get('human_decisions'),
        boundary_activity=detail['boundary_activity'])


def publish_one(row):
    marker = OUT / row['preview'] / 'export.json'
    cached = json.loads(marker.read_text()) if marker.exists() else None
    if cached:
        for k in ('start', 'end', 'sources', 'listener_crop', 'normalized_audio'):
            assert cached[k] == row[k], 'Preview/source mismatch: ' + k
        for mode, check in cached['media_checks'].items():
            assert base.sha(OUT / row['preview'] / (mode + '.mp4')) == check['sha256']
        rendered = dict(row, media_verified=True, media_checks=cached['media_checks'],
                        audio_checks=cached['audio_checks'])
    else:
        rendered = base.export(row)
    return dict(rendered, category_label=CATEGORIES[row['review_category']][0],
                review_evidence=evidence(row))


def main():
    rows, stats = select()
    with ThreadPoolExecutor(max_workers=12) as pool:
        samples = list(pool.map(publish_one, rows))
    assert len(samples) == 80 and len({r['group_id'] for r in samples}) == 80
    payload = dict(seed=SEED, samples=samples, categories=stats,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), complete=True,
        review_version='v7-six-categories',
        scope='Train clips reaching audio QC: keep/review/exclude × primary/onset_reserve. Earlier geometry/orientation rejection and processing errors are outside this preview. No production decisions changed.')
    # Publish assets only after every new sample passed export/decode validation.
    template = (base.ROOT / 'scripts/listen_v7_categories.html').read_text()
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font = TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', fontNumber=2)
    sub = subset.Subsetter(); sub.populate(text=template); sub.subset(font)
    font.flavor = 'woff'; font.save(OUT / 'review-categories.woff')
    base.write(OUT / 'samples_categories.json', payload)
    # Keep original samples.json + historical page so earlier review stays reproducible.
    previous = OUT / 'primary_history.html'
    if not previous.exists():
        shutil.copyfile(OUT / 'index.html', previous)
    temp = OUT / 'index.html.pending'; temp.write_text(template); temp.replace(OUT / 'index.html')
    print(json.dumps(dict(published='http://127.0.0.1:8875/listen_v7/',
        counts=dict(Counter(r['review_category'] for r in samples)), groups=80)), flush=True)


if __name__ == '__main__':
    main()
