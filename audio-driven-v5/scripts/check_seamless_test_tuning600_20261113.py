"""CPU-only acceptance of Seamless600 through the existing evaluation readers."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'OmniResponse'))

from evaluation.seamless.bind import load_references, match
from evaluation.seamless.reader import (
    load_conditioning, load_listener_audio, load_record, resolve,
)
from select_seamless_test_tuning256_20261113 import rows, read, save, sha

WORK = ROOT / 'experiments/seamless_test_tuning600_20261113'
SOURCE_ISSUES = ROOT / 'experiments/glare_final_seamless_test6624_20261112/source_issues/042.json'


def check(media=True):
    prepared = WORK / 'prepared_600'
    refs = load_references(prepared)
    assert len(refs) == 600
    selected = rows(WORK / 'selection.jsonl')
    ids = [r['id'] for r in selected]
    assert [r['id'] for r in refs] == ids
    old = rows(ROOT / 'experiments/seamless_test_tuning256_20261113/selection.jsonl')
    assert ids[:256] == [r['id'] for r in old]
    complete = read(WORK / 'COMPLETE.json')
    for name, digest in complete['files_sha256'].items():
        assert sha(WORK / name) == digest, f'Frozen file changed: {name}'
    inputs = rows(prepared / 'inference_inputs.jsonl')
    annotations = rows(prepared / 'annotations.jsonl')
    template = rows(prepared / 'generation_map.template.jsonl')
    assert all([r['id'] for r in items] == ids for items in (inputs, annotations, template))
    entry = read(ROOT / 'evaluation/SEAMLESS_TUNING600.json')
    assert entry['prepared'] == str(prepared) and entry['samples'] == 600
    assert entry['selection_sha256'] == sha(WORK / 'selection.jsonl')
    assert entry['references_sha256'] == sha(prepared / 'references.jsonl')
    source_checks = []
    media_candidates = {}
    for rank, (ref, selection, inp) in enumerate(zip(refs, selected, inputs)):
        source, _ = load_record(ref['gt_reference'])
        _, resolved = resolve(ref['gt_reference'])
        src = resolved['source']
        assert source['sample_id'] == ref['id'] and source['split'] == 'test'
        assert src['listener_id'] == ref['listener_id'] and src['speaker_id'] == ref['speaker_id']
        assert src['source_start_s'] == ref['source_start'] and src['source_end_s'] == ref['source_end']
        assert (ref['width'], ref['height'], ref['fps']) == (512, 512, 25)
        assert inp['gt_reference'] == ref['gt_reference']
        assert Path(src['source']['archive']).is_file()
        assert Path(resolved['speaker_audio']['path']).is_file()
        assert sha(source['target']['source_file']) == source['target']['source_sha256']
        kind = ('retained' if rank < 256 else 'added', selection['listener_waveform_nonzero'])
        media_candidates.setdefault(kind, ref)
        if rank >= 256 and ref['duration_s'] < media_candidates[kind]['duration_s']:
            media_candidates[kind] = ref
        if source.get('schema') == 'j-upright-response-sft-reference-v1':
            media_candidates.setdefault(('upright', True), ref)
        source_checks.append(dict(id=ref['id'], parent_gt_sha256=ref['gt_reference']['row_sha256'],
                                  roles_times_geometry_and_provenance_valid=True))
        if (rank + 1) % 100 == 0:
            print(f'Resolved {rank + 1}/600 real GT references', flush=True)
    # A blank generation template must not be accepted as completed predictions.
    try:
        match(refs, template, prepared)
    except ValueError as exc:
        assert '600 generated videos missing' in str(exc)
    else:
        raise AssertionError('Unfilled generation template was incorrectly accepted')
    known_issues = {r['id']: r['error'] for r in read(SOURCE_ISSUES)['records']
                    if r['id'] in ids}
    media_checks = []
    decoded = set()
    if media:
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        for (origin, nonzero), ref in media_candidates.items():
            if ref['id'] in decoded:
                continue
            decoded.add(ref['id'])
            condition, audit = load_conditioning(ref['gt_reference'])
            assert set(condition) == {'speaker_audio', 'speaker_sample_rate', 'listener_first_frame'}
            image = condition['listener_first_frame']
            assert image.size == (512, 512)
            speaker = condition['speaker_audio']
            sr = condition['speaker_sample_rate']
            assert np.isfinite(speaker).all() and abs(len(speaker) / sr - ref['duration_s']) <= 1 / sr
            listener, listener_sr, audio_audit = load_listener_audio(ref['gt_reference'])
            assert np.isfinite(listener).all()
            assert abs(len(listener) / listener_sr - ref['duration_s']) <= 1 / listener_sr
            selection = selected[ids.index(ref['id'])]
            assert bool(np.any(listener != 0)) == selection['listener_waveform_nonzero']
            record = dict(id=ref['id'], origin=origin,
                listener_waveform_nonzero=selection['listener_waveform_nonzero'],
                first_frame_size=list(image.size), speaker_samples=len(speaker), speaker_sample_rate=sr,
                listener_samples=len(listener), listener_sample_rate=listener_sr,
                conditioning_audit=audit, listener_audio_audit=audio_audit)
            print('Real conditioning and listener audio checked:', ref['id'], flush=True)
            if ref['id'] in known_issues:
                record.update(known_gt_source_issue=known_issues[ref['id']],
                              issue_evidence=str(SOURCE_ISSUES), full_gt_decode_repeated=False)
            media_checks.append(record)
            save(WORK / 'MEDIA_READBACK_PROGRESS.json', dict(complete=False, cases=media_checks))
    result = dict(success=True, at=time.time(), references=600, retained256=256, added344=344,
        framework='OmniResponse evaluation.seamless.bind / reader / listener_metrics.suite',
        references_sha256=sha(prepared / 'references.jsonl'),
        selection_sha256=sha(WORK / 'selection.jsonl'),
        all600_resolved=True, frozen_artifacts_verified=True, empty_generation_template_rejected=True,
        unique_original_pairs=len({r['pair'] for r in selected}),
        source_groups=len({r['group_id'] for r in selected}),
        max_per_group=max(collections.Counter(r['group_id'] for r in selected).values()),
        source_checks=source_checks, media_checks=media_checks,
        known_source_issues=[dict(id=sid, error=error, evidence=str(SOURCE_ISSUES),
            policy='Keep requested sample; preserve native metric N/A, no frame padding or silent exclusion')
            for sid, error in known_issues.items()],
        full600_media_decode_performed=False, model_inference_started=False, scoring_started=False)
    save(WORK / ('FRAMEWORK_ACCEPTANCE.json' if media else 'FRAMEWORK_METADATA_ACCEPTANCE.json'), result)
    if media:
        save(WORK / 'MEDIA_READBACK_PROGRESS.json', dict(complete=True, cases=media_checks))
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_checks', 'media_checks')}, ensure_ascii=False, indent=2))
    print('Real media samples checked:', len(media_checks))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    check(media=not args.metadata_only)
