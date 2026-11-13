"""CPU-only evidence collection; does not generate a model benchmark score."""
import collections
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT / 'evaluation/lm-listener/dataset'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def key(row):
    return row['fname'], int(row['split_start_frame']), int(row['split_end_frame'])


def fingerprint(row):
    h = hashlib.sha256()
    for name, value in sorted(row.items()):
        h.update(name.encode())
        if torch.is_tensor(value):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def save(name, obj):
    (WORK / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    combined_path = DATA / 'trevorconanstephen/segments_test_no_overlap.pth'
    combined = torch.load(combined_path, map_location='cpu', weights_only=False)
    combined_keys = collections.Counter(key(r) for r in combined)
    combined_signatures = collections.Counter(fingerprint(r) for r in combined)
    all_keys = collections.Counter()
    all_signatures = collections.Counter()
    result = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'combined': {'path': str(combined_path), 'sha256': digest(combined_path),
                           'count': len(combined), 'unique_frame_ranges': len(combined_keys),
                           'repeated_frame_range_rows': len(combined) - len(combined_keys)}, 'persons': {},
              'model_generation_performed': False, 'official_benchmark_score': False}
    manifest = []
    for person in ['trevor', 'conanglobal', 'stephenglobal']:
        filename = 'segments_test_no_overlap.pth'
        source = DATA / person / filename
        inferred = not source.exists()
        if inferred:
            source = DATA / person / 'segments_test.pth'
        segments = torch.load(source, map_location='cpu', weights_only=False)
        keys = collections.Counter(key(s) for s in segments)
        assert not (set(all_keys) & set(keys))
        all_keys.update(keys)
        signatures = collections.Counter(fingerprint(s) for s in segments)
        assert not (signatures - combined_signatures)
        all_signatures.update(signatures)
        lengths = collections.Counter()
        fields = set()
        for row_index, row in enumerate(segments):
            fields |= set(row)
            fn, start, end = key(row)
            lengths[int(row['p0_exp'].shape[0]) - (end - start)] += 1
            for role in ['p0', 'p1']:
                for field, width in [('exp', 50), ('pose', 6), ('shape', 100), ('detail', 128)]:
                    tensor = row[f'{role}_{field}']
                    assert tensor.ndim == 2 and tensor.shape[1] == width
                    assert tensor.shape[0] >= end - start and torch.isfinite(tensor).all()
            manifest.append({'sample_id': hashlib.sha256(json.dumps((person, row_index, key(row))).encode()).hexdigest()[:24],
                             'source_row_index': row_index,
                             'person': person, 'source_segments': str(source), 'fname': fn,
                             'start_frame': start, 'end_frame_exclusive': end, 'fps': 30,
                             'start_s': start / 30, 'end_s': end / 30,
                             'coefficient_frames': int(row['p0_exp'].shape[0]),
                             'membership_verified_in_combined_no_overlap': key(row) in combined_keys,
                             'raw_media_status': 'not_resolved',
                             'needs': ['paired_speaker_waveform', 'listener_first_frame'],
                             'listener_gt': 'p0_exp50+p0_pose6', 'speaker_gt': 'p1_exp50+p1_pose6'})
        result['persons'][person] = {'source': str(source), 'sha256': digest(source),
                                     'count': len(segments), 'videos': len({s['fname'] for s in segments}),
                                     'unique_frame_ranges': len(keys),
                                     'repeated_frame_range_rows': len(segments) - len(keys),
                                     'standalone_no_overlap_file_exists': not inferred,
                                     'all_fields_equal_to_combined_no_overlap': True,
                                     'tensor_length_minus_exclusive_range': dict(lengths),
                                     'fields': sorted(fields)}
        if person == 'trevor':
            first = segments[0]
            smoke = WORK / 'gt_self_check'
            out = smoke.joinpath(*Path(first['fname']).parts[:-1])
            out.mkdir(parents=True, exist_ok=True)
            pred = torch.cat([first['p0_exp'][:60], first['p0_pose'][:60]], dim=1).numpy()
            np.save(out / f"{Path(first['fname']).stem}_{first['split_start_frame']}_pred.npy", pred)
            torch.save([first], smoke / 'segments.pth')
        del segments
    assert all_keys == combined_keys
    assert all_signatures == combined_signatures
    result['combined_is_exact_disjoint_union'] = True
    result['stephen_original_test_exactly_present_in_official_combined_no_overlap'] = True
    result['duplicate_policy'] = 'Preserve released rows and source indices; do not silently deduplicate. Official scorer maps (fname,start) to one row; reconcile generation coverage before a formal run.'
    archive = ROOT / 'evaluation/lm-listener/data_release/lm_listener_data.zip'
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        media = [s for s in names if Path(s).suffix.lower() in
                 {'.wav', '.mp3', '.m4a', '.mp4', '.avi', '.mov', '.png', '.jpg', '.jpeg'}]
    result['release_archive'] = {'path': str(archive), 'entries': len(names),
                                 'raw_media_entries': len(media), 'media_examples': media[:10]}
    with (WORK / 'test_manifest.jsonl').open('w') as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result['manifest'] = {'path': str(WORK / 'test_manifest.jsonl'), 'count': len(manifest),
                          'sha256': digest(WORK / 'test_manifest.jsonl')}
    save('AUDIT.json', result)
    print(json.dumps({'persons': {p: {k: r[k] for k in ['count', 'videos', 'all_fields_equal_to_combined_no_overlap']}
                                 for p, r in result['persons'].items()},
                      'combined_is_exact_disjoint_union': True,
                      'raw_media_entries': len(media), 'manifest_count': len(manifest)}))


if __name__ == '__main__':
    main()
