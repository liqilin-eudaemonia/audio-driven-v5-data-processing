"""CPU audit of frozen holdout candidates; does not score or select models."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from listen_response_sft.data import resolve_example, load_example
from listen_benchmark.j_omni import read_audio
from p2_listen.contracts import sha, atomic_json

WORK = Path(__file__).resolve().parent


def main():
    audit = json.loads((WORK / 'CANDIDATE_AUDIT.json').read_text())
    assert sha(WORK / 'candidates.json') == audit['selected_sha256']
    assert sha(WORK / 'parent_references.json') == audit['references_sha256']
    candidates = json.loads((WORK / 'candidates.json').read_text())
    refs = {r['id']: r for r in json.loads((WORK / 'parent_references.json').read_text())}
    details, failures, seen_groups = [], [], set()
    for candidate in candidates:
        sid = candidate['sample_id']
        try:
            ref = refs[sid]['gt_reference']
            with open(ref['manifest'], 'rb') as stream:
                stream.seek(ref['offset'])
                raw = stream.read(ref['bytes'])
            assert hashlib.sha256(raw).hexdigest() == ref['row_sha256']
            assert sha(ref['crop_config']) == ref['crop_config_sha256']
            row = json.loads(raw)
            crop_config = json.loads(Path(ref['crop_config']).read_text())
            resolved = resolve_example(row, crop_config)
            source = resolved['source']
            for key in ('sample_id', 'pair', 'split', 'group_id', 'duration_s'):
                assert row[key] == candidate[key], key
            for key in ('listener_id', 'speaker_id', 'source_start_s', 'source_end_s',
                        'source', 'video_transform', 'normalized_audio'):
                assert source[key] == candidate['source'][key], key
            wave, sr, identity = read_audio(candidate['audio_recipe'])
            assert np.isfinite(wave).all()
            assert abs(len(wave) / sr - candidate['duration_s']) <= 1 / sr
            assert identity['full_silence'] == candidate['listener_audio_is_silent']
            first_in_group = candidate['group_id'] not in seen_groups
            detail = dict(id=sid, group_id=candidate['group_id'],
                          media_contract_matches=True, listener_samples=len(wave),
                          listener_audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                          current_caption_differs=row['target']['text'] != candidate['caption'],
                          decoded_input_canary=first_in_group)
            if first_in_group:
                loaded = load_example(row, crop_config)
                image = np.asarray(loaded['inputs']['listener_first_frame'])
                assert image.shape == (512, 512, 3), image.shape
                assert loaded['audit']['future_listener_media_in_inputs'] is False
                detail.update(first_frame_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                              speaker_samples=len(loaded['inputs']['speaker_audio']),
                              first_frame_pts=loaded['audit']['source_frame_time_s'])
                seen_groups.add(candidate['group_id'])
            details.append(detail)
        except Exception as exc:
            failures.append(dict(id=sid, error=repr(exc)))
    result = dict(success=not failures, at=time.time(), candidates=len(candidates),
                  checked=len(details), input_decode_canaries=len(seen_groups),
                  no_gpu=True, no_generation_or_quality_scores=True,
                  final_test_ready=False,
                  remaining=['conditioning preparation', 'historical source independence unresolved'],
                  source_sha256=sha(Path(__file__)), selected_sha256=sha(WORK/'candidates.json'),
                  references_sha256=sha(WORK/'parent_references.json'), details=details, failures=failures)
    atomic_json(WORK / 'MEDIA_AUDIT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('details',)}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
