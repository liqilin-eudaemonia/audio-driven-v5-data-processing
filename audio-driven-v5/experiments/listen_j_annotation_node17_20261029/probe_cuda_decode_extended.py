"""Sequential, bounded parity probe after the four-group measurement finishes.

This is an isolated experiment. No production configuration, cache files, or
results are changed. It exercises the opt-in hybrid decoder implementation.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(W)]
from listen_benchmark import visual_media_coverage as media
from listen_benchmark.frame_crop import crop_original_frame
from listen_visual_native8_nod import decode_slot
from listen_benchmark.hybrid_video_decode import prepare_hybrid_media


def save(path, data):
    temp = path.with_suffix('.probe-tmp')
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def signature(result):
    return {'meta': result[8]['meta'],
            'pixels': [hashlib.sha256(im.tobytes()).hexdigest() for im in result[8]['images']]}


def hybrid(row):
    return prepare_hybrid_media(row, device='0', fast_crop=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
    marker = W / 'monitor_snapshots/groups4_trial_complete.json'
    if not marker.exists():
        raise RuntimeError('Wait for four-group measurement; do not contaminate its timing')
    report_path = W / 'monitor_snapshots/cuda_decode_extended.json'
    if report_path.exists():
        raise RuntimeError('Preserve prior evidence')
    manifest = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())
    report = {'at': time.time(), 'pid': os.getpid(), 'complete': False, 'runs': [],
              'code_sha256': {name: hashlib.sha256((ROOT / 'src/listen_benchmark' / name).read_bytes()).hexdigest()
                              for name in ['hybrid_video_decode.py', 'visual_media_coverage.py', 'visual_native8.py']},
              'scope': 'Sequential matched CPU/hybrid pairs on distinct real sources; no deployment or aggregate speed claim.'}
    save(report_path, report)
    try:
        for index, row in enumerate(manifest['rows']):
            signatures, runs = {}, {}
            order = ['cpu', 'hybrid'] if index % 2 == 0 else ['hybrid', 'cpu']
            for arm in order:
                slots = json.loads((W / 'selected.json').read_text())['decode_slots']
                with decode_slot(W / 'resources', slots):
                    start, cpu_start = time.monotonic(), time.process_time()
                    if arm == 'cpu':
                        result = media.prepare_media_covered(row['source'], frame_crop=crop_original_frame)
                        info = {}
                    else:
                        result, info = hybrid(row['source'])
                    elapsed, cpu_elapsed = time.monotonic() - start, time.process_time() - cpu_start
                signatures[arm] = signature(result)
                runs[arm] = {'wall_s': elapsed, 'cpu_s': cpu_elapsed, 'frames': len(result[8]['images']), **info}
                del result
            equal = signatures['cpu'] == signatures['hybrid']
            entry = {'sample_id': row['sample_id'], 'source_size': row['source']['video_transform']['source_size'],
                     'duration_s': row['source']['duration_s'], 'arms': runs, 'exact_equal': equal}
            report['runs'].append(entry)
            save(report_path, report)
            print(json.dumps(entry), flush=True)
            if not equal:
                report['mismatch'] = signatures
                raise RuntimeError('Frame pixels, JPEG hash, timing, crop or provenance differ')
        report.update(complete=True, success=True, finished_at=time.time())
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc), finished_at=time.time())
        raise
    finally:
        save(report_path, report)


if __name__ == '__main__':
    main()
