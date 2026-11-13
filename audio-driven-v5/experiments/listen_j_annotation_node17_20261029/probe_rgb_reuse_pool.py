"""Production-shaped CPU pool ABBA, with exact frames checked in private caches."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark import process_decode
ORIGINAL_INITIALIZER = process_decode.init_cpu_decoder


def init_reused_decoder():
    ORIGINAL_INITIALIZER()
    # Only this isolated spawned benchmark process uses the candidate.
    from listen_benchmark import frame_crop
    frame_crop.crop_original_frame = frame_crop.crop_original_frame_reused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[8, 16], default=16)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    proof = json.loads((W / 'monitor_snapshots/reusable_reformat_probe_threads1.json').read_text())
    assert proof['complete'] and proof['success']
    rows = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())['rows']
    output = W / f'monitor_snapshots/rgb_reuse_pool{args.workers}_probe.json'
    if output.exists():
        raise RuntimeError('Preserve earlier evidence')
    expected = {}
    report = {'at': time.time(), 'pid': os.getpid(), 'complete': False, 'workers': args.workers, 'runs': [],
              'scope': 'Same sixteen clips; CPU process startup, decode, crop, JPEG, private cache and IPC included; ABBA under ongoing production.'}

    def save():
        tmp = output.with_suffix('.reuse-tmp')
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, output)

    save()
    try:
        for arm in ['original', 'reused', 'reused', 'original']:
            process_decode.init_cpu_decoder = ORIGINAL_INITIALIZER if arm == 'original' else init_reused_decoder
            with tempfile.TemporaryDirectory(prefix='listen-rgb-reuse-', dir='/dev/shm') as tmp:
                decoder = process_decode.ProcessMediaDecoder(tmp, W / 'resources', True)
                run = {'arm': arm, 'clips': []}
                start = time.monotonic()
                try:
                    slots = json.loads((W / 'selected.json').read_text())['decode_slots']
                    for row, media, metrics in decoder.map(rows, args.workers, slots):
                        if media is None:
                            raise RuntimeError(f'{row["sample_id"]}: {metrics}')
                        signature = {'meta': media[8]['meta'],
                                     'pixels': [hashlib.sha256(im.tobytes()).hexdigest() for im in media[8]['images']]}
                        sid = row['sample_id']
                        expected.setdefault(sid, signature)
                        if signature != expected[sid]:
                            raise RuntimeError(f'Concurrent crop parity failed: {sid}')
                        run['clips'].append({'sample_id': sid, 'frames': len(media[8]['images']),
                                             'metrics': metrics, 'exact_equal': True})
                        del media
                    run['wall_s'] = time.monotonic() - start
                finally:
                    decoder.close()
            report['runs'].append(run)
            save()
            print(json.dumps({'arm': arm, 'workers': args.workers, 'clips': len(run['clips']),
                              'wall_s': run['wall_s']}), flush=True)
        report.update(complete=True, success=True, finished_at=time.time())
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc), finished_at=time.time())
        raise
    finally:
        process_decode.init_cpu_decoder = ORIGINAL_INITIALIZER
        save()


if __name__ == '__main__':
    main()
