"""Matched ABBA process-pool probe in isolated temporary caches, on GPU6 only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(W)]
from listen_benchmark.process_decode import ProcessMediaDecoder
from listen_benchmark.hybrid_process_decode import HybridMediaDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[4, 8, 16], default=4)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
    proof = json.loads((W / 'monitor_snapshots/cuda_decode_extended.json').read_text())
    assert proof['complete'] and proof['success'] and len(proof['runs']) == 16
    rows = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())['rows']
    assert len(rows) == 16
    output = W / f'monitor_snapshots/hybrid_pool{args.workers}_probe.json'
    if output.exists():
        raise RuntimeError('Preserve prior evidence')
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(6)
    if nv.nvmlDeviceGetMemoryInfo(handle).free < 4 * 1024 ** 3:
        raise RuntimeError('Insufficient free GPU memory; preserve production')
    report = {'at': time.time(), 'pid': os.getpid(), 'workers': args.workers, 'runs': [], 'complete': False,
              'scope': 'One GPU, same sixteen clips in ABBA, isolated caches, production models continue. Includes process startup and cache transport.'}
    expected = {}

    def save():
        temp = output.with_suffix('.poolprobe-tmp')
        temp.write_text(json.dumps(report, indent=2))
        os.replace(temp, output)

    save()
    try:
        for arm in ['cpu', 'hybrid', 'hybrid', 'cpu']:
            with tempfile.TemporaryDirectory(prefix='listen-hybrid-pool-', dir='/dev/shm') as tmp:
                decoder = (ProcessMediaDecoder(tmp, W / 'resources', True) if arm == 'cpu'
                           else HybridMediaDecoder(tmp, W / 'resources', True, '6'))
                done = threading.Event()
                usage = {'peak_own_decoder_memory_bytes': 0, 'peak_gpu_total_memory_bytes': 0, 'samples': 0,
                         'sample_errors': [], 'decoder_pids_seen': []}
                seen = set()

                def monitor():
                    while not done.is_set():
                        try:
                            processes = getattr(decoder.pool, '_processes', None) or {}
                            own = set(processes)
                            seen.update(own)
                            current = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                            used = sum(p.usedGpuMemory for p in current if p.pid in own
                                       and isinstance(p.usedGpuMemory, int) and 0 <= p.usedGpuMemory < 2 ** 63)
                            usage['peak_own_decoder_memory_bytes'] = max(usage['peak_own_decoder_memory_bytes'], used)
                            usage['peak_gpu_total_memory_bytes'] = max(usage['peak_gpu_total_memory_bytes'],
                                                                      nv.nvmlDeviceGetMemoryInfo(handle).used)
                            usage['samples'] += 1
                        except Exception as exc:
                            if len(usage['sample_errors']) < 5:
                                usage['sample_errors'].append(repr(exc))
                        done.wait(1)

                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                run = {'arm': arm, 'clips': [], 'memory': usage}
                started = time.monotonic()
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
                            raise RuntimeError(f'Concurrent media parity failed: {sid}')
                        run['clips'].append({'sample_id': sid, 'frames': len(media[8]['images']),
                                             'metrics': metrics, 'exact_equal': True})
                        del media
                    run['wall_s'] = time.monotonic() - started
                finally:
                    decoder.close()
                    done.set()
                    thread.join()
                    usage['decoder_pids_seen'] = sorted(seen)
                report['runs'].append(run)
                save()
                print(json.dumps({'arm': arm, 'workers': args.workers, 'clips': len(run['clips']),
                                  'wall_s': run['wall_s'], 'memory': usage}), flush=True)
        report.update(complete=True, success=True, finished_at=time.time())
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc), finished_at=time.time())
        raise
    finally:
        save()
        nv.nvmlShutdown()


if __name__ == '__main__':
    main()
