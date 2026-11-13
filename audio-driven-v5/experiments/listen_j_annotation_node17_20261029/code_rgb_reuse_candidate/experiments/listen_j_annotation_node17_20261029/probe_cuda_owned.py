"""Isolated ABBA probe: retain CUDA frames until original 8fps selection."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from types import FunctionType

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import av
from av.codec.hwaccel import HWAccel
from listen_labeling.full_source import source_uri
from listen_benchmark.full_raw import SourceBytes
from listen_benchmark import visual_media_coverage as media
from listen_benchmark.frame_crop import crop_original_frame


@contextmanager
def owned_source(source):
    settings = HWAccel('cuda', device='0', allow_software_fallback=False, is_hw_owned=True)
    try:
        container = av.open(source_uri(source), timeout=(30, 60), hwaccel=settings)
    except av.error.InvalidDataError:
        with SourceBytes(source) as member, av.open(member, hwaccel=settings) as container:
            yield container, 'bounded_original_bytes'
    else:
        with container:
            yield container, 'ffmpeg_subfile'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
    output = W / 'monitor_snapshots/cuda_owned_probe.json'
    if output.exists():
        raise RuntimeError('Preserve earlier evidence')
    manifest = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())
    rows = [manifest['rows'][0], manifest['rows'][7]]
    report = {'at': time.time(), 'complete': False, 'runs': [],
              'scope': 'Two distinct known H264 8-bit sources, ABBA; original sample selection and frame builder.'}
    try:
        for row in rows:
            expected = None
            for arm in ['cpu', 'owned', 'owned', 'cpu']:
                transferred = 0

                def crop(frame, transform):
                    nonlocal transferred
                    if arm == 'owned':
                        assert frame.format.name == 'cuda'
                        original = frame
                        frame = frame.reformat(format='yuv420p')
                        for key in ['colorspace', 'color_range', 'color_primaries', 'color_trc']:
                            setattr(frame, key, getattr(original, key))
                        transferred += 1
                    return crop_original_frame(frame, transform)

                start, cpu_start = time.monotonic(), time.process_time()
                result = media.prepare_media_covered(row['source'], frame_crop=crop,
                            source_opener=owned_source if arm == 'owned' else None)
                elapsed, cpu_elapsed = time.monotonic() - start, time.process_time() - cpu_start
                signature = {'meta': result[8]['meta'],
                             'pixels': [hashlib.sha256(im.tobytes()).hexdigest() for im in result[8]['images']]}
                if expected is None:
                    expected = signature
                entry = {'sample_id': row['sample_id'], 'arm': arm, 'wall_s': elapsed, 'cpu_s': cpu_elapsed,
                         'frames': len(result[8]['images']), 'transferred_frames': transferred,
                         'exact_equal': signature == expected}
                report['runs'].append(entry)
                print(json.dumps(entry), flush=True)
                if not entry['exact_equal']:
                    report['mismatch'] = {'cpu': expected, 'owned': signature}
                    raise RuntimeError('Hardware-owned input differs from CPU')
                del result
        report.update(complete=True, success=True)
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc))
        raise
    finally:
        report['finished_at'] = time.time()
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
