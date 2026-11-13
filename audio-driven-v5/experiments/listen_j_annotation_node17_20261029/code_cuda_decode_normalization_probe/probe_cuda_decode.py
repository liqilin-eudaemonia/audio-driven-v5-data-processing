"""Bounded real-input CUDA decode spike, no production code/cache writes."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import av
from av.codec.hwaccel import HWAccel
from listen_labeling.full_source import source_uri
from listen_benchmark.full_raw import SourceBytes
from listen_benchmark import visual_media_coverage as media
from listen_benchmark.frame_crop import crop_original_frame
from listen_visual_native8_nod import decode_slot


@contextmanager
def open_cuda(source):
    settings = HWAccel('cuda', device='0', allow_software_fallback=False)
    try:
        container = av.open(source_uri(source), timeout=(30, 60), hwaccel=settings)
    except av.error.InvalidDataError:
        with SourceBytes(source) as member, av.open(member, hwaccel=settings) as container:
            yield container, 'bounded_original_bytes'
    else:
        with container:
            yield container, 'ffmpeg_subfile'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=range(8), default=7)
    parser.add_argument('--sample-id')
    parser.add_argument('--normalize-yuv420p', action='store_true')
    parser.add_argument('--tag', default='cuda_decode_probe')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        raise ValueError('Invalid output tag')
    output = W / 'monitor_snapshots' / (args.tag + '.json')
    if output.exists():
        raise RuntimeError('Preserve previous evidence')
    assert json.loads((W / 'monitor_snapshots/decode_slots128_stable_window.json').read_text())['complete']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    probe = json.loads((W / 'monitor_snapshots/nod_crop_redundancy_probe.json').read_text())
    rows = []
    for r in probe['records']:
        sid = r['sample_id']
        p = W / 'nod/results' / hashlib.sha256(sid.encode()).hexdigest()[:2] / (sid + '.json')
        rows.append(json.loads(p.read_text()))
    rows.sort(key=lambda r: (r['source']['video_transform']['source_size'][0] * r['source']['video_transform']['source_size'][1], r['source']['duration_s']))
    chosen = [rows[0], rows[-1]] if args.sample_id is None else [r for r in rows if r['sample_id'] == args.sample_id]
    if not chosen:
        raise ValueError('Requested sample absent from bounded source list')
    baseline = media.open_source
    runs = []
    report = {'at': time.time(), 'pyav': av.__version__, 'gpu': args.gpu, 'runs': runs,
              'normalize_yuv420p': args.normalize_yuv420p,
              'scope': 'Selected real sources, ABBA, one decoder, concurrent production; not production rollout.'}
    try:
        for row in chosen:
            expected = None
            for arm in ['cpu', 'cuda', 'cuda', 'cpu']:
                media.open_source = baseline if arm == 'cpu' else open_cuda
                slots = json.loads((W / 'selected.json').read_text())['decode_slots']
                with decode_slot(W / 'resources', slots):
                    start = time.monotonic()
                    def crop(frame, transform):
                        if arm == 'cuda' and args.normalize_yuv420p:
                            if frame.format.name != 'nv12':
                                raise ValueError('Probe normalization supports NV12 only')
                            frame = frame.reformat(format='yuv420p')
                        return crop_original_frame(frame, transform)
                    result = media.prepare_media_covered(row['source'], frame_crop=crop)
                    elapsed = time.monotonic() - start
                signature = {'meta': result[8]['meta'], 'pixels': [hashlib.sha256(im.tobytes()).hexdigest() for im in result[8]['images']]}
                if expected is None:
                    expected = signature
                item = {'sample_id': row['sample_id'], 'source_size': row['source']['video_transform']['source_size'],
                        'arm': arm, 'seconds': elapsed, 'frames': len(result[8]['images']),
                        'exact_equal': signature == expected}
                runs.append(item)
                print(json.dumps(item), flush=True)
                if signature != expected:
                    report['mismatch'] = {'cpu': expected, 'candidate': signature}
                    raise RuntimeError('Input bytes, timing or metadata differ')
                del result
        report['success'] = True
    except Exception as exc:
        report.update(success=False, error=repr(exc))
        raise
    finally:
        media.open_source = baseline
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
