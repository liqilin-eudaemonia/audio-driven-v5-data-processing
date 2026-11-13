"""CPU-only experiment: aligned planar-YUV ROI before original RGB conversion.

Not a production path. Exact equality, including chroma boundaries/dither, is
required; any mismatch rejects this proposal.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import av
import numpy as np
from PIL import Image
from listen_benchmark.full_raw import open_source
from listen_benchmark.frame_crop import crop_original_frame


def roi_crop(frame, transform):
    fmt = frame.format.name
    if fmt not in ('yuv420p', 'yuv420p10le'):
        raise ValueError('Unsupported experimental planar format')
    c = transform['crop']
    x, y, width, height = (c[k] for k in ('x', 'y', 'width', 'height'))
    align, halo = 32, 32
    left, top = max(0, (x - halo) // align * align), max(0, (y - halo) // align * align)
    right = min(frame.width, ((x + width + halo + align - 1) // align) * align)
    bottom = min(frame.height, ((y + height + halo + align - 1) // align) * align)
    cut = av.VideoFrame(right - left, bottom - top, fmt)
    dtype = np.dtype('uint8' if fmt == 'yuv420p' else '<u2')
    for i, (src, dst) in enumerate(zip(frame.planes, cut.planes)):
        scale = 1 if i == 0 else 2
        source = np.frombuffer(src, dtype=dtype).reshape(src.height, src.line_size // dtype.itemsize)
        target = np.zeros((dst.height, dst.line_size // dtype.itemsize), dtype=dtype)
        target[:, :dst.width] = source[top // scale:top // scale + dst.height,
                                      left // scale:left // scale + dst.width]
        dst.update(target.tobytes())
    for key in ['colorspace', 'color_range', 'color_primaries', 'color_trc']:
        setattr(cut, key, getattr(frame, key))
    rgb = cut.to_ndarray(format='rgb24')
    return Image.fromarray(rgb[y - top:y - top + height, x - left:x - left + width]).resize(
        (512, 512), resample=Image.Resampling.BICUBIC)


def main():
    output = W / 'monitor_snapshots/yuv_roi_probe.json'
    if output.exists():
        raise RuntimeError('Preserve previous evidence')
    rows = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())['rows'][::2]
    report = {'at': time.time(), 'complete': False, 'runs': [], 'scope': 'First selected frame from eight real sources; experimental, not enabled.'}
    try:
        for row in rows:
            source = row['source']
            with open_source(source['source']) as (container, _):
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                container.seek(int(source['source_start_s'] / stream.time_base), stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    if float(frame.pts * frame.time_base) >= source['source_start_s']:
                        break
                original = crop_original_frame(frame, source['video_transform'])
                start = time.monotonic()
                for _ in range(4):
                    candidate = roi_crop(frame, source['video_transform'])
                candidate_s = (time.monotonic() - start) / 4
                start = time.monotonic()
                for _ in range(4):
                    original = crop_original_frame(frame, source['video_transform'])
                baseline_s = (time.monotonic() - start) / 4
                diff = np.abs(np.asarray(original).astype(np.int16) - np.asarray(candidate).astype(np.int16))
                entry = {'sample_id': row['sample_id'], 'format': frame.format.name,
                         'crop': source['video_transform']['crop'], 'baseline_s': baseline_s,
                         'roi_s': candidate_s, 'equal': bool(not np.any(diff)),
                         'max_difference': int(diff.max()), 'different_values': int(np.count_nonzero(diff))}
                report['runs'].append(entry)
                print(json.dumps(entry), flush=True)
        report.update(complete=True, success=all(r['equal'] for r in report['runs']))
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
