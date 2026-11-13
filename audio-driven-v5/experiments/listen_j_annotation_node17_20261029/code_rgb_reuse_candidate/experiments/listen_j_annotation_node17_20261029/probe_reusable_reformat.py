"""Compare crop conversion on distinct real frames, avoiding same-frame warm bias."""
from pathlib import Path
import argparse
import json
import sys
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(W)]
import av
import numpy as np
from PIL import Image
from av.video.reformatter import VideoReformatter
from listen_benchmark.full_raw import open_source
from listen_benchmark.frame_crop import crop_original_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, choices=[0, 1, 2, 4], default=0)
    args = parser.parse_args()
    suffix = '' if args.threads == 0 else f'_threads{args.threads}'
    output = W / f'monitor_snapshots/reusable_reformat_probe{suffix}.json'
    if output.exists():
        raise RuntimeError('Preserve prior evidence')
    rows = json.loads((W / 'monitor_snapshots/cuda_decode_extended_manifest.json').read_text())['rows'][::2]
    report = {'at': time.time(), 'complete': False, 'runs': [], 'reformat_threads': args.threads,
              'scope': 'Sixteen distinct sampled frames per source; same input frame per arm; rotated order; crop CPU time excludes decoding.'}
    try:
        for row in rows:
            source = row['source']
            converter = VideoReformatter()
            c = source['video_transform']['crop']
            entries = []

            def reused(frame):
                rgb = converter.reformat(frame, format='rgb24', threads=args.threads).to_ndarray()
                return Image.fromarray(rgb[c['y']:c['y'] + c['height'], c['x']:c['x'] + c['width']]).resize(
                    (512, 512), resample=Image.Resampling.BICUBIC)

            with open_source(source['source']) as (container, _):
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                start = source['source_start_s']
                container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    t = float(frame.pts * frame.time_base) - start
                    if t < len(entries) / 8:
                        continue
                    if t >= source['duration_s'] or len(entries) >= 16:
                        break
                    choices = ['original', 'reused']
                    if len(entries) % 2:
                        choices.reverse()
                    times, images = {}, {}
                    for arm in choices:
                        tick = time.monotonic()
                        image = (crop_original_frame(frame, source['video_transform']) if arm == 'original'
                                 else reused(frame))
                        times[arm] = time.monotonic() - tick
                        images[arm] = np.asarray(image)
                    diff = np.abs(images['original'].astype(np.int16) - images['reused'].astype(np.int16))
                    entries.append({'source_pts': frame.pts, 'equal': bool(not np.any(diff)),
                                    'max_difference': int(diff.max()), 'times': times})
            if len(entries) != 16:
                raise RuntimeError('Need sixteen distinct covered frames')
            item = {'sample_id': row['sample_id'], 'frames': entries,
                    'all_equal': all(e['equal'] for e in entries),
                    'total_crop_s': {a: sum(e['times'][a] for e in entries) for a in ['original', 'reused']}}
            report['runs'].append(item)
            print(json.dumps({k: v for k, v in item.items() if k != 'frames'}), flush=True)
        report.update(complete=True, success=all(r['all_equal'] for r in report['runs']))
    except BaseException as exc:
        report.update(complete=True, success=False, error=repr(exc))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
