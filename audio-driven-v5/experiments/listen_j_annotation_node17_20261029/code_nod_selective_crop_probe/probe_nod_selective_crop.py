"""Experimental CPU-only ABBA probe; never changes production worker code.

Build a narrowly changed copy of the current renderer and compare the resulting
MP4 bytes and complete media metadata. Refuse to run during the nod80 window.
"""
from pathlib import Path
import hashlib
import inspect
import json
import statistics
import sys
import tempfile
import time

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import listen_j_annotation_nod as baseline
from listen_visual_native8_nod import decode_slot


def candidate_renderer():
    source = inspect.getsource(baseline.render_video)
    replacements = [
        ('def render_video(', 'def render_video_selective('),
        ('images=[];times=[]',
         'images={};times=[];next_grid=0;sample_count=math.ceil(duration*25-1e-9);last_frame=None'),
        ("            images.append(crop_original_frame(frame,source['video_transform']) if fast_crop\n"
         "                          else crop_resize(frame.to_image(),source['video_transform']));times.append(t)",
         "            index=len(times);times.append(t);last_frame=frame\n"
         "            if next_grid < sample_count and next_grid/25 <= t:\n"
         "                images[index]=(crop_original_frame(frame,source['video_transform']) if fast_crop\n"
         "                               else crop_resize(frame.to_image(),source['video_transform']))\n"
         "                while next_grid < sample_count and next_grid/25 <= t:next_grid+=1"),
        ('    if not images or times[0]>=.1', '    if not times or times[0]>=.1'),
        ("    movie_path=directory/'video.mp4'",
         "    if indices[-1] not in images:\n"
         "        assert indices[-1] == len(times)-1\n"
         "        images[indices[-1]]=(crop_original_frame(last_frame,source['video_transform']) if fast_crop\n"
         "                            else crop_resize(last_frame.to_image(),source['video_transform']))\n"
         "    assert set(indices) == set(images), 'Selected source frames differ'\n"
         "    movie_path=directory/'video.mp4'"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Baseline renderer changed; re-review the experimental transformation')
        source = source.replace(old, new)
    env = dict(vars(baseline))
    exec(compile(source, '<experimental-selective-nod-crop>', 'exec'), env)
    return env['render_video_selective'], source


def main():
    report = json.loads((W / 'monitor_snapshots/process_all_nod80_stable_window.json').read_text())
    if not report['complete']:
        raise RuntimeError('Do not add benchmark CPU load during the production comparison')
    output = W / 'monitor_snapshots/nod_selective_crop_probe.json'
    if output.exists():
        raise RuntimeError('Keep the prior probe; do not overwrite it')
    probe = json.loads((W / 'monitor_snapshots/nod_crop_redundancy_probe.json').read_text())
    rows = []
    for r in probe['records']:
        sid = r['sample_id']
        path = W / 'nod/results' / hashlib.sha256(sid.encode()).hexdigest()[:2] / (sid + '.json')
        d = json.loads(path.read_text())
        rows.append({'sample_id': sid, 'source': d['source']})
    rows.sort(key=lambda r: (r['source']['video_transform']['source_size'][0] *
                             r['source']['video_transform']['source_size'][1],
                             r['source']['duration_s'], r['sample_id']))
    chosen = [rows[0], rows[-1]]
    candidate, source = candidate_renderer()
    runs = []
    for row in chosen:
        expected = None
        for arm in ['baseline', 'selective', 'selective', 'baseline']:
            renderer = baseline.render_video if arm == 'baseline' else candidate
            with tempfile.TemporaryDirectory(prefix='listen-nod-crop-probe-', dir='/dev/shm') as tmp:
                started = time.monotonic()
                slots = json.loads((W / 'selected.json').read_text())['decode_slots']
                with decode_slot(W / 'resources', slots):
                    acquired = time.monotonic()
                    path, meta = renderer(row, Path(tmp), fast_crop=True)
                    elapsed = time.monotonic() - acquired
                if expected is None:
                    expected = meta
                if meta != expected:
                    raise RuntimeError(f'Media bytes or provenance changed: {row["sample_id"]}, {arm}')
                entry = {'sample_id': row['sample_id'], 'arm': arm, 'render_s': elapsed,
                         'decode_lock_wait_s': acquired-started, 'video_sha256': meta['video_sha256'],
                         'source_frames': len(meta['source_frame_times_s']),
                         'selected_frames': len(set(meta['output_to_source_indices'])),
                         'source_size': row['source']['video_transform']['source_size']}
                runs.append(entry)
                print(json.dumps(entry), flush=True)
    out = {'at': time.time(), 'runs': runs, 'media_and_metadata_equal': True,
           'candidate_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'baseline_source_sha256': hashlib.sha256(inspect.getsource(baseline.render_video).encode()).hexdigest(),
           'mean_render_s': {arm: statistics.mean(x['render_s'] for x in runs if x['arm'] == arm)
                             for arm in ['baseline', 'selective']},
           'scope': 'Two known production inputs, CPU render only, ABBA under ongoing production load; no end-to-end speedup claim or production deployment.'}
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out['mean_render_s']), flush=True)


if __name__ == '__main__':
    main()
