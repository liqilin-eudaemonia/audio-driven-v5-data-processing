"""Inspect finished rule-turn exports; may run while other pairs render."""
import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_previews/seamless_rule_turns_v2'


def check(job):
    path, duration, dimensions, fps, audio, decode = job
    errors = []
    try:
        p = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                            '-of', 'json', str(path)], capture_output=True, text=True, check=True)
        info = json.loads(p.stdout)
        video = [s for s in info['streams'] if s['codec_type'] == 'video']
        sound = [s for s in info['streams'] if s['codec_type'] == 'audio']
        if dimensions:
            assert len(video) == 1, 'missing video'
            assert (video[0]['width'], video[0]['height']) == dimensions, 'wrong dimensions'
            n, d = map(int, video[0]['avg_frame_rate'].split('/'))
            assert abs(n / d - fps) < .01, 'wrong frame rate'
        assert bool(sound) == audio, 'wrong audio stream count'
        for stream in info['streams']:
            if stream['codec_type'] in ('video', 'audio'):
                assert abs(float(stream['duration']) - duration) <= .13, 'stream duration mismatch'
        assert abs(float(info['format']['duration']) - duration) <= .13, 'container duration mismatch'
        if decode:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1',
                                     '-i', str(path), '-map', '0', '-f', 'null', '-'],
                                    capture_output=True, text=True)
            assert result.returncode == 0 and not result.stderr.strip(), result.stderr[-1000:]
    except Exception as e:
        errors.append(str(e))
    return {'path': str(path.relative_to(OUT)), 'size': path.stat().st_size if path.exists() else 0,
            'mtime_ns': path.stat().st_mtime_ns if path.exists() else 0,
            'decoded': decode, 'errors': errors}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--directory', type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.directory.resolve()
    cache_file = OUT / 'media_validation_cache.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    pairs = [json.loads(p.read_text()) for p in sorted((OUT / 'media').glob('*/pair.json'))]
    jobs = []
    for pair in pairs:
        jobs.append((OUT / pair['conversation_path'], pair['duration'], (640, 360), 10, True, False))
        longest = max(pair['clips'], key=lambda c: c['duration'], default=None)
        for clip in pair['clips']:
            folder = OUT / clip['path']
            for name, shape, fps, audio in [
                ('faces.mp4', (1024, 512), 25, True),
                ('speaker_512.mp4', (512, 512), 25, True),
                ('listener_512.mp4', (512, 512), 25, True),
                ('context.mp4', (640, 360), 15, False),
                ('speaker.m4a', None, None, True), ('listener.m4a', None, None, True),
            ]:
                jobs.append((folder / name, clip['duration'], shape, fps, audio,
                             clip is longest and name == 'faces.mp4'))
    pending = []
    for job in jobs:
        p = job[0]; previous = cache.get(str(p.relative_to(OUT)))
        if not (p.exists() and previous and not previous['errors']
                and previous['size'] == p.stat().st_size
                and previous['mtime_ns'] == p.stat().st_mtime_ns
                and (not job[-1] or previous['decoded'])):
            pending.append(job)
    print(f'Checking {len(pending)} files; {len(jobs)-len(pending)} cached', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(check, pending), 1):
            cache[result['path']] = result
            if result['errors']: print('ERROR', result, flush=True)
            if i % 100 == 0: print('CHECKED', i, flush=True)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    results = [cache[str(j[0].relative_to(OUT))] for j in jobs]
    summary = dict(pairs=len(pairs), clips=sum(len(p['clips']) for p in pairs),
                   inspected_files=len(results), fully_decoded_files=sum(r['decoded'] for r in results),
                   failures=[r for r in results if r['errors']],
                   export_errors=[dict(pair=p['pair'], error=e) for p in pairs for e in p['errors']])
    (OUT / 'media_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary['failures'] or summary['export_errors']: raise SystemExit(1)


if __name__ == '__main__':
    main()
