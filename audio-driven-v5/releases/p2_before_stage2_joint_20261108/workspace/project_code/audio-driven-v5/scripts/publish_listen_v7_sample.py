"""Render a reproducible stratified sample of the frozen V7 primary manifest."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data_corpora/seamless_listen_v7_full'
OUT = ROOT / 'data_previews/seamless_listen_v7_sample'
MANIFEST = CORPUS / 'manifests/train_keep_primary.jsonl'
SEED = 202610221140


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return p.stdout


def select():
    pools = {(i, vocal): [] for i in range(4) for vocal in (True, False)}
    bins = [(2, 3), (3, 6), (6, 8), (8, 10.000001)]
    with MANIFEST.open() as f:
        for line in f:
            r = json.loads(line)
            assert r['effective_decision'] == 'keep' and r['admission_pool'] == 'primary'
            i = next(i for i, (a, b) in enumerate(bins) if a <= r['duration_s'] < b)
            pools[i, bool(r['listener_human_evidence'])].append(r)
    rng = random.Random(SEED)
    used = set()
    rows = []
    counts = {}
    for (i, vocal), pool in pools.items():
        counts[f'{i}/{vocal}'] = len(pool)
        rng.shuffle(pool)
        selected = []
        for r in pool:
            if r['group_id'] in used:
                continue
            used.add(r['group_id'])
            selected.append(dict(r, preview_stratum=f'{bins[i][0]}–{min(bins[i][1], 10):g}s',
                                 preview=f"media/{r['sample_id']}/"))
            if len(selected) == 3:
                break
        assert len(selected) == 3, (i, vocal, len(selected))
        rows.extend(selected)
    write(OUT / 'selection.json', dict(seed=SEED, manifest=str(MANIFEST),
          manifest_sha256=sha(MANIFEST), stratum_counts=counts, samples=rows,
          note='Three random clips per duration/human-evidence stratum; 24 distinct source groups. Not a prevalence estimate.'))
    return rows


def export(row):
    folder = OUT / row['preview']
    folder.mkdir(parents=True, exist_ok=True)
    duration = row['duration_s']
    audio_checks = {}
    for role in ('speaker', 'listener'):
        track = row['normalized_audio'][role]
        assert sha(track['path']) == track['sha256'], 'Source WAV hash changed'
        n = round(duration * 48000)
        with sf.SoundFile(track['path']) as f:
            assert f.samplerate == 48000 and f.channels == 1
            f.seek(round(row['start'] * 48000))
            wave = f.read(n, dtype='int32')
        assert len(wave) == n
        sf.write(folder / (role + '.wav'), wave, 48000, subtype='PCM_24')
        audio_checks[role] = dict(samples=n, sample_rate=48000, source_hash_verified=True)
    src = row['sources']['listener']
    # Read exactly the manifest's indexed MP4 member, without unpacking whole TARs.
    uri = f"subfile,,start,{src['offset']},end,{src['offset'] + src['byte_size']},,:{src['archive']}"
    c = row['listener_crop']
    ff = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '2', '-filter_threads', '1']
    inp = ['-noautorotate', '-ss', str(row['start']), '-i', uri]
    vf = (f"setpts=PTS-STARTPTS,crop={c['size']}:{c['size']}:{c['x']}:{c['y']},"
          'scale=512:512,fps=25,setsar=1')
    run(ff + inp + ['-t', str(duration), '-vf', vf, '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p',
        '-metadata:s:v:0', 'rotate=0', '-movflags', '+faststart', str(folder / 'silent.mp4')])
    run(ff + ['-i', str(folder / 'silent.mp4'), '-frames:v', '1', str(folder / 'reference.jpg')])
    run(ff + inp + ['-vf', f"drawbox=x={c['x']}:y={c['y']}:w={c['size']}:h={c['size']}:color=lime:t=5,scale=640:-2",
        '-frames:v', '1', str(folder / 'source.jpg')])
    cmd = ff + ['-i', str(folder / 'silent.mp4'), '-i', str(folder / 'speaker.wav'),
        '-i', str(folder / 'listener.wav'), '-filter_complex_threads', '1',
        '-filter_complex', '[1:a][2:a]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[m]']
    for mode, audio in [('both', '[m]'), ('speaker', '1:a'), ('listener', '2:a')]:
        cmd += ['-map', '0:v', '-map', audio, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-t', str(duration), '-movflags', '+faststart', str(folder / (mode + '.mp4'))]
    run(cmd)
    checks = {}
    for mode, channels in [('both', 2), ('speaker', 1), ('listener', 1)]:
        path = folder / (mode + '.mp4')
        info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
        v = next(x for x in info['streams'] if x['codec_type'] == 'video')
        a = next(x for x in info['streams'] if x['codec_type'] == 'audio')
        assert (v['width'], v['height'], a['channels']) == (512, 512, channels)
        assert v['pix_fmt'] == 'yuv420p' and v['codec_name'] == 'h264' and a['codec_name'] == 'aac'
        assert abs(float(v['duration']) - duration) <= .081
        assert abs(float(a['duration']) - duration) <= .025
        run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
             '-map', '0:v', '-map', '0:a', '-f', 'null', '-'])
        checks[mode] = dict(video_seconds=float(v['duration']), audio_seconds=float(a['duration']),
                            channels=channels, decoded=True, sha256=sha(path))
    result = dict(row, media_verified=True, media_checks=checks, audio_checks=audio_checks)
    write(folder / 'export.json', result)
    print(json.dumps(dict(verified=row['sample_id'],duration=duration)), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = select()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        samples = list(pool.map(export, rows))
    write(OUT / 'samples.json', dict(seed=SEED, samples=samples,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_manifest_sha256=sha(MANIFEST), complete=True))
    shutil.copyfile(ROOT / 'scripts/listen_v7_sample.html', OUT / 'index.html')
    webroot = ROOT / 'data_previews/seamless_face_listening_v1'
    link = webroot / 'listen_v7'
    if not link.exists():
        link.symlink_to(OUT, target_is_directory=True)
    assert link.resolve() == OUT.resolve()
    old = ROOT / 'data_previews/seamless_listen_qc3_review'
    if not (old / 'qc3_history.html').exists():
        shutil.copyfile(old / 'index.html', old / 'qc3_history.html')
    (old / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/listen_v7/"><a href="/listen_v7/">打开 V7 全量主池随机抽样</a>')
    print('Published http://127.0.0.1:8875/listen_v7/ ; previous /listen_qc3/ redirects here.', flush=True)


if __name__ == '__main__':
    main()
