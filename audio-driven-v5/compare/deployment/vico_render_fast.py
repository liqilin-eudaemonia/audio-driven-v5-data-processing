"""PIRender with bounded asynchronous encoding; official FP32 batch=4 unchanged."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import subprocess
import time
from pathlib import Path

from vico_more_common import WORK, read, save, sha, gpu_guard, test_samples


class Encoder:
    """Two pending byte chunks at most; propagate worker and ffmpeg errors."""
    def __init__(self, path, asynchronous):
        self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '256x256', '-r', '30',
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset',
            'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE)
        self.pool = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = deque()
        self.digest = hashlib.sha256()

    def write(self, pixels):
        data = pixels.tobytes()
        self.digest.update(data)
        if self.pool is None:
            self.process.stdin.write(data)
        else:
            if len(self.pending) == 2:
                self.pending.popleft().result()
            self.pending.append(self.pool.submit(self.process.stdin.write, data))

    def close(self):
        try:
            for task in self.pending:
                task.result()
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise RuntimeError('ffmpeg encoding failed')
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            if self.pool:
                self.pool.shutdown(wait=True)

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self.pool:
            self.pool.shutdown(wait=True)
        self.process.stdin.close()


def model_for(out):
    import sys
    import torch
    import yaml
    from vico_more_render import VENDOR, WEIGHT
    sys.path.insert(0, str(VENDOR))
    from generators.face_model import FaceGenerator
    config = yaml.safe_load((VENDOR / 'config/face_demo.yaml').read_text())
    model = FaceGenerator(**config['gen']['param'])
    checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['net_G_ema'], strict=True)
    save(out / 'RENDER_WEIGHT_AUDIT.json', dict(strict=True, checkpoint_sha256=sha(WEIGHT),
        state='net_G_ema', config_sha256=sha(VENDOR / 'config/face_demo.yaml')))
    return model.eval().cuda()


def encode(model, image, dynamic, dest, fast=True):
    import numpy as np
    import torch
    from vico_more_render import semantics
    start = time.perf_counter()
    # One small transfer per clip instead of one transfer per four frames.
    all_semantics = semantics(dynamic, np.arange(len(dynamic))).cuda() if fast else None
    encoder = Encoder(dest, asynchronous=fast)
    try:
        with torch.inference_mode():
            for start_frame in range(0, len(dynamic), 4):
                end = min(start_frame + 4, len(dynamic))
                x = all_semantics[start_frame:end] if fast else semantics(dynamic, np.arange(start_frame, end)).cuda()
                result = model(image.expand(end-start_frame, -1, -1, -1), x)['fake_image']
                pixels = ((result.clamp(-1, 1)+1)/2*255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                encoder.write(pixels)
        encoder.close()
    except BaseException:
        encoder.abort()
        Path(dest).unlink(missing_ok=True)
        raise
    return dict(seconds=time.perf_counter()-start, rgb_sha256=encoder.digest.hexdigest(), frames=len(dynamic))


def decode_check(path, frames):
    import cv2
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (256, 256, 3)
            count += 1
    finally:
        cap.release()
    assert count == frames, (count, frames)


def render(out):
    import shutil
    import numpy as np
    from vico_more_render import first_frame, semantic_parity
    source = out if (out / 'GENERATION_COMPLETE.json').exists() else WORK / 'listenformer'
    gen = read(source / 'GENERATION_COMPLETE.json')
    assert gen['success']
    samples = {s['id']: s for s in test_samples()}
    assert len(gen['records']) == len(samples) and {r['id'] for r in gen['records']} == set(samples)
    archive = WORK / 'listenformer'
    archive_gen = {r['id']: r for r in read(archive / 'GENERATION_COMPLETE.json')['records']}
    archive_protocol = read(archive / 'protocol.json')
    archive_matches = sha(archive / 'GENERATION_COMPLETE.json') == archive_protocol['generation_complete_sha256']
    from vico_more_render import WEIGHT
    renderer_sha = sha(WEIGHT)
    archive_matches = archive_matches and renderer_sha == read(WORK / 'renderer/WEIGHT_AUDIT.json')['checkpoint_sha256']
    model = None
    records = []
    for rec in gen['records']:
        uid = rec['id']
        dest = out / 'scoring/listenformer' / (uid + '.listener.mp4')
        dest.parent.mkdir(parents=True, exist_ok=True)
        receipt = dest.with_suffix('.json')
        source_frame = samples[uid]['streams']['gt']
        if receipt.exists():
            prior = read(receipt)
            if dest.exists() and sha(dest) == prior['sha256'] and prior.get('driver_sha256') == rec['sha256'] and prior.get('source_sha256') == source_frame['sha256'] and prior.get('renderer_sha256') == renderer_sha:
                records.append(prior)
                continue
        old_receipt = archive / 'scoring/listenformer' / (uid + '.listener.json')
        if archive_matches and old_receipt.exists() and archive_gen[uid]['sha256'] == rec['sha256']:
            previous = read(old_receipt)
            if Path(previous['path']).is_file() and sha(previous['path']) == previous['sha256']:
                records.append(previous)
                continue
        assert sha(rec['path']) == rec['sha256']
        assert sha(source_frame['path']) == source_frame['sha256']
        dyn = np.load(rec['path'])['dynamic']
        if model is None:
            gpu_guard(18)
            model = model_for(out)
            save(out / 'RENDER_SEMANTIC_PARITY.json', semantic_parity(dyn))
        image = first_frame(source_frame['path']).cuda()
        temp = dest.with_name(dest.stem + '.tmp.mp4')
        timing = encode(model, image, dyn, temp)
        decode_check(temp, len(dyn))
        temp.replace(dest)
        r = dict(id=uid, path=str(dest), sha256=sha(dest), driver_sha256=rec['sha256'],
            source_sha256=source_frame['sha256'], renderer_sha256=renderer_sha, fps=30, width=256, height=256,
            complete_decoded=True, source_frames=1, future_listener_identity=False,
            future_listener_crop=False, predicted_crop=True, **timing)
        save(receipt, r)
        records.append(r)
        print('render', len(records), uid, timing['seconds'], flush=True)
    if not (out / 'RENDER_WEIGHT_AUDIT.json').exists():
        shutil.copyfile(WORK / 'renderer/WEIGHT_AUDIT.json', out / 'RENDER_WEIGHT_AUDIT.json')
    if source != out:
        shutil.copyfile(source / 'GENERATION_COMPLETE.json', out / 'GENERATION_COMPLETE.json')
    save(out / 'protocol.json', dict(generation_complete_sha256=sha(out / 'GENERATION_COMPLETE.json'),
        precision='FP32', batch=4, encoder='unchanged libx264 medium CRF18',
        optimization='precomputed semantic windows and bounded asynchronous pipe writes'))
    save(out / 'RENDER_COMPLETE.json', dict(success=True, outputs=len(records), records=records))


def benchmark(out, frames=132, rounds=3):
    import statistics
    import numpy as np
    import torch
    from vico_more_render import first_frame, semantics, semantic_parity
    device = gpu_guard(18)
    out.mkdir(parents=True, exist_ok=True)
    sample = test_samples()[0]
    uid = sample['id']
    rec = next(r for r in read(WORK / 'listenformer/GENERATION_COMPLETE.json')['records'] if r['id'] == uid)
    assert sha(rec['path']) == rec['sha256']
    assert sha(sample['streams']['gt']['path']) == sample['streams']['gt']['sha256']
    dyn = np.load(rec['path'])['dynamic'][:frames]
    start = time.perf_counter()
    model = model_for(out)
    load_seconds = time.perf_counter()-start
    image = first_frame(sample['streams']['gt']['path']).cuda()
    parity = semantic_parity(dyn)
    with torch.inference_mode():
        for _ in range(3):
            model(image.expand(4, -1, -1, -1), semantics(dyn, np.arange(4)).cuda())
    torch.cuda.synchronize()
    results = []
    for run in range(rounds):
        for fast in ([False, True] if run % 2 == 0 else [True, False]):
            dest = out / ('%d-%s.mp4' % (run, 'fast' if fast else 'baseline'))
            result = encode(model, image, dyn, dest, fast)
            decode_check(dest, len(dyn))
            results.append(dict(run=run, fast=fast, file_sha256=sha(dest), **result))
            print(results[-1], flush=True)
    rgb_equal = len({r['rgb_sha256'] for r in results}) == 1
    video_equal = len({r['file_sha256'] for r in results}) == 1
    baseline = statistics.median(r['seconds'] for r in results if not r['fast'])
    optimized = statistics.median(r['seconds'] for r in results if r['fast'])
    report = dict(device=device, sample=uid, frames=len(dyn), rounds=rounds,
        model_load_seconds=load_seconds, batch=4, precision='FP32', semantic_parity=parity,
        rgb_exact=rgb_equal, encoded_video_exact=video_equal, baseline_median_seconds=baseline,
        fast_median_seconds=optimized, speedup=baseline/optimized, records=results,
        limitation='Shared GPU; short clip benchmark, not a full 43-clip run')
    save(out / 'BENCHMARK.json', report)
    assert rgb_equal and video_equal, 'Output parity failed; do not use optimized rendering'
    return report


if __name__ == '__main__':
    import argparse
    from vico_inference_fast import FAST
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=FAST / 'benchmark')
    p.add_argument('--frames', type=int, default=132)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    benchmark(a.output, a.frames, a.rounds)
