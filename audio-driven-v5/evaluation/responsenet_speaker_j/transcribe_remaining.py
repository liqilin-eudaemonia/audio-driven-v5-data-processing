"""给还没切出 pair 的 ResponseNet 样本补词时间。

只读官方分轨 wav，把 Whisper large-v3 的词时间写到 local_asr/。
不改官方 test/text。调用前设置 CUDA_VISIBLE_DEVICES，只占用指定的那一张卡。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
OUT = HERE / 'local_asr'
CACHE = Path(os.environ.get('WHISPER_CACHE', str(OUT / 'cache')))
AUDIO = Path('/public/0_DATA/2_Avatar/ResponseNet/test/audio')
MODEL = Path('/private/codes/exp/boxian_workspace/models/benchmark/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1')
SOURCE = 'whisper_large_v3_on_official_separated_wav'
TARGET_SR = 16000


def pending_jobs():
    jobs = []
    for path in sorted(RESULTS.glob('*.json')):
        result = json.loads(path.read_text())
        if result.get('pair_count'):
            continue
        sample_id = result['sample_id']
        jobs.append((sample_id, 'speaker', AUDIO / 'speaker' / f'{sample_id}.wav'))
        if not result.get('listener_words'):
            jobs.append((sample_id, 'listener', AUDIO / 'listener' / f'{sample_id}.wav'))
    return jobs


def load_wave(path):
    data, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != TARGET_SR:
        count = max(1, int(round(len(mono) * TARGET_SR / sample_rate)))
        mono = np.interp(np.linspace(0, len(mono) - 1, count), np.arange(len(mono)), mono).astype(np.float32)
        sample_rate = TARGET_SR
    duration = len(mono) / sample_rate
    hop = sample_rate // 10
    frames = len(mono) // hop
    if frames:
        rms = np.sqrt(np.mean(mono[:frames * hop].reshape(frames, hop) ** 2, axis=1))
        active = float((rms > 0.01).sum()) * 0.1
    else:
        active = 0.0
    return mono, duration, active


def words_from_output(output, duration):
    words = []
    for chunk in output.get('chunks') or []:
        text = str(chunk.get('text') or '').strip()
        stamp = chunk.get('timestamp')
        if not text or not stamp or stamp[0] is None or stamp[1] is None:
            continue
        start, end = float(stamp[0]), float(stamp[1])
        start = min(max(0.0, start), duration)
        end = min(max(0.0, end), duration)
        if end - start <= 0:
            continue
        words.append(dict(text=text, start_time=round(start, 3), end_time=round(end, 3)))
    words.sort(key=lambda word: (word['start_time'], word['end_time']))
    return words


def write_doc(path, audio, words, batch_size, skipped_silence):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(
        text=' '.join(word['text'] for word in words),
        word_time_source=SOURCE,
        model=str(MODEL),
        audio=str(audio),
        device=f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        batch_size=batch_size,
        skipped_silence=skipped_silence,
        timestamps=dict(words=words, segments=[]),
    )
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')


def cache_paths(sample_id, role):
    stem = CACHE / f'{sample_id}_{role}'
    return stem.with_suffix('.npy'), stem.with_suffix('.meta.json')


def prepare_one(item):
    sample_id, role, audio, _target = item
    wave_path, meta_path = cache_paths(sample_id, role)
    if wave_path.exists() and meta_path.exists():
        return sample_id, role, 'cached'
    wave, duration, active = load_wave(audio)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(wave_path, wave)
    meta_path.write_text(json.dumps(dict(duration=duration, active=active)))
    return sample_id, role, 'written'


def prepare(load_workers):
    jobs = pending_jobs()
    todo = [(sample_id, role, audio, OUT / sample_id / f'{role}.json') for sample_id, role, audio in jobs]
    print('prepare', len(todo), 'loaders', load_workers, flush=True)
    with ThreadPoolExecutor(max_workers=load_workers) as pool:
        for sample_id, role, state in pool.map(prepare_one, todo):
            print(state, sample_id, role, flush=True)
    return 0


def load_inputs(load_workers, batch_size):
    jobs = pending_jobs()
    todo = []
    for sample_id, role, audio in jobs:
        target = OUT / sample_id / f'{role}.json'
        if target.exists():
            print('skip', sample_id, role, flush=True)
            continue
        todo.append((sample_id, role, audio, target))
    print('todo', len(todo), 'batch_ceiling', batch_size, 'loaders', load_workers, flush=True)
    if not todo:
        return 0

    def cached(item):
        sample_id, role, audio, target = item
        wave_path, meta_path = cache_paths(sample_id, role)
        if not (wave_path.exists() and meta_path.exists()):
            prepare_one(item)
        meta = json.loads(meta_path.read_text())
        wave = np.load(wave_path)
        return sample_id, role, audio, target, wave, float(meta['duration']), float(meta['active'])

    with ThreadPoolExecutor(max_workers=load_workers) as pool:
        loaded = list(pool.map(cached, todo))

    spoken = []
    for sample_id, role, audio, target, wave, duration, active in loaded:
        if active < 0.4:
            write_doc(target, audio, [], batch_size, True)
            print('silent', sample_id, role, f'{duration:.1f}s', flush=True)
            continue
        spoken.append(dict(sample_id=sample_id, role=role, audio=audio, target=target,
                           wave=wave, duration=duration))
    print('speech', len(spoken), 'seconds', round(sum(item['duration'] for item in spoken), 1), flush=True)
    return spoken


def build_pipe():
    import torch
    import transformers.pipelines.automatic_speech_recognition as asr
    import transformers.utils as utils
    from transformers import pipeline

    utils.is_torchcodec_available = lambda: False
    asr.is_torchcodec_available = lambda: False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(16)
    pipe = pipeline(
        'automatic-speech-recognition',
        model=str(MODEL),
        device=0,
        torch_dtype=torch.float16,
        model_kwargs=dict(attn_implementation='sdpa'),
    )
    return torch, pipe


def try_batch(torch, asr_pipe, sample, size):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    asr_pipe(
        [dict(sample) for _ in range(size)],
        batch_size=size,
        num_workers=0,
        chunk_length_s=30,
        ignore_warning=True,
        return_timestamps='word',
        language='en',
        task='transcribe',
    )
    reserved = torch.cuda.max_memory_reserved()
    print('probe', 'batch', size, 'reserved_gib', round(reserved / 1024 ** 3, 1), flush=True)
    return reserved


def pick_batch(torch, asr_pipe, ceiling):
    """用 30 秒片段把 batch 加到这张卡显存占用超过约 110GB，或碰到上限。"""
    wave = np.random.default_rng(0).standard_normal(TARGET_SR * 30).astype(np.float32) * 0.05
    sample = dict(raw=wave, sampling_rate=TARGET_SR)
    size = 8
    best = 4
    while size <= ceiling:
        try:
            reserved = try_batch(torch, asr_pipe, sample, size)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print('probe_oom', size, flush=True)
            mid = best + max(1, (size - best) // 2)
            if best < mid < size:
                try:
                    reserved = try_batch(torch, asr_pipe, sample, mid)
                    best = mid
                    if reserved > 110 * 1024 ** 3:
                        break
                except RuntimeError as mid_exc:
                    if 'out of memory' not in str(mid_exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    print('probe_oom', mid, flush=True)
            break
        best = size
        if reserved > 110 * 1024 ** 3:
            break
        size *= 2
    return best


def transcribe(batch_size, load_workers):
    spoken = load_inputs(load_workers, batch_size)
    if not spoken:
        return 0
    torch, asr_pipe = build_pipe()
    fixed = os.environ.get('WHISPER_FIXED_BATCH')
    size = int(fixed) if fixed else pick_batch(torch, asr_pipe, batch_size)
    print('chosen_batch', size, flush=True)
    torch.cuda.empty_cache()

    def run(current):
        # preprocess 会 pop 掉字典里的 raw。每次重试都重新组，避免 OOM 后续跑读到空字典。
        fresh = [dict(raw=item['wave'], sampling_rate=TARGET_SR) for item in spoken]
        return asr_pipe(
            fresh,
            batch_size=current,
            num_workers=0,
            chunk_length_s=30,
            stride_length_s=5,
            ignore_warning=True,
            return_timestamps='word',
            language='en',
            task='transcribe',
        )

    while True:
        try:
            print('generate', 'batch', size, 'items', len(spoken), flush=True)
            outputs = run(size)
            break
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower() or size <= 4:
                raise
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            size = max(4, size - 4) if size > 16 else max(4, size // 2)
            print('oom_retry', size, flush=True)
    decoded = []
    long_words = 0
    kept_words = 0
    for item, output in zip(spoken, outputs):
        words = words_from_output(output, item['duration'])
        kept_words += len(words)
        long_words += sum(1 for word in words if word['end_time'] - word['start_time'] > 1.5)
        decoded.append((item, words))
    if kept_words and long_words / kept_words > 0.5:
        raise SystemExit(f'词时间不像词：{long_words}/{kept_words} 条长于 1.5 秒，结果未写入')
    for item, words in decoded:
        write_doc(item['target'], item['audio'], words, size, False)
        print('wrote', item['sample_id'], item['role'], 'words', len(words), flush=True)
    print('done', 'words', kept_words, 'batch', size, flush=True)
    return 0


def main(argv):
    batch_size = int(os.environ.get('WHISPER_BATCH', '128'))
    load_workers = int(os.environ.get('WHISPER_LOAD_WORKERS', '8'))
    if argv and argv[0] == '--list':
        jobs = pending_jobs()
        print(len(jobs))
        return 0
    if argv and argv[0] == '--prepare':
        return prepare(load_workers)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != '7':
        raise SystemExit(f'拒绝启动：CUDA_VISIBLE_DEVICES={visible!r}，这次只允许 7 号卡')
    return transcribe(batch_size, load_workers)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
