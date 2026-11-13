"""C prompt, paired image8 / native video8 / native video16, frozen 50 clips."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import socket
import sys
import time
import fcntl
from contextlib import contextmanager
from pathlib import Path

ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.full_raw import open_source, views
from listen_benchmark.simple_timeline import response_schema, validate_timeline
ARMS = ['image8', 'video8', 'video16']
MODEL = '/private/ckpts/Qwen3-VL-8B-Instruct'
MM_SIZE = {'shortest_edge':65536, 'longest_edge':67108864}


@contextmanager
def decode_slot():
    """One original-video reader per host, shared by its three inference workers."""
    (RUN / 'io_slots').mkdir(exist_ok=True)
    handles = [open(RUN / 'io_slots' / f'decode-{socket.gethostname()}.lock', 'a')]
    selected = None
    try:
        while selected is None:
            for h in handles:
                try:
                    fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = h
                    break
                except BlockingIOError:
                    pass
            if selected is None:
                time.sleep(.25)
        yield
    finally:
        if selected is not None:
            fcntl.flock(selected.fileno(), fcntl.LOCK_UN)
        for h in handles:
            h.close()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def observation_prompt(duration):
    return f'''Watch this {duration:.9g}-second listener clip. Describe all directly visible movements and facial-expression changes you recognize.
For each occurrence, give its start time, end time (seconds from clip start), and a brief description of what visibly changes.
Describe physical observations only, without interpreting their psychological or communicative meaning.
Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).'''


def prepare_media(row):
    """Decode original listener once at 16fps; select the identical prior 8fps frames."""
    from PIL import Image
    from listen_labeling.media import crop_resize
    rate = row['normalized_audio']['sample_rate']
    first, last = round(row['source_start_s'] * rate), round(row['source_end_s'] * rate)
    origin, duration = first / rate, (last - first) / rate
    result = {fps: dict(images=[], audio=None, duration_s=duration, meta={}) for fps in (8, 16)}
    records = {8: [], 16: []}
    next_grid = {8: 0, 16: 0}
    last_end, previous = None, None
    with open_source(row['source']) as (container, reader):
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        assert [stream.width, stream.height] == row['video_transform']['source_size']
        source_fps = float(stream.average_rate)
        container.seek(int(origin / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            assert frame.pts is not None
            source_t = float(frame.pts * frame.time_base)
            t = source_t - origin
            if previous is not None and source_t <= previous:
                raise ValueError('Nonmonotonic original PTS')
            previous = source_t
            if t >= duration:
                break
            last_end = t + (float(frame.duration * frame.time_base) if frame.duration else 1 / source_fps)
            chosen = [fps for fps in (8, 16) if t >= 0 and t + 1e-8 >= next_grid[fps] / fps]
            if not chosen:
                continue
            image = crop_resize(frame.to_image(), row['video_transform'])
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=85)
            data = buf.getvalue()
            image = Image.open(io.BytesIO(data)).convert('RGB')
            for fps in chosen:
                result[fps]['images'].append(image)
                records[fps].append(dict(time_s=t, source_time_s=source_t, source_pts=frame.pts,
                    size=[512, 512], jpeg_sha256=hashlib.sha256(data).hexdigest()))
                next_grid[fps] = math.floor((t + 1e-8) * fps) + 1
    for fps in (8, 16):
        assert records[fps] and len(records[fps]) <= 256 and abs(last_end - duration) <= 1 / source_fps + 1e-5
        result[fps]['meta']['video'] = dict(frames=records[fps], last_frame_s=records[fps][-1]['time_s'],
            decoded_end_s=last_end, source_time_origin_s=origin, requested_fps=fps,
            video_transform=row['video_transform'], source=row['source'], reader=reader,
            source_fps=source_fps)
    assert set(r['jpeg_sha256'] for r in records[8]) <= set(r['jpeg_sha256'] for r in records[16])
    return result


def native(media):
    import numpy as np
    frames = media['meta']['video']['frames']
    source_fps = media['meta']['video']['source_fps']
    # Official metadata uses integer frame indices and original FPS. A cut need
    # not land on a source frame: record nearest-grid timestamp errors explicitly.
    indices = [round(r['time_s'] * source_fps) for r in frames]
    errors = [i / source_fps - r['time_s'] for i, r in zip(indices, frames)]
    assert indices == sorted(set(indices)) and max(abs(x) for x in errors) <= .5 / source_fps + 1e-6
    metadata = dict(fps=source_fps, total_num_frames=max(indices[-1] + 1, math.ceil(media['duration_s'] * source_fps)),
        duration=media['duration_s'], frames_indices=indices, width=512, height=512,
        video_backend='original-pyav-presampled', do_sample_frames=False)
    evidence = dict(metadata=metadata, timestamp_rounding_errors_s=errors,
        max_abs_timestamp_rounding_error_s=max(abs(x) for x in errors),
        native_temporal_patch_size=2, final_frame_padding_count=len(frames) % 2,
        do_resize=False, do_sample_frames=False, audio_input=False)
    return (np.stack([np.asarray(im) for im in media['images']]), metadata), evidence


def prepared_inputs(processor, whole, start, end, view):
    from listen_benchmark.media import timestamped_images
    from listen_benchmark.windows import slice_media
    inputs, evidence = [], []
    prompt = observation_prompt(end - start)
    for arm in ARMS:
        fps = 16 if arm == 'video16' else 8
        media = whole[fps] if view == 'full' else slice_media(whole[fps], start, end, fps)
        if arm == 'image8':
            parts = [dict(type='text', text=prompt)] + timestamped_images(media)
            payload = {'image': media['images']}
            extra = {'representation': 'timestamped-independent-images'}
        else:
            parts = [dict(type='text', text=prompt), dict(type='video', video='predecoded-listener')]
            video, extra = native(media)
            payload = {'video': [video]}
        rendered = processor.apply_chat_template([dict(role='user', content=parts)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        entry = dict(prompt=rendered, multi_modal_data=payload)
        if arm != 'image8':
            entry['mm_processor_kwargs'] = dict(do_resize=False, do_sample_frames=False)
        inputs.append(entry)
        evidence.append(dict(media=media['meta'], native=extra, prompt=prompt, rendered_prompt=rendered))
    assert evidence[0]['media'] == evidence[1]['media']
    return inputs, evidence


def init(shards):
    samples = json.load(open(RUN / 'samples.json'))
    ranks = {str(i): [] for i in range(shards)}
    costs = [0.] * shards
    for sample in sorted(samples, key=lambda s: -s['source']['duration_s']):
        rank = min(range(shards), key=lambda i: costs[i])
        ranks[str(rank)].append(sample['sample_id'])
        costs[rank] += sample['source']['duration_s'] * 2 + len(views(sample['source']['duration_s'])) * 3
    write(RUN / 'assignments.json', dict(ranks=ranks, estimated_costs=costs))
    write(RUN / 'protocol.json', dict(model=MODEL, samples=50, arms=ARMS, prompt_variant='observation_v1',
        prompt_example=observation_prompt(6), window_seconds=3, stride_seconds=1, crop_size=[512, 512],
        jpeg_quality=85, same_frames_for_image8_video8=True, video16_superset=True,
        native_video_input='original file decoded once, passed as official video tensor+metadata; not independent images',
        native_do_sample_frames=False, native_do_resize=False, native_temporal_patch_size=2,
        native_timestamp_precision='nearest source-frame index; error bounded by half source-frame period; logged per frame',
        seed=20261029, temperature=0, max_tokens=4096, max_model_len=49152, max_num_seqs=3,
        mm_size_budget=MM_SIZE, source_decode_concurrency=3, source_readers_per_host=1,
        max_num_batched_tokens=4096, dtype='bfloat16', enforce_eager=True, shards=shards,
        same_batch_all_three_arms=True, production_modified=False, human_ground_truth=False,
        audio_input=False, previous_runs_are_not_concurrent_baseline=True,
        samples_sha256=hashlib.sha256((RUN / 'samples.json').read_bytes()).hexdigest()))


def worker(rank, memory, smoke=False):
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    (RUN / 'io_slots').mkdir(exist_ok=True)
    with open(RUN / 'io_slots' / f'model-init-{socket.gethostname()}.lock', 'a') as init_lock:
        fcntl.flock(init_lock.fileno(), fcntl.LOCK_EX)
        llm = LLM(model=MODEL, dtype='bfloat16', gpu_memory_utilization=memory, max_model_len=49152,
            max_num_seqs=3, max_num_batched_tokens=4096, enable_chunked_prefill=True,
            limit_mm_per_prompt={'image':128, 'video':1}, skip_mm_profiling=True,
            mm_processor_kwargs={'size':MM_SIZE},
            enforce_eager=True, seed=20261029)
        fcntl.flock(init_lock.fileno(), fcntl.LOCK_UN)
    rows = json.load(open(RUN / 'samples.json'))
    if smoke:
        rows = [next(r for r in rows if '_S0404_' in r['sample_id'])]
    else:
        selected = set(json.load(open(RUN / 'assignments.json'))['ranks'][str(rank)])
        rows = sorted((r for r in rows if r['sample_id'] in selected), key=lambda r: -r['source']['duration_s'])
    done = 0
    for row in rows:
        sid = row['sample_id']
        path = RUN / ('smoke_results' if smoke else 'results') / (sid + '.json')
        if path.exists():
            continue
        began = time.time()
        print(sid, 'decode start', flush=True)
        with decode_slot():
            whole = prepare_media(row['source'])
        print(sid, 'decode complete', time.time()-began, flush=True)
        old = json.load(open(ROOT / 'experiments/listen_visual_observation50_20261029/results' / (sid + '.json')))
        old_frames = next(r for r in old['records'] if r['view'] == 'full')['media']['video']['frames']
        assert whole[8]['meta']['video']['frames'] == old_frames, '8fps must match historical decoded input exactly'
        partial_path = RUN / 'partial' / (sid + '.json')
        records = json.load(open(partial_path))['records'] if partial_path.exists() and not smoke else []
        completed_views = {r['view'] for r in records}
        for name in completed_views:
            assert {r['arm'] for r in records if r['view']==name} == set(ARMS)
        todo = [('w000', 0., 3.)] if smoke else views(whole[8]['duration_s'])
        for view, start, end in todo:
            if view in completed_views:
                continue
            inputs, evidence = prepared_inputs(processor, whole, start, end, view)
            params = [SamplingParams(temperature=0, max_tokens=4096, structured_outputs=StructuredOutputsParams(json=response_schema(end-start))) for _ in ARMS]
            t0 = time.time()
            responses = llm.generate(inputs, params, use_tqdm=False)
            batch_elapsed = time.time() - t0
            print(sid, view, 'batch complete', batch_elapsed, flush=True)
            for arm, ev, response in zip(ARMS, evidence, responses):
                out = response.outputs[0]
                rec = dict(arm=arm, view=view, start=start, end=end, raw=out.text,
                    finish_reason=out.finish_reason, completion_tokens=len(out.token_ids),
                    prompt_tokens=len(response.prompt_token_ids), batch_elapsed=batch_elapsed, **ev)
                try:
                    rec['parsed'] = json.loads(out.text)
                    validate_timeline(rec['parsed'], end-start)
                    assert all(e['modality'] == 'visual' for e in rec['parsed']['events'])
                    assert out.finish_reason == 'stop', 'Truncated output'
                    rec['status'] = 'valid'
                except Exception as exc:
                    rec.update(status='invalid', error=str(exc))
                records.append(rec)
            write(RUN / 'partial' / (sid + '.json'), dict(sample_id=sid, records=records))
        write(path, dict(sample_id=sid, duration_s=whole[8]['duration_s'], records=records, elapsed=time.time()-began,
            host=socket.gethostname(), rank=rank, pid=os.getpid()))
        done += 1
        write(RUN / 'workers' / f'{rank}.json', dict(completed=done, time=time.time(), pid=os.getpid(), host=socket.gethostname(), last_sample=sid))
        print(sid, 'complete', time.time()-began, flush=True)
    if not smoke:
        write(RUN / 'workers' / f'{rank}.complete.json', dict(completed=done, time=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--init', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    p.add_argument('--memory', type=float, default=.20)
    a = p.parse_args()
    if a.init:
        init(a.shards)
    else:
        worker(a.rank, a.memory, a.smoke)
