#!/usr/bin/env python3
"""Full-scale 4-node 32-GPU pipeline for J-pair natural-language captioning with:
- Quality-retained manifest (experiments/listen_j_final_quality_selection_20261101/retained.jsonl, 1,107,323 clips)
- Final reviewed bounded crop (quality-bounded-crop-v1-20261030)
- Official word/sentence/VAD union purified listener audio
- 25fps MediaPipe + FER reaction & nod detection with headshake_direction_v1 and expression_gate_v3
- v4 three-tier normalized ASR vocal cues + overlap compaction + cue-completeness retry
- Speed optimizations: FULL_DECODE_ONLY CUDA Graph, cross-source continuous prefetch,
  ready-first adaptive request groups (1..4), and persistent CPU-only FER workers.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import queue as queue_mod
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from listen_benchmark.caption_guard import (
    CAPTION_GPU_MEMORY,
    GPU_BUSY_MIB,
    INFERENCE_LIMIT_S,
    PREPARE_LIMIT_S,
    WRITE_LIMIT_S,
    caption_file,
    caption_job_eligible,
    caption_prep_workers,
    clip_identity,
    detect_worker_ceiling,
    detected_uncaptioned,
    due_prepares,
    measurement_sha256,
    partition_caption_rows,
    record_stage_timeout,
    release_after_children_exit,
    row_needs_caption,
    write_completion_status,
)
from listen_benchmark.j_annotation import (
    JobQueue,
    acquire,
    ensure_decode_fd_capacity,
    locked,
    read,
    result_path,
    write,
)
from listen_benchmark.j_omni import EvidenceUnavailable, immutable
from listen_benchmark.j_omni_crop import REVISION, result_sources
from listen_benchmark.j_omni_reaction_vocal import (
    PIPELINE_ID,
    PROMPT_VERSION,
    ReactionNotReady,
    detect_clip,
    generate_with_cue_checks,
    prepare_input,
)
from listen_j_annotation_supervisor import alive, identity, spawn


class WorkerRestart(Exception):
    """Exit this caption worker after a stage timeout so the supervisor can start a fresh process."""

PYTHON = '/private/software/conda/envs/vllm/bin/python'
DEFAULT_WORK = ROOT / 'experiments/listen_j_reaction_vocal_omni_20261103'
RETAINED_MANIFEST = (
    ROOT / 'experiments/listen_j_final_quality_selection_20261101/retained.jsonl'
)
BASE_OMNI_WORK = ROOT / 'experiments/listen_j_omni_20261030'
SHARED_ENV_ROOT = DEFAULT_WORK / 'reaction_runtime_env'


def environment(gpu=None, fer_gpu=''):
    out = dict(
        os.environ,
        LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',
        OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',
        J_OMNI_BOUNDED_CROP_V1='1',
        J_OMNI_FER_GPU=str(fer_gpu),
        J_OMNI_INPUT_OPTIMIZATIONS=json.dumps(
            {
                'fast_crop': True,
                'roi_crop': True,
                'shared_decode': True,
                'shared_raw_decode': True,
            }
        ),
    )
    if gpu is not None:
        out['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return out


def init_work(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    base_cfg = read(BASE_OMNI_WORK / 'config.json')
    fer_py = (
        work / 'reaction_runtime_env/bin/python'
        if (work / 'reaction_runtime_env/bin/python').exists()
        else SHARED_ENV_ROOT / 'bin/python'
    )
    cfg = {
        'schema': 'listen-j-reaction-vocal-omni-stream-v1',
        'pipeline_id': PIPELINE_ID,
        'prompt_version': PROMPT_VERSION,
        'project_root': str(ROOT),
        'retained_manifest': str(RETAINED_MANIFEST),
        'total_retained_clips': 1107323,
        'total_retained_pairs': 60768,
        'model': '/private/ckpts/Qwen3-Omni/Instruct',
        'fps': 8,
        'output_size': [512, 512],
        'audio_policy': 'official_word_sentence_vad_union_only',
        'new_wav_files': False,
        'training_admitted': False,
        'backlog_sources': 1024,
        'decode_slots': 112,
        'visual_quality_gate': base_cfg['visual_quality_gate'],
        'bounded_crop_input': {
            'revision': REVISION,
            'runtime_python': str(
                BASE_OMNI_WORK / 'nod_cpu_share/runtime_env/bin/python'
            ),
            'nod_cache': str(work / 'reaction_cache'),
            'nod_slots_per_host': 32,
            'source_record_policy': 'source_is_actual_original_source_is_J',
        },
        'reaction_input': {
            'revision': 'glare-nod-shake-frown-surprise-gate-v3-20261103',
            'runtime_python': str(
                BASE_OMNI_WORK / 'nod_cpu_share/runtime_env/bin/python'
            ),
            'fer_python': str(fer_py),
            'reaction_cache': str(work / 'reaction_cache'),
            'reaction_slots_per_host': 32,
        },
        'nodes': base_cfg['nodes'],
        'preparation_profiles': base_cfg['preparation_profiles'],
    }
    for node_cfg in cfg['nodes'].values():
        res_dir = Path(node_cfg['resource_work'])
        res_dir.mkdir(parents=True, exist_ok=True)
        sel = res_dir / 'selected.json'
        if not sel.exists():
            write(sel, {'decode_slots': 112})
    write(work / 'config.json', cfg)
    JobQueue(work, 'caption')
    return cfg


def _prepare_worker_task(row, old_work, work, resources, fer_gpu=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['J_OMNI_FER_GPU'] = ''
    os.environ['J_OMNI_CAPTION_NO_MEASURE'] = '1'
    t0 = time.monotonic()
    out = prepare_input(row, old_work, work, resources)
    out['_preparation_metrics'] = {
        'seconds': round(time.monotonic() - t0, 4),
        'pid': os.getpid(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'fer_gpu': str(fer_gpu),
    }
    return out


def producer_step(work: Path, limit_pairs: int = 128) -> dict:
    work = Path(work)
    if not (work / 'config.json').exists():
        init_work(work)
    cfg = read(work / 'config.json')
    with locked(work / 'producer-step.lock'):
        state_path = work / 'producer_state.json'
        state = (
            read(state_path)
            if state_path.exists()
            else {
                'byte_offset': 0,
                'carry_pair': None,
                'carry_rows': [],
                'published_sources': 0,
                'published_clips': 0,
                'eof': False,
            }
        )
        queue_branch = 'detect' if (work / 'DETECT_ONLY').exists() else 'caption'
        queue = JobQueue(work, queue_branch)
        qstate = read(queue.path)
        if state['published_sources'] - qstate['completed'] >= cfg.get('backlog_sources', 1024):
            return dict(state, stage='backpressure', at=time.time())

        manifest_path = Path(cfg['retained_manifest'])
        published_this_step = 0

        def flush_pair(pair_id: str, items: list[dict]):
            nonlocal published_this_step
            rows = []
            for r in items:
                if r['quality']['decision'] != 'retain':
                    raise ValueError(f"Non-retain row in retained manifest: {r['sample_id']}")
                cap = read(r['caption_file'])
                orig_source = cap.get('original_source') or cap['source']
                rows.append(
                    {
                        'sample_id': r['sample_id'],
                        'split': r['split'],
                        'duration_s': r['duration_s'],
                        'source': orig_source,
                        'quality': r['quality'],
                        'prior_caption_file': r['caption_file'],
                    }
                )
            job_path = work / 'jobs' / f'{pair_id}.json'
            if not job_path.exists():
                write(
                    job_path,
                    {
                        'job_id': pair_id,
                        'pair': pair_id,
                        'count': len(rows),
                        'rows': rows,
                    },
                )
            entry = {'job_id': pair_id, 'count': len(rows), 'at': time.time()}
            with (work / 'ready.jsonl').open('a') as rf:
                rf.write(json.dumps(entry) + '\n')
                rf.flush()
                os.fsync(rf.fileno())
            state['published_sources'] += 1
            state['published_clips'] += len(rows)
            published_this_step += 1

        with manifest_path.open('rb') as handle:
            handle.seek(state['byte_offset'])
            while published_this_step < limit_pairs:
                line = handle.readline()
                if not line:
                    if state['carry_pair'] and state['carry_rows']:
                        flush_pair(state['carry_pair'], state['carry_rows'])
                        state['carry_pair'] = None
                        state['carry_rows'] = []
                    state['byte_offset'] = handle.tell()
                    state['eof'] = True
                    break
                row = json.loads(line)
                pair = row['pair']
                if state['carry_pair'] is None:
                    state['carry_pair'] = pair
                    state['carry_rows'] = [row]
                elif pair == state['carry_pair']:
                    state['carry_rows'].append(row)
                else:
                    flush_pair(state['carry_pair'], state['carry_rows'])
                    state['carry_pair'] = pair
                    state['carry_rows'] = [row]
                state['byte_offset'] = handle.tell()

        state.update(at=time.time(), stage='complete' if state['eof'] else 'feeding')
        write(state_path, state)
        if state['eof']:
            write(work / 'producer_complete.json', state)
        return state


def producer(work: Path):
    work = Path(work)
    owner = acquire(work / 'producer.lock', False)
    if owner is None:
        return
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    while not stopped.is_set():
        try:
            s = producer_step(work)
            write(work / 'producer.json', dict(s, pid=os.getpid(), error=None))
            if s.get('eof'):
                break
        except Exception as exc:
            write(work / 'producer.json', {'pid': os.getpid(), 'at': time.time(), 'error': repr(exc)})
        stopped.wait(2)
    owner.close()


class CrossPairPrefetchStream:
    """Continuously prefetch across up to `max_sources` pairs in ready-first order."""

    def __init__(
        self,
        work: Path,
        q: JobQueue,
        name: str,
        pool: ProcessPoolExecutor,
        old_work: str,
        resources: str,
        fer_gpu: str,
        prefetch_capacity: int,
        max_sources: int,
        stop: threading.Event,
    ):
        self.work = work
        self.q = q
        self.name = name
        self.pool = pool
        self.old_work = old_work
        self.resources = resources
        self.fer_gpu = fer_gpu
        self.capacity = prefetch_capacity
        self.max_sources = max_sources
        self.stop = stop
        self.active = {}
        self.pending = {}
        self.seq = 0

    def _claim_more(self) -> bool:
        if self.stop.is_set() or len(self.active) >= self.max_sources:
            return False
        claim = self.q.claim(self.name, eligible=lambda job_id: caption_job_eligible(self.work, job_id))
        if claim is None:
            return False
        jid, handle = claim
        job = read(self.work / 'jobs' / f'{jid}.json')
        ready, awaiting = partition_caption_rows(self.work, job['rows'])
        ready.sort(key=lambda r: -r['source']['duration_s'])
        if not ready and awaiting:
            self.q.defer(jid)
            handle.close()
            return True
        self.active[jid] = {
            'handle': handle,
            'job': job,
            'rows': deque(ready),
            'waiting': {r['sample_id'] for r in ready},
            'awaiting_detection': [r['sample_id'] for r in awaiting],
        }
        self._retire_finished()
        return True

    def _retire_finished(self):
        for jid, item in list(self.active.items()):
            if item['rows'] or item['waiting']:
                continue
            if item.get('awaiting_detection'):
                self.q.defer(jid, reason='awaiting_detection')
                item['handle'].close()
                del self.active[jid]
                continue
            rows = item['job']['rows']
            if any(row_needs_caption(self.work, r) for r in rows):
                continue
            cap_n = sum(1 for r in rows if caption_file(self.work, r['source'], measurement_sha256(self.work, r['sample_id']), r['sample_id']).exists())
            held_n = sum(1 for r in rows if result_path(self.work, 'held', r['sample_id']).exists())
            bad_n = 0
            for r in rows:
                path = caption_file(self.work, r['source'], measurement_sha256(self.work, r['sample_id']), r['sample_id'])
                if path.exists():
                    bad_n += not read(path).get('generation_checks_passed', True)
            self.q.finish(
                jid,
                receipt={
                    'job_id': jid,
                    'clips': len(rows),
                    'caption_clips': cap_n,
                    'failed_generation_clips': bad_n,
                    'held_clips': held_n,
                    'completed_at': time.time(),
                },
            )
            item['handle'].close()
            del self.active[jid]

    def _fill_pool(self):
        while not self.stop.is_set() and len(self.pending) < self.capacity:
            while len(self.active) < self.max_sources and any(
                not v['rows'] for v in self.active.values()
            ) or not self.active:
                if not self._claim_more():
                    break
            submitted = False
            for jid, item in list(self.active.items()):
                if not item['rows']:
                    continue
                row = item['rows'].popleft()
                fut = self.pool.submit(
                    _prepare_worker_task,
                    row,
                    self.old_work,
                    str(self.work),
                    self.resources,
                    self.fer_gpu,
                )
                self.seq += 1
                self.pending[fut] = {'jid': jid, 'row': row, 'seq': self.seq, 'submitted_at': time.monotonic()}
                submitted = True
                if len(self.pending) >= self.capacity:
                    break
            if not submitted and not self._claim_more():
                break

    def note_awaiting(self, jid, sid):
        item = self.active.get(jid)
        if item is not None and sid not in item['awaiting_detection']:
            item['awaiting_detection'].append(sid)

    def mark_done(self, jid: str, sid: str):
        item = self.active.get(jid)
        if item is not None:
            item['waiting'].discard(sid)
            # Only the prefetch reader retires pairs. The writer previously
            # retired the same pair concurrently, double-closing its claim and
            # raising KeyError after a valid result had already been written.

    def close(self):
        """Stop new claims, let in-flight prepares finish or be cancelled, then release locks."""

        def stop_reader():
            self.stop.set()

        def settle_pending():
            running = []
            for fut in list(self.pending):
                if not fut.cancel():
                    running.append(fut)
            for fut in running:
                try:
                    fut.result(timeout=PREPARE_LIMIT_S)
                except Exception:
                    pass
            self.pending.clear()

        def release_claims():
            for item in self.active.values():
                try:
                    item['handle'].close()
                except Exception:
                    pass
            self.active.clear()

        from listen_benchmark.caption_guard import ordered_prefetch_shutdown

        ordered_prefetch_shutdown(stop_reader, settle_pending, release_claims)

    def __iter__(self):
        while not self.stop.is_set():
            self._retire_finished()
            self._fill_pool()
            if not self.pending:
                if self.active:
                    self._retire_finished()
                if not self.active and (self.work / 'producer_complete.json').exists():
                    qs = read(self.q.path)
                    if not qs['inflight'] and not qs.get('deferred'):
                        break
                self.stop.wait(1.0)
                continue
            wait_t0 = time.monotonic()
            done, _ = wait(list(self.pending), timeout=0.5, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            expired = [
                fut for fut in due_prepares(self.pending, now, PREPARE_LIMIT_S) if fut not in done
            ]
            ready = list(done) + expired
            if not ready:
                continue
            wait_s = now - wait_t0
            ordered = sorted(ready, key=lambda f: self.pending[f]['seq'])
            for idx, fut in enumerate(ordered):
                meta = self.pending.pop(fut)
                self._fill_pool()
                yield meta['jid'], meta['row'], fut, (wait_s if idx == 0 else 0.0), meta['submitted_at']


def worker(
    work: Path,
    node: str,
    gpu: int,
    max_clips: int | None = None,
    fer_gpu: str | None = None,
    workers_override: int | None = None,
    prefetch_override: int | None = None,
    batch_size: int = 4,
    eager: bool = False,
):
    from vllm import LLM
    from transformers import AutoProcessor

    work = Path(work)
    cfg = read(work / 'config.json')
    nc = cfg['nodes'][node]
    if socket.gethostname() != nc['hostname']:
        raise ValueError(f"Host mismatch: {socket.gethostname()} != {nc['hostname']}")
    name = f'{node}-gpu-{gpu}'
    owner = acquire(work / 'workers' / f'{name}.lock', False)
    if owner is None:
        raise RuntimeError(f'Worker {name} already running')

    override_path = work / 'nodes' / node / 'caption_overrides.json'
    card_profile = (read(override_path).get(str(gpu), {}) if override_path.exists() else {})
    base_prof = {'workers': 2, 'prefetch': 4}
    num_workers = card_profile.get('workers') or workers_override or base_prof['workers']
    prefetch_cap = card_profile.get('prefetch') or prefetch_override or base_prof['prefetch']
    actual_fer_gpu = '' if fer_gpu is None else str(fer_gpu)

    ensure_decode_fd_capacity(num_workers, cfg['decode_slots'])
    resources = nc['resource_work']
    old_work = nc.get('old_work') or str(
        ROOT / 'experiments/listen_j_annotation_node17_20261029'
    )

    state = {
        'pid': os.getpid(),
        'node': node,
        'gpu': gpu,
        'start_ticks': identity(os.getpid()),
        'fer_gpu': actual_fer_gpu,
        'prep_workers': num_workers,
        'prefetch': prefetch_cap,
        'production_batch_size': batch_size,
        'scheduler_revision': 'reader-owned-retirement-v1',
        'cudagraph': not eager,
        'stage': 'loading_model',
        'started_at': time.time(),
        'completed_clips': 0,
        'failed_clips': 0,
        'held_clips': 0,
        'preparation_wait_s': 0.0,
        'input_idle_s': 0.0,
        'inference_s': 0.0,
    }

    def status():
        write(work / 'workers' / f'{name}.json', dict(state, heartbeat=time.time()))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    status()

    mem_frac = float(os.environ.get('VLLM_GPU_MEM', nc.get('memory', 0.80)))
    processor = AutoProcessor.from_pretrained(
        cfg['model'], trust_remote_code=True, local_files_only=True
    )
    llm_kwargs = {
        'model': cfg['model'],
        'trust_remote_code': True,
        'dtype': 'bfloat16',
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': mem_frac,
        'max_model_len': 49152,
        'max_num_seqs': 1,
        'max_num_batched_tokens': 49152,
        'enable_chunked_prefill': False,
        'limit_mm_per_prompt': {'image': 128, 'audio': 1, 'video': 1},
        'skip_mm_profiling': True,
        'enforce_eager': eager,
    }
    if not eager:
        llm_kwargs['compilation_config'] = {
            'mode': 0,
            'cudagraph_mode': 'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes': [1],
            'max_cudagraph_capture_size': 1,
        }
    llm = LLM(**llm_kwargs)

    q = JobQueue(work, 'caption')
    pool = ProcessPoolExecutor(max_workers=num_workers)
    stream = CrossPairPrefetchStream(
        work=work,
        q=q,
        name=name,
        pool=pool,
        old_work=old_work,
        resources=resources,
        fer_gpu=actual_fer_gpu,
        prefetch_capacity=prefetch_cap,
        max_sources=4,
        stop=stop,
    )

    mailbox = queue_mod.Queue(maxsize=max(1, batch_size))
    done_sentinel = object()

    def reader_loop():
        try:
            for item in stream:
                if stop.is_set():
                    break
                while not stop.is_set():
                    try:
                        mailbox.put(('item', item), timeout=0.1)
                        break
                    except queue_mod.Full:
                        pass
        except BaseException as exc:
            mailbox.put(('error', exc))
        finally:
            mailbox.put(('done', done_sentinel))

    reader_thread = threading.Thread(target=reader_loop, daemon=True)
    reader_thread.start()

    restarting = False
    try:
        ended = False
        while not ended and not stop.is_set():
            prep_path = work / 'nodes' / node / 'caption_prep.json'
            if prep_path.exists():
                card_profile = (read(override_path).get(str(gpu), {}) if override_path.exists() else {})
                wanted = int(card_profile.get('workers') or read(prep_path).get('workers') or num_workers)
                if wanted != num_workers:
                    state['prep_resize'] = {'from': num_workers, 'to': wanted, 'at': time.time()}
                    status()
                    stop.set()
                    break
            idle_started = time.monotonic()
            try:
                kind, value = mailbox.get(timeout=0.2)
            except queue_mod.Empty:
                state['input_idle_s'] += time.monotonic() - idle_started
                continue
            state['input_idle_s'] += time.monotonic() - idle_started
            if kind == 'error':
                raise value
            if kind == 'done':
                break
            raw_batch = [value]
            while len(raw_batch) < batch_size:
                try:
                    k2, v2 = mailbox.get_nowait()
                except queue_mod.Empty:
                    break
                if k2 == 'done':
                    ended = True
                    break
                if k2 == 'error':
                    raise v2
                raw_batch.append(v2)

            records = []
            for jid, row, fut, w_s, submitted_at in raw_batch:
                sid = row['sample_id']
                state.update(stage='prepare', sample_id=sid, stage_started_at=time.time())
                status()
                t0 = time.monotonic()
                remain = PREPARE_LIMIT_S - (time.monotonic() - submitted_at)
                try:
                    if remain <= 0 and not fut.done():
                        raise TimeoutError('prepare exceeded its limit before the result was collected')
                    prepared = fut.result(timeout=max(0., remain))
                except ReactionNotReady:
                    state['stage_finished_at'] = time.time()
                    stream.note_awaiting(jid, sid)
                    stream.mark_done(jid, sid)
                    continue
                except EvidenceUnavailable as exc:
                    write(
                        result_path(work, 'held', sid),
                        {
                            'sample_id': sid,
                            'source': row['source'],
                            'reason': repr(exc),
                            'at': time.time(),
                            'training_admitted': False,
                        },
                    )
                    state['held_clips'] += 1
                    state['stage_finished_at'] = time.time()
                    stream.mark_done(jid, sid)
                    continue
                except TimeoutError as exc:
                    state['stage_finished_at'] = time.time()
                    record_stage_timeout(
                        work,
                        sid,
                        'prepare',
                        {'error': repr(exc), 'limit_s': PREPARE_LIMIT_S},
                    )
                    raise WorkerRestart
                total_wait = w_s + (time.monotonic() - t0)
                state['stage_finished_at'] = time.time()
                state['preparation_wait_s'] += total_wait
                records.append({'jid': jid, 'row': row, 'prepared': prepared, 'wait_s': total_wait})

            if not records or stop.is_set():
                continue

            state.update(
                stage='inference',
                sample_id=records[0]['row']['sample_id'],
                job=records[0]['jid'],
                submitted_requests=len(records),
                stage_started_at=time.time(),
            )
            status()
            inf_t0 = time.monotonic()
            inferred = {}

            def _infer():
                try:
                    inferred['out'] = generate_with_cue_checks(llm, processor, records, cfg)
                except BaseException as exc:
                    inferred['error'] = exc

            infer_thread = threading.Thread(target=_infer, daemon=True)
            infer_thread.start()
            infer_thread.join(INFERENCE_LIMIT_S * max(1, len(records)))
            if infer_thread.is_alive():
                state['stage_finished_at'] = time.time()
                record_stage_timeout(
                    work,
                    records[0]['row']['sample_id'],
                    'inference',
                    {'limit_s': INFERENCE_LIMIT_S, 'batch': len(records)},
                )
                raise WorkerRestart
            if 'error' in inferred:
                raise inferred['error']
            generated = inferred['out']
            state['stage_finished_at'] = time.time()
            inf_elapsed = time.monotonic() - inf_t0
            state['inference_s'] += inf_elapsed

            for rec, gen in zip(records, generated):
                row = rec['row']
                prepared = rec['prepared']
                sid = row['sample_id']
                best = gen['best']
                sha = ((prepared.get('evidence') or {}).get('nod_supplement') or {}).get('detector_sha256')
                sha = sha or measurement_sha256(work, sid)
                dest = caption_file(work, row['source'], sha, sid)
                sources = result_sources(row, prepared)
                record = {
                    'sample_id': sid,
                    **sources,
                    'split': row.get('split'),
                    'pipeline_id': cfg['pipeline_id'],
                    'prompt_version': cfg['prompt_version'],
                    'prompt': best['prompt'],
                    'rendered_prompt': best['rendered_prompt'],
                    'caption': best['caption'],
                    'finish_reason': best['finish_reason'],
                    'generation_checks_passed': not best['failures'],
                    'review_flags': list(best['failures']),
                    'remaining_gaps': best['gaps'],
                    'completion_tokens': best['completion_tokens'],
                    'attempts': gen['attempts'],
                    'input_evidence': prepared['evidence'],
                    'audio_recipe': prepared['recipe'],
                    'seconds': round(inf_elapsed / len(records), 4),
                    'group_seconds': round(inf_elapsed, 4),
                    'group_size': len(records),
                    'node': node,
                    'gpu': gpu,
                    'preparation_metrics': dict(
                        prepared['_preparation_metrics'], main_wait_s=round(rec['wait_s'], 4)
                    ),
                    'completed_at': time.time(),
                    'training_admitted': False,
                    'human_ground_truth': False,
                    'cache_identity': dict(
                        clip_identity(row['source'], sha),
                        final_crop=(prepared.get('effective_source') or row['source']).get('video_transform'),
                    ),
                }
                state.update(stage='write', sample_id=sid, stage_started_at=time.time())
                status()
                written = {}

                def _write():
                    try:
                        immutable(dest, record)
                        written['ok'] = True
                    except BaseException as exc:
                        written['error'] = exc

                write_thread = threading.Thread(target=_write, daemon=True)
                write_thread.start()
                write_thread.join(WRITE_LIMIT_S)
                state['stage_finished_at'] = time.time()
                if write_thread.is_alive():
                    record_stage_timeout(work, sid, 'write', {'limit_s': WRITE_LIMIT_S, 'path': str(dest)})
                    raise WorkerRestart
                if 'error' in written:
                    raise written['error']
                stream.mark_done(rec['jid'], sid)
                state['completed_clips'] += 1
                state['failed_clips'] += bool(best['failures'])
                state['last_progress_at'] = time.time()
                status()
                if max_clips and state['completed_clips'] >= max_clips:
                    stop.set()
                    break
    except WorkerRestart:
        restarting = True
    finally:
        stop.set()
        reader_thread.join(timeout=5)
        if restarting:
            prepare_processes = list(getattr(pool, '_processes', {}).values())
            pool.shutdown(wait=False, cancel_futures=True)

            def release_claims():
                for item in list(stream.active.values()):
                    try:
                        item['handle'].close()
                    except Exception:
                        pass
                stream.active.clear()
                stream.pending.clear()

            still = release_after_children_exit(prepare_processes, release_claims)
            state['prepare_workers_still_alive'] = still
        else:
            stream.close()
            pool.shutdown(wait=False, cancel_futures=True)
        state['stage'] = 'restarting' if restarting else 'stopped'
        status()
        owner.close()
    if restarting:
        os._exit(75)


def _manifest_rows(work: Path):
    rows = []
    folder = Path(work) / 'jobs'
    if folder.exists():
        for path in sorted(folder.glob('*.json')):
            rows.extend(read(path).get('rows') or [])
    return rows


def complete(work: Path) -> bool:
    """Processing has drained. The status file says whether every sample succeeded."""
    if not (work / 'producer_complete.json').exists():
        return False
    q = JobQueue(work, 'caption')
    s = read(q.path)
    drained = (
        not s['inflight']
        and not s.get('deferred')
        and (work / 'ready.jsonl').exists()
        and s['offset'] >= (work / 'ready.jsonl').stat().st_size
    )
    if not drained:
        return False
    write_completion_status(work)
    return True


def stalled_caption_worker(child, record, now, backlog):
    """A live process can still stall; do not confuse idle supply with a hang."""
    if backlog <= 0:
        return None
    if record.get('pid') != child.get('pid'):
        # A stale/mismatched status is an observation failure, not proof of a
        # stalled worker. Never recycle a GPU solely from its launch timestamp.
        return None
    stage = record.get('stage', 'loading_model')
    limit = {
        'loading_model': 900,
        'prepare': PREPARE_LIMIT_S + 90,
        'inference': INFERENCE_LIMIT_S * min(4, max(1, int(record.get('submitted_requests') or 1))) + 90,
        'write': WRITE_LIMIT_S + 90,
        'stopped': 240,
        'restarting': 240,
    }.get(stage, 900)
    age = now - float(record.get('heartbeat') or child.get('started_at') or now)
    if age > limit:
        return {'stage': stage, 'heartbeat_age_s': age, 'limit_s': limit}
    return None


def supervise(work: Path, node: str):
    cfg = read(work / 'config.json')
    nc = cfg['nodes'][node]
    if socket.gethostname() != nc['hostname']:
        raise ValueError('Supervisor host mismatch')
    nodework = work / 'nodes' / node
    owner = acquire(nodework / 'supervisor.lock', False)
    if owner is None:
        return
    sf = nodework / 'supervisor.json'
    state = read(sf) if sf.exists() else {'children': {}, 'restarts': {}}
    state.update(
        pid=os.getpid(),
        start_ticks=identity(os.getpid()),
        started_at=time.time(),
        node=node,
        stage='initializing',
        heartbeat=time.time(),
    )
    # Publish our live identity before an audit can wait on another host. The
    # watchdog must not mistake that wait for a missing supervisor and spawn
    # more auditors. Each caption identity needs only one cluster-wide audit.
    write(sf, state)
    from listen_benchmark.caption_guard import digest, shared_identity, republish_caption_jobs
    audit_marker = work / f'caption-republished-{digest(shared_identity())[:16]}.json'
    if not audit_marker.exists():
        with locked(work / 'caption-republish.lock'):
            if not audit_marker.exists():
                republished = republish_caption_jobs(work)
                write(audit_marker, {'at': time.time(), 'republished_jobs': len(republished)})
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    prep_samples = deque()
    while not stop:
        try:
            if complete(work):
                state['stage'] = read(work / 'completion_status.json').get('status', 'completed_with_failures')
                break
            raw = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                text=True,
                timeout=20,
            )
            devices = {
                int(x.split(',')[0]): list(map(float, x.split(',')[1:]))
                for x in raw.strip().splitlines()
            }
            producer_state = (
                read(work / 'producer.json') if (work / 'producer.json').exists() else {}
            )
            if (
                not (work / 'DETECT_ONLY').exists()
                and not (work / 'producer_complete.json').exists()
                and time.time() - producer_state.get('at', 0) > 45
            ):
                child = state['children'].get('producer')
                if not alive(child):
                    state['children']['producer'] = spawn(
                        nodework,
                        'producer',
                        [PYTHON, str(Path(__file__).resolve()), 'producer', '--work', str(work)],
                        environment(),
                    )
            wait_s = infer_s = 0.0
            worker_dir = work / 'workers'
            if worker_dir.exists():
                for path in worker_dir.glob(f'{node}-gpu-*.json'):
                    rec = read(path)
                    wait_s += float(rec.get('input_idle_s', rec.get('preparation_wait_s')) or 0)
                    infer_s += float(rec.get('inference_s') or 0)
            # Recent five-minute deltas, not lifetime totals. A cold start must
            # not keep every later worker at three preparers indefinitely.
            now = time.monotonic()
            prep_samples.append((now, wait_s, infer_s))
            while len(prep_samples) > 2 and prep_samples[1][0] < now - 300:
                prep_samples.popleft()
            base = prep_samples[0]
            recent_wait = max(0., wait_s - base[1])
            recent_infer = max(0., infer_s - base[2])
            backlog = detected_uncaptioned(work)
            prep_n = caption_prep_workers(recent_wait, recent_infer, backlog)
            profile_path = nodework / 'caption_prep.json'
            previous = read(profile_path) if profile_path.exists() else {}
            changed_at = previous.get('changed_at', time.time())
            previous_n = previous.get('workers', 2)
            # Preserve a successful expansion across worker counter resets.
            # Supply pressure can always reclaim the third preparer; expansions
            # wait ten minutes to avoid repeated model reloads near the threshold.
            if backlog >= 25000:
                if previous_n == 3 or time.time() - changed_at < 600:
                    prep_n = previous_n
            if prep_n != previous_n:
                changed_at = time.time()
            write(
                profile_path,
                {
                    'workers': prep_n,
                    'changed_at': changed_at,
                    'prefetch': prep_n * 2,
                    'batch_size': 4,
                    'gpu_memory': CAPTION_GPU_MEMORY,
                    'backlog': backlog,
                    'backlog_is_receipt_estimate': True,
                    'recent_wait_s': recent_wait,
                    'recent_inference_s': recent_infer,
                    'at': time.time(),
                },
            )
            for gpu in nc['gpu_ids']:
                key = f'gpu-{gpu}'
                child = state['children'].get(key)
                if child and child.get('stalled_stop_at') and time.time() - child['stalled_stop_at'] > 60:
                    # The recorded process group was created solely for this
                    # worker. Reap its engine/preparers even if the parent died.
                    try:
                        os.killpg(child['stalled_group'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.pop('stalled_stop_at', None)
                if alive(child) and not child.get('stalled_stop_at'):
                    worker_path = work / 'workers' / f'{node}-{key}.json'
                    rec = read(worker_path) if worker_path.exists() else {}
                    problem = stalled_caption_worker(child, rec, time.time(), backlog)
                    if problem and os.getpgid(child['pid']) == child['pid']:
                        write(nodework / 'stall_diagnostics' / f'{key}-{int(time.time())}.json',
                              dict(worker=child, problem=problem, at=time.time()))
                        child['stalled_stop_at'] = time.time()
                        child['stalled_group'] = child['pid']
                        os.killpg(child['pid'], signal.SIGTERM)
                if child and child.get('stalled_stop_at'):
                    continue
                if not alive(child):
                    if devices[gpu][0] > GPU_BUSY_MIB:
                        continue
                    restarts = [
                        t for t in state['restarts'].get(key, []) if time.time() - t < 3600
                    ]
                    if len(restarts) >= 4:
                        state.setdefault('blocked_gpus', {})[str(gpu)] = (
                            'four exits in one hour; inspect logs'
                        )
                        continue
                    env = environment(gpu, fer_gpu='')
                    env['VLLM_GPU_MEM'] = CAPTION_GPU_MEMORY
                    env['J_OMNI_CAPTION_NO_MEASURE'] = '1'
                    env['J_OMNI_FER_GPU'] = ''
                    state['children'][key] = spawn(
                        nodework,
                        key,
                        [
                            PYTHON,
                            str(Path(__file__).resolve()),
                            'worker',
                            '--work',
                            str(work),
                            '--node',
                            node,
                            '--gpu',
                            str(gpu),
                            '--workers',
                            str(prep_n),
                            '--prefetch',
                            str(prep_n * 2),
                            '--batch-size',
                            '4',
                        ],
                        env,
                    )
                    state['restarts'][key] = restarts + [time.time()]
            q = JobQueue(work, 'caption')
            qs = read(q.path)
            state.update(
                stage='running',
                devices=devices,
                queue={
                    'resolved_clips': qs['completed_clips'],
                    'completed_sources': qs['completed'],
                    'failed_sources': len(qs['failed']),
                    'inflight_sources': len(qs['inflight']),
                },
                error=None,
            )
        except Exception as exc:
            state['error'] = repr(exc)
        state['heartbeat'] = time.time()
        write(sf, state)
        time.sleep(15)
    state.update(stopped_at=time.time(), heartbeat=time.time())
    write(sf, state)
    owner.close()


def watchdog(work: Path, node: str):
    nodework = work / 'nodes' / node
    owner = acquire(nodework / 'watchdog.lock', False)
    if owner is None:
        return
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    state = {
        'pid': os.getpid(),
        'start_ticks': identity(os.getpid()),
        'started_at': time.time(),
        'restarts': [],
    }
    while not stop:
        try:
            if (nodework / 'STOP_SUPERVISION').exists() or complete(work):
                break
            f = nodework / 'supervisor.json'
            s = read(f) if f.exists() else None
            if not alive(s):
                state['restarts'].append(
                    spawn(
                        nodework,
                        'supervisor',
                        [
                            PYTHON,
                            str(Path(__file__).resolve()),
                            'supervise',
                            '--work',
                            str(work),
                            '--node',
                            node,
                        ],
                        environment(),
                    )
                )
                state['restarts'] = state['restarts'][-20:]
            state['error'] = None
        except Exception as exc:
            state['error'] = repr(exc)
        state['heartbeat'] = time.time()
        write(nodework / 'watchdog.json', state)
        time.sleep(15)
    state['stopped_at'] = time.time()
    write(nodework / 'watchdog.json', state)
    owner.close()


DETECT_SLOT_CAP = 128


def _allowed_cores() -> int:
    """CPUs this cgroup can actually run. os.cpu_count() includes CPUs outside the quota."""
    allowed = len(os.sched_getaffinity(0))
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        parts = quota.read_text().split()
        if len(parts) == 2 and parts[0] != 'max':
            allowed = min(allowed, max(1, int(parts[0]) // int(parts[1])))
    return max(1, allowed)


def _cpu_usage_sample():
    """Observe system busy share and this container's actual consumed cores."""

    def snap():
        parts = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
        idle = parts[3] + parts[4]
        return idle, sum(parts)

    def usage():
        try:
            values = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
            return int(values['usage_usec'])
        except (OSError, KeyError, ValueError):
            return None

    before = snap()
    group_before = usage()
    started = time.monotonic()
    time.sleep(0.4)
    after = snap()
    group_after = usage()
    elapsed = time.monotonic() - started
    delta = max(1, after[1] - before[1])
    consumed = None
    if group_before is not None and group_after is not None:
        consumed = max(0, group_after - group_before) / 1e6 / elapsed
    return 1.0 - (after[0] - before[0]) / delta, consumed


def _cpu_busy_fraction() -> float:
    return _cpu_usage_sample()[0]


def detection_worker_target(active, cores, busy, load, mem, ceiling):
    """Validated legacy control. Higher detector concurrency reduced throughput
    in the live quota-based pilot; cgroup usage remains diagnostic only.
    """
    desired = active
    if mem >= 40 and busy < 0.78:
        desired += min(8, max(4, int((0.83 - busy) * cores / 2)))
    elif load > cores * 1.08 and busy > 0.80:
        desired = max(8, desired - max(2, int((load - cores) / 2)))
    return max(1, min(DETECT_SLOT_CAP, ceiling, desired))


def _mem_available_gb() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _detect_resources(work: Path, node: str, slots: int) -> Path:
    directory = work / 'detect_resources' / node
    directory.mkdir(parents=True, exist_ok=True)
    selected = directory / 'selected.json'
    current = read(selected) if selected.exists() else {}
    if current.get('decode_slots') != slots:
        write(selected, {'decode_slots': slots})
    return directory


def detect_worker(work: Path, node: str, slot: int):
    """Claim retained pairs and run the five CPU detectors. Never touches a GPU."""
    cfg = read(work / 'config.json')
    if socket.gethostname() != cfg['nodes'][node]['hostname']:
        raise ValueError('Detect worker host mismatch')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['J_OMNI_FER_GPU'] = ''
    os.environ['J_OMNI_BOUNDED_CROP_V1'] = '1'
    os.environ['J_OMNI_REACTION_SLOTS'] = str(DETECT_SLOT_CAP)
    name = f'{node}-detect-{slot}'
    nodework = work / 'nodes' / node
    stop_file = nodework / 'detect_stop' / name
    resources = _detect_resources(work, node, DETECT_SLOT_CAP)
    queue = JobQueue(work, 'detect')
    owner = acquire(nodework / 'detect_workers' / f'{name}.lock', False)
    if owner is None:
        return
    state = {
        'pid': os.getpid(),
        'slot': slot,
        'node': node,
        'started_at': time.time(),
        'clips': 0,
        'pairs': 0,
        'errors': 0,
    }
    heartbeat = nodework / 'detect_heartbeats' / f'{name}.json'

    def status(**extra):
        state.update(heartbeat_at=time.time(), **extra)
        write(heartbeat, state)

    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    status(stage='running')
    try:
        while not stop and not stop_file.exists():
            claim = queue.claim(name)
            if claim is None:
                if (work / 'producer_complete.json').exists():
                    break
                status(stage='waiting')
                time.sleep(2)
                continue
            job_id, handle = claim
            try:
                job = read(work / 'jobs' / f'{job_id}.json')
                pending = list(job['rows'])
                errors = []
                for _attempt in range(3):
                    if stop or stop_file.exists() or not pending:
                        break
                    failed = []
                    for row in pending:
                        if stop or stop_file.exists():
                            failed.append(row)
                            continue
                        t0 = time.monotonic()
                        try:
                            summary = detect_clip(row, work, resources)
                            state['clips'] += 1
                            state['last'] = dict(summary, seconds=round(time.monotonic() - t0, 3))
                        except Exception as exc:
                            state['errors'] += 1
                            failed.append(row)
                            errors.append({'sample_id': row.get('sample_id'), 'error': repr(exc)})
                            status(stage='clip-error')
                            continue
                        if state['clips'] % 5 == 0:
                            status(stage='running')
                    pending = failed
                if stop or stop_file.exists():
                    status(stage='stopping')
                    break
                state['pairs'] += 1
                if errors:
                    with (work / 'detect_errors.jsonl').open('a') as handle_err:
                        for item in errors:
                            handle_err.write(json.dumps(dict(item, job_id=job_id)) + '\n')
                queue.finish(
                    job_id,
                    receipt={
                        'job_id': job_id,
                        'clips': len(job['rows']) - len(pending),
                        'unfinished': [r.get('sample_id') for r in pending],
                        'errors': errors[-8:],
                        'completed_at': time.time(),
                    },
                )
                status(stage='running')
            finally:
                handle.close()
    finally:
        status(stage='stopped')
        owner.close()


def detect_supervise(work: Path, node: str, with_producer: bool = False):
    """Fill this host's idle CPUs with detection workers. Does not start vLLM."""
    cfg = read(work / 'config.json')
    nc = cfg['nodes'][node]
    if socket.gethostname() != nc['hostname']:
        raise ValueError('Detect supervisor host mismatch')
    nodework = work / 'nodes' / node
    nodework.mkdir(parents=True, exist_ok=True)
    owner = acquire(nodework / 'detect_supervisor.lock', False)
    if owner is None:
        return
    write(work / 'DETECT_ONLY', {'node': node, 'at': time.time(), 'purpose': 'cpu-detection'})
    _detect_resources(work, node, DETECT_SLOT_CAP)
    sf = nodework / 'detect_supervisor.json'
    state = read(sf) if sf.exists() else {}
    children = {}
    for key, child in (state.get('children') or {}).items():
        if alive(child):
            children[key] = child
    state = {
        'children': children,
        'node': node,
        'started_at': state.get('started_at', time.time()),
        'pid': os.getpid(),
        'start_ticks': identity(os.getpid()),
        'adopted_workers': len(children),
    }
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    used_slots = [
        int(key.rsplit('-', 1)[-1])
        for key in state['children']
        if key.startswith(f'{node}-detect-')
    ]
    next_slot = (max(used_slots) + 1) if used_slots else 0
    while not stop:
        try:
            if with_producer and not (work / 'producer_complete.json').exists():
                producer_state = read(work / 'producer.json') if (work / 'producer.json').exists() else {}
                if time.time() - producer_state.get('at', 0) > 45 and not alive(state['children'].get('producer')):
                    state['children']['producer'] = spawn(
                        nodework,
                        'producer',
                        [PYTHON, str(Path(__file__).resolve()), 'producer', '--work', str(work)],
                        environment(),
                    )
            alive_slots = []
            for key, child in list(state['children'].items()):
                if key == 'producer':
                    continue
                if alive(child):
                    alive_slots.append(int(key.split('-')[-1]))
                else:
                    state['children'].pop(key, None)
            cores = _allowed_cores()
            load = os.getloadavg()[0]
            mem = _mem_available_gb()
            busy, consumed_cores = _cpu_usage_sample()
            desired = len(alive_slots)
            caption_gpus = 0
            caption_supervisor = work / 'nodes' / node / 'supervisor.json'
            if caption_supervisor.exists():
                for key, child in (read(caption_supervisor).get('children') or {}).items():
                    if str(key).startswith('gpu-') and alive(child):
                        caption_gpus += 1
            backlog = detected_uncaptioned(work)
            ceiling = detect_worker_ceiling(cores, backlog, caption_gpus)
            state['detected_uncaptioned'] = backlog
            state['detect_worker_ceiling'] = ceiling
            quota_fraction = consumed_cores / cores if consumed_cores is not None else None
            desired = detection_worker_target(desired, cores, busy, load, mem, ceiling)
            state['cpu_busy'] = round(busy, 3)
            state['cpu_used_cores'] = round(consumed_cores, 1) if consumed_cores is not None else None
            state['cpu_quota_fraction'] = round(quota_fraction, 3) if quota_fraction is not None else None
            state['cpu_control_revision'] = 'legacy-target-with-cgroup-observation-v1'
            state['target_workers'] = desired
            if len(alive_slots) > desired:
                for slot in sorted(alive_slots, reverse=True)[: len(alive_slots) - desired]:
                    write(
                        nodework / 'detect_stop' / f'{node}-detect-{slot}',
                        {'at': time.time(), 'reason': 'cpu-over-target'},
                    )
            elif len(alive_slots) < desired and mem >= 40:
                for _ in range(desired - len(alive_slots)):
                    slot = next_slot
                    next_slot += 1
                    name = f'{node}-detect-{slot}'
                    stop_path = nodework / 'detect_stop' / name
                    if stop_path.exists():
                        stop_path.unlink()
                    state['children'][name] = spawn(
                        nodework,
                        name,
                        [
                            PYTHON,
                            str(Path(__file__).resolve()),
                            'detect-worker',
                            '--work',
                            str(work),
                            '--node',
                            node,
                            '--slot',
                            str(slot),
                        ],
                        environment(),
                    )
                    alive_slots.append(slot)
            beats = nodework / 'detect_heartbeats'
            clips = 0
            if beats.exists():
                for path in beats.glob(f'{node}-detect-*.json'):
                    try:
                        clips += int(read(path).get('clips') or 0)
                    except Exception:
                        pass
            state.update(
                stage='running',
                load=load,
                cores=cores,
                mem_available_gb=round(mem, 1),
                alive_workers=len(alive_slots),
                clips=clips,
                error=None,
            )
        except Exception as exc:
            state['error'] = repr(exc)
        state['heartbeat'] = time.time()
        write(sf, state)
        time.sleep(20)
    state.update(stage='stopped', heartbeat=time.time())
    write(sf, state)
    owner.close()


def verify_sample(work: Path, limit: int = 3):
    """End-to-end verification of input preparation on retained.jsonl rows."""
    if not (work / 'config.json').exists():
        init_work(work)
    cfg = read(work / 'config.json')
    os.environ['J_OMNI_BOUNDED_CROP_V1'] = '1'
    os.environ['J_OMNI_INPUT_OPTIMIZATIONS'] = json.dumps(
        {'fast_crop': True, 'roi_crop': True, 'shared_decode': True, 'shared_raw_decode': True}
    )
    resources = cfg['nodes']['node12']['resource_work']
    old_work = str(ROOT / 'experiments/listen_j_annotation_node17_20261029')
    picked = []
    with Path(cfg['retained_manifest']).open() as handle:
        for line in handle:
            r = json.loads(line)
            cap = read(r['caption_file'])
            orig_source = cap.get('original_source') or cap['source']
            picked.append(
                {
                    'sample_id': r['sample_id'],
                    'split': r['split'],
                    'duration_s': r['duration_s'],
                    'source': orig_source,
                    'quality': r['quality'],
                }
            )
            if len(picked) >= limit:
                break
    summaries = []
    for row in picked:
        prep = prepare_input(row, old_work, str(work), resources)
        ev = prep['evidence']
        assert ev['crop_application']['revision'] == REVISION
        assert prep['effective_source']['crop_proposal']['revision'] == REVISION
        assert ev['audio_origin'] == 'virtual_official_listener_union'
        summaries.append(
            {
                'sample_id': row['sample_id'],
                'duration_s': row['duration_s'],
                'crop_changed': ev['crop_application']['coordinates_changed'],
                'actual_crop': ev['crop_application']['actual_crop'],
                'silent': ev['listener_audio_is_silent'],
                'nod_events': len(prep['nod_events']),
                'vocal_cues': len(prep['vocal_cues']),
                'reaction_cues': prep['reaction_cues'],
                'fer_status': ev['reaction_supplement']['fer_status'],
            }
        )
    print(json.dumps({'verified': len(summaries), 'samples': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'command',
        choices=[
            'init',
            'producer',
            'produce-once',
            'worker',
            'supervise',
            'watchdog',
            'verify-sample',
            'detect-worker',
            'detect-supervise',
        ],
    )
    p.add_argument('--work', type=Path, default=DEFAULT_WORK)
    p.add_argument('--node')
    p.add_argument('--gpu', type=int)
    p.add_argument('--fer-gpu', default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--prefetch', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--max-clips', type=int)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--slot', type=int)
    p.add_argument('--with-producer', action='store_true')
    args = p.parse_args()
    w = args.work.resolve()
    if args.command == 'init':
        print(json.dumps(init_work(w), ensure_ascii=False, indent=2))
    elif args.command == 'producer':
        producer(w)
    elif args.command == 'produce-once':
        print(json.dumps(producer_step(w, limit_pairs=args.limit), ensure_ascii=False))
    elif args.command == 'worker':
        worker(
            w,
            args.node,
            args.gpu,
            max_clips=args.max_clips,
            fer_gpu=args.fer_gpu,
            workers_override=args.workers,
            prefetch_override=args.prefetch,
            batch_size=args.batch_size,
            eager=args.eager,
        )
    elif args.command == 'supervise':
        supervise(w, args.node)
    elif args.command == 'watchdog':
        watchdog(w, args.node)
    elif args.command == 'detect-worker':
        detect_worker(w, args.node, args.slot)
    elif args.command == 'detect-supervise':
        detect_supervise(w, args.node, with_producer=args.with_producer)
    else:
        verify_sample(w, args.limit)
