"""Bounded cross-group request refill; the engine is owned by one thread.

Annotation threads retain the existing parsing and checkpoint code. They only
submit batches here and wait for their own ordered responses. Model execution,
including rendering fallback inputs and aborting requests, stays on the owner.
"""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time


class VllmStepDriver:
    def __init__(self, llm):
        from .engine_input_renderer import check_vllm_version
        from vllm.sampling_params import RequestOutputKind
        check_vllm_version()
        if llm.llm_engine.has_unfinished_requests():
            raise ValueError('Rolling driver requires an exclusively owned idle engine')
        self.llm = llm
        self.final_only = RequestOutputKind.FINAL_ONLY

    def add(self, prepared, params, engine_inputs):
        from .engine_input_renderer import copy_video_metadata
        if any(x is not None for x in engine_inputs):
            if not all(x is not None for x in engine_inputs):
                raise ValueError('Cannot mix rendered and unrendered inputs')
            inputs = engine_inputs
        else:
            inputs = self.llm.renderer.render_cmpl(
                [copy_video_metadata(x[0]) for x in prepared])
        if len(inputs) != len(params):
            raise ValueError('Renderer returned incomplete inputs')
        added = []
        try:
            for prompt, sampling in zip(inputs, params):
                sampling.output_kind = self.final_only
                # vLLM returns randomized INTERNAL IDs from add_request, but
                # RequestOutput carries the EXTERNAL ID. Never zip these up.
                external = str(next(self.llm.request_counter))
                added.append(external)
                self.llm.llm_engine.add_request(
                    external, prompt, sampling,
                    lora_request=self.llm._resolve_mm_lora(prompt, None), priority=0)
        except BaseException:
            if added:self.abort(added)
            raise
        return added

    def step(self):
        return self.llm.llm_engine.step()

    def abort(self, external_ids):
        self.llm.llm_engine.abort_request(list(external_ids), internal=False)


@dataclass(eq=False)
class Batch:
    prepared: list
    params: list
    engine_inputs: list
    counts: list
    future: Future = field(default_factory=Future)
    responses: list = field(default_factory=list)
    remaining: int = 0


class RollingBatchEngine:
    def __init__(self, driver, max_requests, max_frames):
        if max_requests < 1 or max_frames < 1:raise ValueError('Invalid rolling capacity')
        self.driver = driver
        self.max_requests, self.max_frames = max_requests, max_frames
        self.owner = threading.get_ident()
        self.lock = threading.Lock()
        self.waiting = deque()
        self.pending = {}
        self.frames = 0
        self.closed = False
        self.peak_requests = self.peak_frames = self.refills_while_busy = 0

    def submit_inference(self, prepared, params, engine_inputs, counts):
        if not (len(prepared) == len(params) == len(engine_inputs) == len(counts)):
            raise ValueError('Batch input lengths differ')
        if not counts:return []
        if len(counts) > self.max_requests or sum(counts) > self.max_frames or any(x <= 0 for x in counts):
            raise ValueError('Batch exceeds rolling capacity')
        batch = Batch(prepared, params, engine_inputs, counts,
                      responses=[None] * len(counts), remaining=len(counts))
        with self.lock:
            if self.closed:raise RuntimeError('Rolling engine closed')
            self.waiting.append(batch)
        return batch.future.result()

    def _check_owner(self):
        if threading.get_ident() != self.owner:raise RuntimeError('Engine accessed off owner thread')

    def step(self):
        self._check_owner()
        with self.lock:
            batch = self.waiting[0] if self.waiting else None
            if batch and (len(self.pending) + len(batch.counts) > self.max_requests or
                          self.frames + sum(batch.counts) > self.max_frames):batch = None
            if batch:self.waiting.popleft()
        if batch:
            try:
                ids = self.driver.add(batch.prepared, batch.params, batch.engine_inputs)
                if len(ids) != len(batch.counts) or len(set(ids)) != len(ids) or any(x in self.pending for x in ids):
                    self.driver.abort(ids)
                    raise RuntimeError('Missing or duplicate request identity')
                self.refills_while_busy += bool(self.pending)
                for i, (request_id, count) in enumerate(zip(ids, batch.counts)):
                    self.pending[request_id] = (batch, i, count)
                self.frames += sum(batch.counts)
                self.peak_requests = max(self.peak_requests, len(self.pending))
                self.peak_frames = max(self.peak_frames, self.frames)
            except BaseException as exc:
                batch.future.set_exception(exc)
                raise
        if not self.pending:return bool(batch)
        for response in self.driver.step():
            if not response.finished:continue
            if response.request_id not in self.pending:
                raise RuntimeError(f'Unknown or duplicate final response: {response.request_id}')
            batch, i, count = self.pending.pop(response.request_id)
            self.frames -= count
            batch.responses[i] = response
            batch.remaining -= 1
            if batch.remaining == 0:batch.future.set_result(batch.responses)
        return True

    def close(self):
        self._check_owner()
        with self.lock:
            self.closed = True
            batches = list(self.waiting) + [x[0] for x in self.pending.values()]
            self.waiting.clear()
        try:
            if self.pending:self.driver.abort(list(self.pending))
        finally:
            self.pending.clear();self.frames = 0
            for batch in batches:
                if not batch.future.done():batch.future.set_exception(RuntimeError('Rolling engine closed'))


def rolling_groups(groups, load, annotate, engine, stop, status=lambda active: None):
    """Two annotation groups plus one CPU-prefetch group, finalized in order.

    The caller must close this generator before closing its source-claim
    iterator. Shutdown drains started annotation groups on a normal stop;
    failures unblock all waiters before joining their threads.
    """
    loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rolling-decode')
    annotations = ThreadPoolExecutor(max_workers=2, thread_name_prefix='rolling-annotate')
    active = deque();pending = None;exhausted = False

    def schedule():
        nonlocal exhausted
        if exhausted or stop.is_set():return None
        try:group = next(groups)
        except StopIteration:
            exhausted = True
            return None
        return loader.submit(load, group)

    try:
        pending = schedule()
        while active or pending is not None:
            if pending is not None and len(active) < 2 and (pending.done() or not active):
                if stop.is_set():pending.result();pending = None
                else:
                    group, loaded = pending.result()
                    active.append((group, loaded, annotations.submit(annotate, group, loaded, engine)))
                    pending = schedule()
            while active and active[0][2].done():
                group, loaded, future = active.popleft()
                result = future.result()
                yield group, loaded, result
                del group, loaded, future, result
            status([(group, loaded) for group, loaded, _ in active])
            if active:
                if not engine.step():time.sleep(.001)
            elif pending is not None and stop.is_set():
                pending.result();pending = None
    finally:
        # Close first: annotation threads may be waiting on request futures.
        try:engine.close()
        finally:
            loader.shutdown(wait=True, cancel_futures=True)
            annotations.shutdown(wait=True, cancel_futures=True)
