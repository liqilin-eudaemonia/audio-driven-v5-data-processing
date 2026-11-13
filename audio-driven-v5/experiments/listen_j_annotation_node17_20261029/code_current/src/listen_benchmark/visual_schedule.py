"""CPU-only scheduling helpers. Scheduling never changes the sampled input."""
import heapq
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .full_raw import views


def source_duration(row):
    source = row['source']
    rate = source['normalized_audio']['sample_rate']
    duration = (round(source['source_end_s'] * rate) -
                round(source['source_start_s'] * rate)) / rate
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration: {row['sample_id']}")
    return duration


def estimated_work(row):
    # Frame-equivalent proxy, not a measured runtime. Include every request,
    # since a longer clip also creates more overlapping windows.
    return sum(math.ceil((end - start) * 8) + 16
               for _, start, end in views(source_duration(row)))


def assign_rows(rows, shards, mode='balanced'):
    if shards < 1 or mode not in ('balanced', 'round-robin'):
        raise ValueError('Invalid scheduling options')
    if mode == 'round-robin':
        return [rows[rank::shards] for rank in range(shards)]
    assignments = [[] for _ in range(shards)]
    loads = [(0, rank) for rank in range(shards)]
    heapq.heapify(loads)
    jobs = sorted(((estimated_work(row), index, row) for index, row in enumerate(rows)),
                  key=lambda item: (-item[0], item[1]))
    for cost, _, row in jobs:
        load, rank = heapq.heappop(loads)
        assignments[rank].append(row)
        heapq.heappush(loads, (load + cost, rank))
    return assignments


def prefetched(items, load, ahead=1):
    """One CPU reader, at most `ahead` future clips plus the current clip.

    Exceptions propagate to the consumer. Closing the generator cancels queued
    reads and waits only for an already-running read; no orphan background IO.
    """
    if ahead not in (0, 1, 2):
        raise ValueError('prefetch must be 0, 1 or 2')
    if not ahead:
        for item in items:
            yield item, load(item)
        return
    iterator = iter(items)
    pending = deque()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-decode') as pool:
        def submit():
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.append((item, pool.submit(load, item)))

        for _ in range(ahead):
            submit()
        try:
            while pending:
                item, future = pending.popleft()
                value = future.result()
                submit()
                yield item, value
                del value
        finally:
            for _, future in pending:
                future.cancel()


def window_batches(windows, batch_size, frame_budget=256):
    """Batch similar durations; never split or truncate an individual request."""
    if batch_size < 1 or frame_budget < 1:
        raise ValueError('Batch limits must be positive')
    batch, frames = [], 0
    for window in sorted(windows, key=lambda w: -(w[2] - w[1])):
        cost = 2 * math.ceil(math.ceil((window[2] - window[1]) * 8) / 2)
        if batch and (len(batch) >= batch_size or frames + cost > frame_budget):
            yield batch
            batch, frames = [], 0
        batch.append(window)
        frames += cost
    if batch:
        yield batch
