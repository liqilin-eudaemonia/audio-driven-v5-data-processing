"""Bounded CPU read/pixel prefetch around the unchanged video metric runner.

The GPU runner retains its original frame order, arithmetic and GT cache keys.
Only native load_pair/pixel_scores scheduling is wrapped; no GPU model runs in
the preparation threads. The execution wrapper is separately recorded.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from pathlib import Path
import time
from unittest.mock import patch


def frame_key(frame):
    return (frame.__array_interface__['data'][0], frame.shape,
            frame.strides, frame.dtype.str)


class MetricPipeline:
    def __init__(self, runner, depth=2, pixel_workers=4, compute_pixels=True):
        if depth < 1 or pixel_workers < 1:
            raise ValueError('Prefetch depth and pixel workers must be positive')
        self.runner = runner
        self.depth = depth
        self.pixel_workers = pixel_workers
        self.compute_pixels = compute_pixels
        self.read = runner.read_manifest
        self.load = runner.load_pair
        self.pixel = runner.pixel_scores
        self.producer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='metric-read')
        self.pixels = ThreadPoolExecutor(max_workers=pixel_workers, thread_name_prefix='metric-pixel')
        self.pending = deque()
        self.rows = None
        self.next_index = 0
        self.current_pixels = {}
        self.current_error = None
        self.current_frames = None
        self.events = []
        self.closed = False

    def _prepare(self, row):
        start = time.monotonic()
        gt, generated, metadata = self.load(row)
        decoded = time.monotonic()
        values = None
        pixel_error = None
        if self.compute_pixels:
            try:
                if self.pixel_workers == 1:
                    values = [self.pixel(a, b) for a, b in zip(gt, generated)]
                else:
                    values = list(self.pixels.map(self.pixel, gt, generated))
            except Exception as exc:
                # Preserve native handling: load_pair succeeds and the pixel
                # stage raises, rather than misreporting a decode failure.
                pixel_error = exc
        return (gt, generated, metadata, values, pixel_error,
                dict(id=row['id'], decode_cpu_seconds=decoded-start,
                     pixels_cpu_seconds=time.monotonic()-decoded))

    def _fill(self):
        while len(self.pending) < self.depth and self.next_index < len(self.rows):
            row = self.rows[self.next_index]
            self.next_index += 1
            self.pending.append((row, self.producer.submit(self._prepare, row)))

    def read_manifest(self, path):
        if self.rows is not None:
            raise RuntimeError('One pipeline instance supports one manifest evaluation')
        self.rows = self.read(path)
        self._fill()
        return self.rows

    def load_pair(self, row):
        if not self.pending:
            raise RuntimeError('Read/pixel pipeline exhausted before native runner')
        expected, future = self.pending.popleft()
        if expected is not row:
            raise RuntimeError('Native manifest iteration order changed')
        self.current_frames = None
        self.current_pixels = {}
        self.current_error = None
        start = time.monotonic()
        try:
            gt, generated, metadata, values, error, event = future.result()
        except Exception as exc:
            self.events.append(dict(id=row['id'], error=repr(exc), wait_seconds=time.monotonic()-start))
            raise
        finally:
            self._fill()
        event['wait_seconds'] = time.monotonic()-start
        event['pixel_frames'] = 0 if values is None else len(values)
        event['pixel_error'] = None if error is None else repr(error)
        self.events.append(event)
        self.current_frames = (gt, generated)  # keep array pointers alive
        self.current_error = error
        if values is not None:
            if len(values) != len(gt) or len(gt) != len(generated):
                raise ValueError('Prepared pixel frame count mismatch')
            self.current_pixels = {(frame_key(a), frame_key(b)): value
                                   for a, b, value in zip(gt, generated, values)}
        return gt, generated, metadata

    def pixel_scores(self, gt, generated):
        if self.current_error is not None:
            raise self.current_error
        if not self.compute_pixels:
            return self.pixel(gt, generated)
        key = (frame_key(gt), frame_key(generated))
        if key not in self.current_pixels:
            raise RuntimeError('Native pixel stage received frames outside the prepared pair')
        return self.current_pixels[key]

    @contextmanager
    def installed(self):
        try:
            with patch.object(self.runner, 'read_manifest', self.read_manifest), \
                 patch.object(self.runner, 'load_pair', self.load_pair), \
                 patch.object(self.runner, 'pixel_scores', self.pixel_scores):
                yield self
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for _, future in self.pending:
            future.cancel()
        self.producer.shutdown(wait=True, cancel_futures=True)
        self.pixels.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.current_frames = None
        self.current_pixels.clear()

    def report(self):
        return dict(schema='native-video-metrics-cpu-pipeline-v1',
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    prefetch_depth=self.depth, pixel_workers=self.pixel_workers,
                    compute_pixels=self.compute_pixels, events=self.events,
                    native_loader_and_pixel_formulas_unchanged=True,
                    native_metric_frame_and_aggregation_order_unchanged=True,
                    closed=self.closed)
