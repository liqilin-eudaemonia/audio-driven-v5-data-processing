"""Small crash-recoverable shared-filesystem queue for the frozen J policy."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2,
                      default=lambda x: x.item())
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        handle.close(); return None
    return handle


@contextmanager
def locked(path):
    handle = lock(path)
    try:
        yield
    finally:
        handle.close()


def shard(pair, count):
    return int(hashlib.sha256(pair.encode()).hexdigest()[:16], 16) % count


class Queue:
    def __init__(self, work):
        self.work = Path(work)
        self.manifest = read(self.work / 'manifest.json')
        self.nodes = [n['name'] for n in self.manifest['nodes']]
        self.items = {r['pair']: r for r in self.manifest['items']}
        self.buckets = [[p for p in self.items if shard(p, len(self.nodes)) == i]
                        for i in range(len(self.nodes))]
        self.path = self.work / 'queue.json'

    def initialize(self):
        with locked(self.work / 'queue.lock'):
            if not self.path.exists():
                write(self.path, dict(cursor=[0] * len(self.nodes), inflight={},
                                      completed=0, failed={}, attempts={}))

    def complete_result(self, pair):
        folder = self.work / 'results' / pair
        receipt = folder / 'complete.json'
        result = folder / 'result.json'
        if not receipt.exists():
            return False
        value = read(receipt)
        if value['contract'] != self.manifest['contract']:
            raise ValueError('Completed result belongs to another contract: ' + pair)
        if not result.exists() or sha(result) != value['result_sha256']:
            raise ValueError('Completed result checksum mismatch: ' + pair)
        return True

    def claim(self, node):
        if node not in self.nodes:
            raise ValueError('Unknown node: ' + node)
        with locked(self.work / 'queue.lock'):
            state = read(self.path)
            # A live owner still holds the pair flock. A dead owner can be
            # recovered immediately without guessing from a stale heartbeat.
            for pair in list(state['inflight']):
                handle = lock(self.work / 'claims' / (pair + '.lock'), False)
                if handle is None:
                    continue
                try:
                    done = self.complete_result(pair)
                except BaseException:
                    handle.close(); raise
                if done:
                    del state['inflight'][pair]; state['completed'] += 1
                    write(self.path, state); handle.close(); continue
                state['inflight'][pair] = dict(node=node, pid=os.getpid(), at=time.time())
                write(self.path, state)
                return pair, handle
            own = self.nodes.index(node)
            for bucket in [own] + [i for i in range(len(self.nodes)) if i != own]:
                values = self.buckets[bucket]
                while state['cursor'][bucket] < len(values):
                    pair = values[state['cursor'][bucket]]
                    handle = lock(self.work / 'claims' / (pair + '.lock'), False)
                    if handle is None:
                        raise RuntimeError('Unjournaled active claim: ' + pair)
                    state['cursor'][bucket] += 1
                    try:
                        done = self.complete_result(pair)
                    except BaseException:
                        handle.close(); raise
                    if done:
                        state['completed'] += 1
                        write(self.path, state); handle.close(); continue
                    state['inflight'][pair] = dict(node=node, pid=os.getpid(), at=time.time())
                    write(self.path, state)
                    return pair, handle
            write(self.path, state)
            return None

    def finish(self, pair, error=None):
        with locked(self.work / 'queue.lock'):
            state = read(self.path)
            if pair not in state['inflight']:
                return
            if error is None:
                if not self.complete_result(pair):
                    raise ValueError('Cannot finish without committed result: ' + pair)
                del state['inflight'][pair]; state['completed'] += 1
            else:
                attempts = state['attempts'].get(pair, 0) + 1
                state['attempts'][pair] = attempts
                if attempts >= self.manifest.get('max_attempts', 3):
                    state['failed'][pair] = dict(attempts=attempts, error=error)
                    del state['inflight'][pair]
            write(self.path, state)


def metrics(rows):
    def describe(key):
        values = sorted(float(r[key]) for r in rows)
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid training timing samples')
        return dict(median=statistics.median(values),
                    p95=values[max(0, math.ceil(.95 * len(values)) - 1)])
    return dict(count=len(rows), first_step=rows[0]['step'], last_step=rows[-1]['step'],
                seconds=describe('seconds'),
                between_batches_seconds=describe('between_batches_seconds'))


def slow(current, baseline, ratio=1.05):
    return any(current['seconds'][key] > baseline['seconds'][key] * ratio
               for key in ('median', 'p95'))


class Guard:
    """Two disjoint bad windows establish sustained slowdown."""
    def __init__(self, baseline, window=40):
        self.baseline = baseline; self.window = window
        self.pending = []; self.bad_windows = 0; self.last_step = None
        self.windows = []

    def add(self, rows):
        for row in rows:
            if self.last_step is not None and row['step'] <= self.last_step:
                raise ValueError('Training step reset or duplicate')
            self.last_step = row['step']; self.pending.append(row)
            if len(self.pending) == self.window:
                value = metrics(self.pending)
                value['slow'] = slow(value, self.baseline)
                self.bad_windows = self.bad_windows + 1 if value['slow'] else 0
                self.windows.append(value); self.pending = []
        return self.bad_windows >= 2
