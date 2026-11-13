"""Bounded recovery scans: at most four inflight flock probes per claim.

Same journal, pair locks, durable receipts and shard order as the original queue.
The shared recovery cursor prevents repeatedly scanning hundreds of live owners.
"""
import json
import os
import time
from pathlib import Path
import uuid
from listen_pair_j_queue import Queue as OriginalQueue, lock, locked, read


def write_journal(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(value, f, separators=(',', ':'))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Queue(OriginalQueue):
    recovery_budget = 4

    def claim(self, node):
        if node not in self.nodes: raise ValueError('Unknown node: ' + node)
        with locked(self.work / 'queue.lock'):
            state = read(self.path)
            pairs = list(state['inflight'])
            offset = state.get('recovery_cursor', 0) % max(1, len(pairs))
            count = min(self.recovery_budget, len(pairs))
            state['recovery_cursor'] = (offset + count) % max(1, len(pairs))
            for index in range(count):
                pair = pairs[(offset + index) % len(pairs)]
                handle = lock(self.work / 'claims' / (pair + '.lock'), False)
                if handle is None: continue
                try:
                    done = self.complete_result(pair)
                except BaseException:
                    handle.close(); raise
                if done:
                    del state['inflight'][pair]; state['completed'] += 1
                    handle.close(); continue
                state['inflight'][pair] = dict(node=node, pid=os.getpid(), at=time.time())
                write_journal(self.path, state)
                return pair, handle
            own = self.nodes.index(node)
            for bucket in [own] + [i for i in range(len(self.nodes)) if i != own]:
                values = self.buckets[bucket]
                while state['cursor'][bucket] < len(values):
                    pair = values[state['cursor'][bucket]]
                    handle = lock(self.work / 'claims' / (pair + '.lock'), False)
                    if handle is None: raise RuntimeError('Unjournaled active claim: ' + pair)
                    state['cursor'][bucket] += 1
                    try:
                        done = self.complete_result(pair)
                    except BaseException:
                        handle.close(); raise
                    if done:
                        state['completed'] += 1; handle.close(); continue
                    state['inflight'][pair] = dict(node=node, pid=os.getpid(), at=time.time())
                    write_journal(self.path, state)
                    return pair, handle
            write_journal(self.path, state)
            return None

    def finish(self, pair, error=None):
        # Caller retains the pair lock, so validate its immutable commit outside
        # the global journal lock. No other process can publish this pair.
        if error is None and not self.complete_result(pair):
            raise ValueError('Cannot finish without committed result: ' + pair)
        with locked(self.work / 'queue.lock'):
            state = read(self.path)
            if pair not in state['inflight']: return
            if error is None:
                del state['inflight'][pair]; state['completed'] += 1
            else:
                attempts = state['attempts'].get(pair, 0) + 1
                state['attempts'][pair] = attempts
                if attempts >= self.manifest.get('max_attempts', 3):
                    state['failed'][pair] = dict(attempts=attempts, error=error)
                    del state['inflight'][pair]
            write_journal(self.path, state)
