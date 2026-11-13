"""Cooperative tail sharing. Source receipts stay with their original owner.

All participants reserve a clip before preparation and keep its flock through
output persistence (or cancellation/draining). Helpers only consume advertised,
unstarted rows; heartbeat age never revokes a live lock.
"""
import time
from pathlib import Path
from .j_annotation import acquire, read, result_path

PROTOCOL = 'clip-lock-tail-v1'
TERMINAL = ('caption', 'held', 'visual_excluded', 'visual_review')


def terminal(work, sid):
    return any(result_path(work, branch, sid).exists() for branch in TERMINAL)


class TailSharing:
    def __init__(self, work, queue, name, cfg, state):
        self.work = Path(work)
        self.queue, self.name, self.state = queue, name, state
        self.config = cfg.get('tail_sharing', {})
        self.enabled = self.config.get('enabled', False)
        self.last_check = float('-inf')
        self.last_help = float('-inf')
        self.tail = False
        self.activated = False
        self.snapshot = {}
        if not self.enabled:
            return
        self.workers = self.config['worker_count']
        self.packet_size = self.config.get('packet_size', 8)
        self.threshold = self.config.get('ready_sources_per_worker', 2) * self.workers
        self.poll_seconds = self.config.get('poll_seconds', 5)
        if (type(self.workers) is not int or self.workers < 1 or
                type(self.packet_size) is not int or not 1 <= self.packet_size <= 16 or
                type(self.threshold) is not int or self.threshold < self.workers or
                not 1 <= self.poll_seconds <= 30):
            raise ValueError('Invalid tail sharing policy')
        state.update(tail_protocol=PROTOCOL, tail_mode=False, tail_offers=[],
                     tail_lock_release='consumer-close-after-persist',
                     tail_packet_lifecycle='finish-before-source-claim',
                     assisted_clips_claimed=0, assisted_packets=0)

    def limit(self, usual):
        if not self.enabled:
            return usual
        now = time.monotonic()
        if now - self.last_check >= self.poll_seconds:
            self.last_check = now
            self.snapshot = read(self.queue.path)
            pending = len(self.snapshot.get('deferred', {}))
            journal = self.work / 'ready.jsonl'
            if journal.exists():
                # Bounded lookahead, not a scan of the source/result tree.
                with journal.open() as f:
                    f.seek(self.snapshot['offset'])
                    while pending < self.threshold:
                        line = f.readline()
                        if not line or not line.endswith('\n'):
                            break
                        pending += 1
            self.activated = (self.work / 'tail_sharing' / 'ENABLED.json').exists()
            self.tail = self.activated and pending < self.threshold
            self.state.update(tail_mode=self.tail, tail_ready_source_lower_bound=pending,
                              tail_helpers_enabled=self.activated)
        limit = 1 if self.tail else usual
        self.state['effective_max_active_sources'] = limit
        return limit

    def reserve(self, sid):
        """Caller checks terminal again after acquiring the exclusive lock."""
        return acquire(result_path(self.work, 'tail_sharing/clip_locks', sid).with_suffix('.lock'), False)

    def advertise(self, active):
        if not self.enabled:
            return
        # Atomic replacement of the list for the existing heartbeat writer.
        self.state['tail_offers'] = [
            dict(job_id=jid, remaining=len(item['rows']),
                 sample_ids=[r['sample_id'] for r in list(item['rows'])[:64]],
                 source_result_sha256=item['job']['source_result_sha256'])
            for jid, item in active.items() if not item.get('helper') and item['rows']
        ] if self.tail else []

    def offers(self, active):
        if not self.enabled or not self.tail:
            return []
        now = time.monotonic()
        if now - self.last_help < self.poll_seconds:
            return []
        self.last_help = now
        snapshot = read(self.queue.path)
        owners = {}
        for jid, claim in snapshot['inflight'].items():
            if jid not in active and claim['worker'] != self.name:
                owners.setdefault(claim['worker'], {})[jid] = claim
        offers = []
        for owner, claims in owners.items():
            path = self.work / 'workers' / f'{owner}.json'
            if not path.exists():
                continue
            state = read(path)
            if state.get('tail_protocol') != PROTOCOL or time.time()-state.get('heartbeat', 0) > 60:
                continue
            for offer in state.get('tail_offers', []):
                claim = claims.get(offer['job_id'])
                if claim and claim['pid'] == state.get('pid'):
                    offers.append(dict(offer, source_owner=owner))
        return sorted(offers, key=lambda x: (-x['remaining'], x['job_id']))

    def helper_failed_path(self, sid):
        return result_path(self.work, 'tail_sharing/failed_assists', sid)
