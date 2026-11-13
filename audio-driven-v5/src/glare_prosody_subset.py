"""Source-grid prosody admission. Unknown/failed outputs never become negatives."""
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
from glare_prosody_audit import parse_reply, frame_targets

SCHEMA = 'glare-source-prosody-subset-v1'
CHUNK_SECONDS = 2.4
SAMPLE_RATE = 16000
FPS = 25
CHUNK_SAMPLES = 38400
NUMERIC_LINE = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*')
EXECUTION_STATUSES = {'input_error_exhausted', 'inference_error_exhausted', 'uncovered_source_tail'}


class IntegrityError(ValueError):
    pass


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


@contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream


def frame_count(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid pair duration')
    return math.ceil(duration * FPS - 1e-7)


def required_chunks(start, end):
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
        raise ValueError('Invalid source interval')
    def quotient(t):
        q = t / CHUNK_SECONDS
        return float(round(q)) if abs(q-round(q)) < 1e-9 else q
    return list(range(math.floor(quotient(start)), math.ceil(quotient(end))))


def audit_raw(raw, truncated=False):
    """Stricter admission around the unchanged official generator/normalizer.

    Do not repair code fences, escaped newlines, partial replies or mixed NULL.
    Legal below-threshold numeric events remain a separate known-zero category,
    following the official threshold, rather than being called explicit NULL.
    """
    if truncated:
        return parse_reply(raw, CHUNK_SECONDS, truncated=True)
    if not isinstance(raw, str) or not raw.strip():
        return parse_reply(raw, CHUNK_SECONDS)
    text = raw.strip()
    if text.upper() == 'NULL':
        return parse_reply(raw, CHUNK_SECONDS)
    if any(not NUMERIC_LINE.fullmatch(line.strip()) for line in text.splitlines() if line.strip()):
        refusal = bool(re.search(r"sorry|cannot|can't|provide.{0,40}audio|抱歉|无法|请提供", text, re.I))
        return dict(raw_text=raw, status='refusal' if refusal else 'parse_error', accepted=False,
                    normalized_text=None, events=[], errors=['raw_reply_not_numeric_lines_or_exact_NULL'],
                    format_repairs=[])
    result = parse_reply(raw, CHUNK_SECONDS)
    if result.get('format_repairs'):
        raise IntegrityError('Strict numeric reply unexpectedly needed format repair')
    return result


def generation_rows(new_ids, eos_ids, max_new_tokens):
    """Per-row EOS accounting; padding after a short answer is not truncation."""
    eos = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids]) - {None}
    rows = []
    for ids in new_ids:
        stop = next((i for i, token in enumerate(ids) if token in eos), None)
        used = ids if stop is None else ids[:stop+1]
        rows.append(dict(tokens=len(used), token_ids=list(used), eos_seen=stop is not None,
                         truncated=stop is None and len(ids) >= max_new_tokens))
    return rows


def check_generation(generation):
    derived = generation_rows([generation['token_ids']],generation['eos_ids'],generation['max_new_tokens'])[0]
    if generation['max_new_tokens'] != 64 or any(derived[k] != generation[k] for k in derived):
        raise IntegrityError('Invalid EOS/truncation evidence')


def track_identity(audio, speaker_id):
    return canonical(dict(speaker_id=speaker_id, path=str(Path(audio['path']).resolve()),
                          sha256=audio['sha256'], sample_rate=audio['sample_rate'],
                          participant_source=audio['contract']['source']))


def pair_vector(pair, chunk_records):
    """Shift continuous intervals, then raster on this pair's own25fps grid.

    Decimal subtraction prevents large source timestamps causing cancellation;
    convert to local float once, without a second two-decimal normalization.
    Native floor/ceil and chronological last-event assignment are retained.
    """
    n = frame_count(pair['duration'])
    values = np.zeros((n, 1), np.float32)
    start, end = Decimal(str(pair['start'])), Decimal(str(pair['end']))
    required = required_chunks(float(start), float(end))
    if sorted(chunk_records) != required:
        raise ValueError('Incomplete source-grid coverage')
    events = []
    for index in required:
        analysis = chunk_records[index]['analysis']
        if not analysis.get('accepted'):
            raise ValueError('Failed prosody cannot be rasterized as a zero target')
        origin = Decimal(index) * Decimal('2.4')
        for event in analysis['events']:
            a = max(start, origin + Decimal(str(event['start'])))
            b = min(end, origin + Decimal(str(event['end'])))
            if b <= a:
                continue
            local_a, local_b = float(a-start), float(b-start)
            first = max(0, math.floor(local_a*FPS))
            last = min(n, math.ceil(local_b*FPS))
            values[first:last, 0] = event['score']
            events.append(dict(start=local_a, end=local_b, score=event['score'], chunk_index=index,
                               first_frame=first, end_frame_exclusive=last))
    return values, np.ones((n, 1), bool), events


class ChunkCache:
    def __init__(self, root, contract):
        self.contract = contract
        self.contract_id = canonical(contract)
        self.root = Path(root).resolve() / self.contract_id

    def chunk_id(self, track, index):
        return canonical(dict(track_id=track['track_id'], chunk_index=int(index)))

    def identity(self, track, index):
        return dict(schema=SCHEMA, contract_id=self.contract_id, track_id=track['track_id'],
                    audio_sha256=track['audio']['sha256'], chunk_index=int(index),
                    first_sample=int(index)*CHUNK_SAMPLES, last_sample=(int(index)+1)*CHUNK_SAMPLES)

    def paths(self, track, index):
        key = self.chunk_id(track, index)
        folder = self.root/'chunks'/key[:2]
        return folder/(key+'.npz'), folder/(key+'.json')

    def load(self, track, index):
        payload, receipt = self.paths(track, index)
        if not receipt.exists():
            return None
        result = read(receipt)
        if result.get('identity') != self.identity(track, index):
            raise IntegrityError('Chunk cache identity mismatch: '+str(receipt))
        analysis = result['analysis']
        if analysis['status'] not in EXECUTION_STATUSES:
            check_generation(result['generation'])
            expected = audit_raw(analysis['raw_text'], result['generation'].get('truncated', False))
            if expected != analysis:
                raise IntegrityError('Raw reply/audit mismatch: '+str(receipt))
        if analysis['accepted']:
            if result['official_normalized_text'] != analysis['normalized_text']:
                raise IntegrityError('Official normalization differs from audited target')
            if not payload.exists() or sha(payload) != result['payload_sha256']:
                raise IntegrityError('Corrupt accepted chunk payload: '+str(payload))
            with np.load(payload, allow_pickle=False) as data:
                scores, valid = data['intensity'], data['intensity_valid']
            expected, mask = frame_targets(analysis, 60)
            if not (scores.dtype == np.float32 and valid.dtype == bool and
                    np.array_equal(scores, expected) and np.array_equal(valid, mask)):
                raise IntegrityError('Chunk vector differs from audited raw reply')
        elif result.get('payload_sha256') is not None:
            raise IntegrityError('Rejected chunk points to a training target')
        return result

    def commit(self, track, index, raw, generation, native_text, waveform_sha, attempts):
        analysis = audit_raw(raw, generation.get('truncated', False))
        if analysis['accepted'] and native_text != analysis['normalized_text']:
            raise IntegrityError('Audited accepted output differs from official normalization')
        check_generation(generation)
        return self._write(track, index, analysis, generation, native_text, waveform_sha, attempts)

    def error(self, track, index, status, error, attempts=()):
        if status not in EXECUTION_STATUSES:
            raise ValueError('Unknown execution error category')
        analysis = dict(status=status, accepted=False, raw_text=None, normalized_text=None,
                        events=[], errors=[str(error)], format_repairs=[])
        return self._write(track, index, analysis, {}, None, None, list(attempts))

    def _write(self, track, index, analysis, generation, native_text, waveform_sha, attempts):
        if self.load(track, index) is not None:
            raise IntegrityError('Refuse overwriting a terminal chunk receipt')
        payload, receipt = self.paths(track, index)
        payload.parent.mkdir(parents=True, exist_ok=True)
        digest = None
        if analysis['accepted']:
            scores, valid = frame_targets(analysis, 60)
            tmp = payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, intensity=scores, intensity_valid=valid)
                stream.flush(); os.fsync(stream.fileno())
            with np.load(tmp, allow_pickle=False) as back:
                if not np.array_equal(back['intensity'], scores) or not np.array_equal(back['intensity_valid'], valid):
                    raise IntegrityError('Chunk write/readback differs')
            tmp.replace(payload); digest = sha(payload)
        elif payload.exists():
            payload.unlink()  # Uncommitted orphan from an interrupted write only.
        result = dict(identity=self.identity(track, index), analysis=analysis, generation=generation,
                      official_normalized_text=native_text, waveform_sha256=waveform_sha,
                      payload_sha256=digest, attempts=attempts, completed_at=time.time(),
                      human_verified=False, full_glare_training_admitted=False)
        save(receipt, result)
        return result

    def retry_execution(self, track, index):
        """Archive a terminal transient failure, never overwrite accepted labels."""
        old = self.load(track, index)
        if old is None or old['analysis']['status'] not in {'input_error_exhausted', 'inference_error_exhausted'}:
            return False
        _, path = self.paths(track, index)
        archive = self.root/'retry_history'/(self.chunk_id(track, index)+'.'+str(time.time_ns())+'.json')
        archive.parent.mkdir(parents=True, exist_ok=True); path.replace(archive)
        return True

    def pair_paths(self, pair):
        key = canonical(dict(sample_id=pair['sid'], row_sha256=pair['row_sha']))
        folder = self.root/'pairs'/key[:2]
        return folder/(key+'.npz'), folder/(key+'.json')

    def admit_pair(self, pair, track, pool_sha):
        rate=track['audio']['sample_rate']
        sample_duration=(round(pair['end']*rate)-round(pair['start']*rate))/rate
        if pair['split'] not in {'train', 'dev'} or abs(pair['duration']-sample_duration) > 1e-6:
            raise IntegrityError('Invalid split or inconsistent pair interval')
        required = required_chunks(pair['start'], pair['end'])
        records = {i: self.load(track, i) for i in required}
        failures = [dict(chunk_index=i, status=r['analysis']['status']) for i,r in records.items()
                    if r is not None and not r['analysis']['accepted']]
        missing = [i for i,r in records.items() if r is None]
        if failures:
            return dict(decision='exclude', failures=failures, pending_chunks=missing)
        if missing:
            return dict(decision='pending', failures=[], pending_chunks=missing)
        refs = {str(i):sha(self.paths(track, i)[1]) for i in required}
        identity = dict(schema=SCHEMA, contract_id=self.contract_id, selection_sha256=pool_sha,
                        sample_id=pair['sid'], row_sha256=pair['row_sha'], split=pair['split'],
                        track_id=track['track_id'], start=pair['start'], end=pair['end'],
                        duration=pair['duration'], chunk_receipts_sha256=refs)
        payload, receipt = self.pair_paths(pair)
        scores, valid, events = pair_vector(pair, records)
        if receipt.exists():
            old = read(receipt)
            if old['identity'] != identity or sha(payload) != old['payload_sha256']:
                raise IntegrityError('Pair cache identity/payload changed')
            with np.load(payload, allow_pickle=False) as data:
                if (data['intensity'].dtype != np.float32 or data['intensity_valid'].dtype != bool or
                        not np.array_equal(data['intensity'], scores) or
                        not np.array_equal(data['intensity_valid'], valid)):
                    raise IntegrityError('Invalid pair vector shape/mask')
            reused = True
        else:
            scores, valid, events = pair_vector(pair, records)
            payload.parent.mkdir(parents=True, exist_ok=True)
            tmp = payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, intensity=scores, intensity_valid=valid)
                stream.flush(); os.fsync(stream.fileno())
            with np.load(tmp, allow_pickle=False) as back:
                if not np.array_equal(back['intensity'],scores) or not np.array_equal(back['intensity_valid'],valid):
                    raise IntegrityError('Pair write/readback differs')
            tmp.replace(payload)
            save(receipt, dict(identity=identity, payload_sha256=sha(payload), events=events,
                               frames=len(scores), explicit_null_chunks=[i for i,r in records.items() if r['analysis']['status']=='explicit_null'],
                               human_verified=False, full_glare_training_admitted=False, completed_at=time.time()))
            reused = False
        return dict(decision='keep', intensity=str(payload), intensity_receipt=str(receipt),
                    intensity_receipt_sha256=sha(receipt), reused=reused, required_chunks=required)
