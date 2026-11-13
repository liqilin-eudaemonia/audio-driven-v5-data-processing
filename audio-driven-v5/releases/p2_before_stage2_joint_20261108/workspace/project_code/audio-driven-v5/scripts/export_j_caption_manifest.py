#!/usr/bin/env python3
"""Export version-bound J captions without filtering on generation review flags.

The retained manifest is the authority. Old captions/statistics are never used as
fallbacks. One coverage record is written for every requested sample, including
held or missing captions. This does not grant training admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.caption_guard import (
    caption_file, clip_identity, digest, identity_matches,
    measurement_sha256, shared_identity,
)
from listen_benchmark.j_annotation import result_path


def read(path):
    return json.loads(Path(path).read_bytes())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_caption(record, row, authority, metadata, measurement, config):
    sid, source = row['sample_id'], row['source']
    require(record['sample_id'] == sid == authority['sample_id'] == metadata['sid'], 'sample_id_mismatch')
    require(record['split'] == source['split'] == row['split'] == authority['split'] == metadata['split'], 'split_mismatch')
    require(source['pair'] == authority['pair'] == metadata['pair'], 'pair_mismatch')
    require(source['group_id'] == metadata['group_id'], 'group_mismatch')
    require(source['listener_id'] == metadata['listener'], 'listener_mismatch')
    require(source['listener_id'] != source['speaker_id'], 'speaker_listener_identical')
    for a, b in [(source['source_start_s'], metadata['start']), (source['source_end_s'], metadata['end']),
                 (source['duration_s'], metadata['duration']), (source['duration_s'], authority['duration_s'])]:
        require(abs(a - b) < 1e-7, 'source_time_mismatch')
    require(authority['selection_status'] == 'retained' and authority['quality']['decision'] == 'retain', 'not_quality_retained')
    require(digest(source) == authority['quality']['source_sha256'], 'original_source_hash_mismatch')
    require(record.get('original_source', record['source']) == source, 'original_source_mismatch')
    require(identity_matches(record.get('cache_identity'), clip_identity(source, measurement)), 'caption_identity_mismatch')
    require(record['pipeline_id'] == config['pipeline_id'] and record['prompt_version'] == config['prompt_version'], 'generation_version_mismatch')
    require(record['source']['sample_id'] == sid and record['source']['split'] == authority['split'], 'effective_source_mismatch')
    provenance = record['crop_provenance']
    require(provenance['revision'] == shared_identity()['crop_revision'], 'crop_revision_mismatch')
    require(provenance['actual_crop'] == record['source']['video_transform']['crop'], 'actual_crop_mismatch')
    require(provenance['measurement_sha256'] == authority['quality']['measurement_sha256'], 'quality_measurement_mismatch')
    require(record['audio_recipe']['listener_id'] == source['listener_id'], 'audio_listener_mismatch')
    require(isinstance(record.get('caption'), str) and record['caption'].strip(), 'empty_caption')


def label_record(record, authority, metadata, path, raw_sha, version):
    source = record['source']
    return dict(
        sample_id=record['sample_id'], pair=authority['pair'], split=authority['split'],
        group_id=source['group_id'], listener_id=source['listener_id'], speaker_id=source['speaker_id'],
        source_start_s=source['source_start_s'], source_end_s=source['source_end_s'],
        duration_s=authority['duration_s'], face_p10=metadata['face_p10'], face_p50=metadata['face_p50'],
        caption=record['caption'], caption_file=str(path), caption_file_sha256=raw_sha,
        caption_identity_version=version, pipeline_id=record['pipeline_id'], prompt_version=record['prompt_version'],
        source=source, quality=authority['quality'], crop_provenance=record['crop_provenance'],
        audio_recipe=record['audio_recipe'], listener_audio_is_silent=record['input_evidence']['listener_audio_is_silent'],
        detector_measurement_sha256=record['cache_identity']['detector_measurement_sha256'],
        generation_checks_passed=record['generation_checks_passed'],
        review_flags=record.get('review_flags') or [], remaining_gaps=record.get('remaining_gaps') or {},
        finish_reason=record['finish_reason'], completed_at=record['completed_at'],
        training_admitted=False, human_ground_truth=False,
        behavior_statistics_status='not_recomputed_for_this_caption_version',
    )


class Exporter:
    def __init__(self, work, selection, config, version):
        self.work, self.selection, self.config, self.version = work, selection, config, version
        self.local = threading.local()

    def process_pair(self, group):
        pair, authorities = group
        if not hasattr(self.local, 'db'):
            self.local.db = sqlite3.connect(f'file:{self.selection}?mode=ro', uri=True)
            self.local.db.row_factory = sqlite3.Row
        metadata = {r['sid']: dict(r) for r in self.local.db.execute('SELECT * FROM samples WHERE pair=?', (pair,))}
        job_path = self.work / 'jobs' / (pair + '.json')
        job = read(job_path)
        require(job['job_id'] == pair, 'job_id_mismatch')
        rows = {r['sample_id']: r for r in job['rows']}
        require(len(rows) == len(job['rows']), 'duplicate_job_sample')
        require(set(rows) == {r['sample_id'] for r in authorities}, 'job_authority_ids_mismatch')
        output = []
        for authority in authorities:
            sid = authority['sample_id']
            coverage = dict(sample_id=sid, pair=pair, split=authority['split'], duration_s=authority['duration_s'])
            try:
                row = rows[sid]
                measurement = measurement_sha256(self.work, sid)
                path = caption_file(self.work, row['source'], measurement, sid)
                if not path.exists():
                    for branch in ('held', 'failed'):
                        diagnostic = result_path(self.work, branch, sid)
                        if diagnostic.exists():
                            raw = diagnostic.read_bytes()
                            item = json.loads(raw)
                            require(item['sample_id'] == sid, 'diagnostic_id_mismatch')
                            coverage.update(status=branch, reason=item.get('reason'),
                                diagnostic_file=str(diagnostic), diagnostic_sha256=hashlib.sha256(raw).hexdigest())
                            break
                    else:
                        coverage.update(status='missing', reason='no_caption_for_current_identity')
                    output.append((None, coverage))
                    continue
                raw = path.read_bytes()
                record = json.loads(raw)
                validate_caption(record, row, authority, metadata[sid], measurement, self.config)
                raw_sha = hashlib.sha256(raw).hexdigest()
                label = label_record(record, authority, metadata[sid], path, raw_sha, self.version)
                coverage.update(status='caption', caption_file=str(path), caption_file_sha256=raw_sha,
                    generation_checks_passed=label['generation_checks_passed'])
                output.append((label, coverage))
            except Exception as exc:
                coverage.update(status='invalid', reason=f'{type(exc).__name__}: {exc}')
                output.append((None, coverage))
        return output


_PROCESS_EXPORTER = None


def process_init(work, selection, config, version):
    global _PROCESS_EXPORTER
    _PROCESS_EXPORTER = Exporter(work, selection, config, version)


def process_pair(group):
    return _PROCESS_EXPORTER.process_pair(group)


def authority_groups(path):
    with path.open() as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        for pair, items in itertools.groupby(rows, key=lambda r: r['pair']):
            yield pair, list(items)


class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self.handle = path.open('xb')
        self.hash = hashlib.sha256()
        self.count = 0
        self.bytes = 0

    def write(self, value):
        raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
        self.handle.write(raw)
        self.hash.update(raw)
        self.count += 1
        self.bytes += len(raw)

    def close(self):
        self.handle.close()
        return dict(file=self.path.name, rows=self.count, bytes=self.bytes, sha256=self.hash.hexdigest())


def export(work, output, selection, workers=12, limit_pairs=None, backend='process'):
    work, output, selection = work.resolve(), output.resolve(), selection.resolve()
    require(not output.exists(), 'output_already_exists_use_new_directory')
    config = read(work / 'config.json')
    authority = Path(config['retained_manifest'])
    producer = read(work / 'producer_complete.json')
    queue = read(work / 'queues/caption/state.json')
    require(producer.get('eof'), 'producer_not_finished')
    require(not any(queue.get(k) for k in ('inflight', 'deferred', 'failed')), 'queue_not_drained')
    require(queue['offset'] >= (work / 'ready.jsonl').stat().st_size, 'queue_not_drained')
    version = digest(shared_identity())
    snapshots = {str(p): file_sha(p) for p in [Path(__file__).resolve(), authority, work / 'config.json', work / 'producer_complete.json',
        work / 'queues/caption/state.json', work / 'ready.jsonl']}
    output.mkdir(parents=True)
    (output / 'STARTED.json').write_text(json.dumps(dict(work=str(work), identity_version=version,
        workers=workers, backend=backend, scope='pilot' if limit_pairs else 'full', inputs=snapshots), indent=2) + '\n')
    writers = {name: JsonlWriter(output / f'{name}.jsonl') for name in ('labels', 'coverage', 'exceptions')}
    counts, splits, flags, flag_combinations = Counter(), {}, Counter(), Counter()
    groups_seen, ids_seen, group_splits = set(), set(), {}
    started = last_report = time.monotonic()
    exporter = Exporter(work, selection, config, version)
    groups = authority_groups(authority)
    if limit_pairs:
        groups = itertools.islice(groups, limit_pairs)
    try:
        pool_args = dict(max_workers=workers)
        if backend == 'process':
            pool_class, worker_function = ProcessPoolExecutor, process_pair
            pool_args.update(initializer=process_init, initargs=(work, selection, config, version))
        else:
            pool_class, worker_function = ThreadPoolExecutor, exporter.process_pair
        with pool_class(**pool_args) as pool:
            # Bounded batches retain authority order and avoid a million queued futures.
            while batch := list(itertools.islice(groups, workers * 2)):
                for pair, items in batch:
                    require(pair not in groups_seen, 'non_contiguous_or_duplicate_pair')
                    groups_seen.add(pair)
                    for item in items:
                        require(item['sample_id'] not in ids_seen, 'duplicate_authority_id')
                        ids_seen.add(item['sample_id'])
                for results in pool.map(worker_function, batch):
                    for label, coverage in results:
                        writers['coverage'].write(coverage)
                        status, split = coverage['status'], coverage['split']
                        counts[status] += 1
                        splits.setdefault(split, Counter())[status] += 1
                        if label is None:
                            writers['exceptions'].write(coverage)
                            continue
                        writers['labels'].write(label)
                        counts['generation_pass' if label['generation_checks_passed'] else 'generation_review'] += 1
                        counts['silent' if label['listener_audio_is_silent'] else 'nonzero_audio'] += 1
                        flags.update(set(label['review_flags']))
                        flag_combinations[tuple(sorted(set(label['review_flags'])))] += 1
                        gid = label['group_id']
                        require(group_splits.setdefault(gid, split) == split, 'group_crosses_split')
                now = time.monotonic()
                if now - last_report >= 20:
                    print(json.dumps(dict(stage='export', expected_seen=len(ids_seen), captions=counts['caption'],
                        exceptions=writers['exceptions'].count, pairs=len(groups_seen), elapsed_s=round(now-started, 1),
                        clips_per_s=round(len(ids_seen)/(now-started), 2))), flush=True)
                    last_report = now
        files = {name: writer.close() for name, writer in writers.items()}
        for path, original_sha in snapshots.items():
            require(file_sha(path) == original_sha, 'upstream_changed_during_export: ' + path)
        shared_identity.cache_clear()
        require(digest(shared_identity()) == version, 'caption_identity_changed_during_export')
        if not limit_pairs:
            require(len(ids_seen) == config['total_retained_clips'] == producer['published_clips'], 'authority_total_mismatch')
        require(files['coverage']['rows'] == files['labels']['rows'] + files['exceptions']['rows'] == len(ids_seen), 'coverage_accounting_mismatch')
        summary = dict(schema='j-current-caption-manifest-v1', scope='pilot' if limit_pairs else 'full',
            source_work=str(work), authority_manifest=str(authority), source_selection_db=str(selection),
            caption_identity_version=version, pipeline_id=config['pipeline_id'], prompt_version=config['prompt_version'],
            expected_samples=len(ids_seen), exported_captions=files['labels']['rows'], exceptions=files['exceptions']['rows'],
            counts=dict(counts), splits={k:dict(v) for k,v in splits.items()},
            review_flag_counts=dict(flags), review_flag_combinations=[dict(flags=list(k), samples=v) for k,v in flag_combinations.items()],
            source_coverage_complete=not limit_pairs, label_coverage_complete=not limit_pairs and files['exceptions']['rows']==0,
            original_review_flags_preserved=True, semantic_acceptance_applied=False, training_admission_applied=False,
            old_caption_fallback=False, old_behavior_statistics_reused=False,
            duplicate_ids=0, source_metadata_mismatches=counts['invalid'], group_cross_split=0,
            files=files, inputs=snapshots, elapsed_s=round(time.monotonic()-started, 2))
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        (output / 'EXPORT_COMPLETE.json').write_text(json.dumps(dict(
            export_completed=True, source_coverage_complete=summary['source_coverage_complete'],
            label_coverage_complete=summary['label_coverage_complete'], summary_sha256=file_sha(output/'summary.json')),
            indent=2) + '\n')
        print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','files','review_flag_combinations')}, ensure_ascii=False), flush=True)
        return summary
    finally:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', type=Path, default=ROOT/'experiments/listen_j_final_quality_selection_20261101/selection.sqlite')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--backend', choices=('thread', 'process'), default='process')
    p.add_argument('--limit-pairs', type=int, help='Pilot only; never marked as full coverage.')
    args = p.parse_args()
    require(args.workers > 0, 'workers_must_be_positive')
    require(args.limit_pairs is None or args.limit_pairs > 0, 'limit_pairs_must_be_positive')
    export(args.work, args.output, args.selection, args.workers, args.limit_pairs, args.backend)


if __name__ == '__main__':
    main()
