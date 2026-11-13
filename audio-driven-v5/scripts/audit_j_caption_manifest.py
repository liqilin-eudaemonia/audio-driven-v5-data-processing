#!/usr/bin/env python3
"""Independently reconcile an exported caption manifest against every authority ID."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import time


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open('rb')
        self.hash = hashlib.sha256()
        self.count = 0
        self.bytes = 0

    def next(self):
        raw = self.handle.readline()
        if not raw:
            return None
        check(raw.endswith(b'\n'), 'unterminated_jsonl: ' + str(self.path))
        self.hash.update(raw)
        self.bytes += len(raw)
        self.count += 1
        return json.loads(raw)

    def close(self):
        self.handle.close()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(directory, output=None):
    directory = Path(directory).resolve()
    summary = json.loads((directory/'summary.json').read_bytes())
    marker = json.loads((directory/'EXPORT_COMPLETE.json').read_bytes())
    check(marker['summary_sha256'] == sha(directory/'summary.json'), 'summary_marker_hash_mismatch')
    readers = {name: Reader(directory/f'{name}.jsonl') for name in ['labels','coverage','exceptions']}
    authority = Reader(summary['authority_manifest'])
    # The finalized selection DB is immutable. Avoid a network-FS lock round trip
    # for every indexed lookup; verify its file identity before and after use.
    selection_sha = sha(summary['source_selection_db'])
    db = sqlite3.connect(f'file:{summary["source_selection_db"]}?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA cache_size=-262144')
    counts, flags, split_counts, seen_ids, group_splits = Counter(), Counter(), {}, set(), {}
    coverage_seconds, caption_seconds = Counter(), Counter()
    raw_checked, diagnostics_checked, selected_strata = 0, 0, Counter()
    started = last_report = time.monotonic()
    try:
        for n in itertools.count(1):
            coverage = readers['coverage'].next()
            if coverage is None:
                break
            original = authority.next()
            check(original is not None, 'extra_coverage_record')
            sid = coverage['sample_id']
            check(sid not in seen_ids, 'duplicate_coverage_id: '+sid)
            seen_ids.add(sid)
            check(sid == original['sample_id'], 'coverage_order_or_id_mismatch: '+sid)
            check(coverage['pair'] == original['pair'] and coverage['split'] == original['split'], 'coverage_metadata_mismatch: '+sid)
            check(abs(coverage['duration_s']-original['duration_s']) < 1e-7, 'coverage_duration_mismatch: '+sid)
            status, split = coverage['status'], coverage['split']
            counts[status] += 1
            split_counts.setdefault(split, Counter())[status] += 1
            coverage_seconds[split] += coverage['duration_s']
            if status != 'caption':
                exception = readers['exceptions'].next()
                check(exception == coverage, 'exception_coverage_mismatch: '+sid)
                if exception.get('diagnostic_file'):
                    raw=Path(exception['diagnostic_file']).read_bytes()
                    record=json.loads(raw)
                    check(hashlib.sha256(raw).hexdigest()==exception['diagnostic_sha256'], 'diagnostic_hash_mismatch: '+sid)
                    check(record['sample_id']==sid and record.get('reason')==exception.get('reason'), 'diagnostic_content_mismatch: '+sid)
                    if record.get('source'):
                        check(record['source']['split']==split and record['source']['pair']==original['pair'], 'diagnostic_source_mismatch: '+sid)
                    diagnostics_checked+=1
                continue
            label = readers['labels'].next()
            check(label is not None and label['sample_id'] == sid, 'label_missing_or_order_mismatch: '+sid)
            caption_seconds[split] += label['duration_s']
            check(label['split'] == split and label['pair'] == original['pair'], 'label_split_pair_mismatch: '+sid)
            check(label['caption_file'] == coverage['caption_file'] and label['caption_file_sha256'] == coverage['caption_file_sha256'], 'label_reference_mismatch: '+sid)
            prefix = Path(summary['source_work'])/'captions'/summary['caption_identity_version'][:16]
            check(Path(label['caption_file']).is_relative_to(prefix), 'old_or_other_version_caption: '+sid)
            check(label['caption_identity_version'] == summary['caption_identity_version'], 'caption_version_mismatch: '+sid)
            check(isinstance(label['caption'],str) and label['caption'].strip(), 'empty_caption: '+sid)
            check('statistics_file' not in label and label['behavior_statistics_status']=='not_recomputed_for_this_caption_version', 'stale_statistics_reference: '+sid)
            check(not label['training_admitted'] and not label['human_ground_truth'], 'unexpected_admission: '+sid)
            check(label['quality'] == original['quality'], 'quality_provenance_mismatch: '+sid)
            meta = db.execute('SELECT * FROM samples WHERE sid=?', (sid,)).fetchone()
            check(meta is not None, 'sample_not_in_authority_database: '+sid)
            for a,b in [('split','split'),('pair','pair'),('group_id','group_id'),('listener_id','listener'),('face_p10','face_p10'),('face_p50','face_p50')]:
                check(label[a] == meta[b], 'authority_field_mismatch '+a+': '+sid)
            for a,b in [('source_start_s','start'),('source_end_s','end'),('duration_s','duration')]:
                check(abs(label[a]-meta[b]) < 1e-7, 'authority_time_mismatch '+a+': '+sid)
            src=label['source']
            for key in ['sample_id','split','pair','group_id','listener_id','speaker_id','source_start_s','source_end_s','duration_s']:
                check(src[key] == label[key], 'effective_source_field_mismatch '+key+': '+sid)
            gid = label['group_id']
            check(group_splits.setdefault(gid,split)==split, 'group_cross_split: '+gid)
            review=tuple(sorted(set(label['review_flags'])))
            flags.update(review)
            counts['generation_pass' if label['generation_checks_passed'] else 'generation_review'] += 1
            counts['silent' if label['listener_audio_is_silent'] else 'nonzero_audio'] += 1
            stratum=(split,review,label['listener_audio_is_silent'])
            if selected_strata[stratum]<2 or int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%2048==0:
                raw=Path(label['caption_file']).read_bytes()
                record=json.loads(raw)
                check(hashlib.sha256(raw).hexdigest()==label['caption_file_sha256'], 'raw_caption_hash_mismatch: '+sid)
                for key in ['sample_id','split','caption','source','crop_provenance','audio_recipe','pipeline_id','prompt_version','generation_checks_passed','review_flags','remaining_gaps','finish_reason']:
                    check(record[key] == label[key], 'raw_caption_field_mismatch '+key+': '+sid)
                check(record['input_evidence']['listener_audio_is_silent']==label['listener_audio_is_silent'], 'raw_audio_silence_mismatch: '+sid)
                selected_strata[stratum]+=1
                raw_checked+=1
            now=time.monotonic()
            if now-last_report>=20:
                print(json.dumps(dict(stage='independent_audit',coverage=n,raw_caption_spotchecks=raw_checked,elapsed_s=round(now-started,1))),flush=True)
                last_report=now
        if summary['scope']=='full':
            check(authority.next() is None, 'missing_authority_ids')
        check(readers['labels'].next() is None and readers['exceptions'].next() is None, 'extra_label_or_exception')
        check(counts == Counter(summary['counts']), 'summary_count_mismatch')
        check(flags == Counter(summary['review_flag_counts']), 'review_flag_count_mismatch')
        check({k:dict(v) for k,v in split_counts.items()}==summary['splits'], 'split_count_mismatch')
        for name,reader in readers.items():
            expected=summary['files'][name]
            check(reader.count==expected['rows'] and reader.bytes==expected['bytes'] and reader.hash.hexdigest()==expected['sha256'], 'export_file_hash_or_count_mismatch: '+name)
        check(readers['coverage'].count==summary['expected_samples'], 'expected_sample_count_mismatch')
        for path,expected in summary['inputs'].items():
            check(sha(path)==expected, 'frozen_input_hash_mismatch: '+path)
        check(sha(summary['source_selection_db'])==selection_sha, 'selection_database_changed_during_audit')
        report=dict(passed=True,scope=summary['scope'],source_coverage_complete=summary['source_coverage_complete'],
            label_coverage_complete=summary['label_coverage_complete'],expected_samples=summary['expected_samples'],
            exported_captions=readers['labels'].count,exceptions=readers['exceptions'].count,
            duplicate_ids=0,missing_or_extra_coverage_ids=0,source_metadata_mismatches=0,group_cross_split=0,
            all_export_jsonl_hashes_verified=True,raw_caption_spotchecks=raw_checked,
            exception_diagnostics_verified=diagnostics_checked,source_selection_db_sha256=selection_sha,
            auditor_sha256=sha(__file__),
            coverage_hours_by_split={k:v/3600 for k,v in coverage_seconds.items()},
            caption_hours_by_split={k:v/3600 for k,v in caption_seconds.items()},
            raw_caption_spotcheck_policy='two per split/review-flags/audio-state stratum plus deterministic sha256 modulo 2048',
            original_review_flags_preserved=True,semantic_quality_verified=False,
            elapsed_s=round(time.monotonic()-started,2))
        target=Path(output) if output else directory/'COVERAGE_AUDIT.json'
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return report
    finally:
        authority.close()
        for reader in readers.values():reader.close()
        db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    audit(args.data,args.output)
