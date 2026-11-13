#!/usr/bin/env python3
"""Apply completed, source-bound visual-quality receipts to existing annotations.

No media decoding or inference. Original annotations and statistics are immutable.
The output is a selection index, not automatic training admission.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def category(branch, decision):
    if branch == 'held':
        return 'no_face_excluded'
    if decision == 'exclude':
        return 'quality_excluded'
    if decision == 'review':
        return 'quality_review'
    if branch == 'caption':
        return 'retained'
    return 'missing_annotation_review'


def run(source_work, output):
    started = time.time()
    source_work, output = Path(source_work).resolve(), Path(output).resolve()
    cfg = read(source_work / 'config.json')
    original_summary = read(source_work / 'tables/summary.json')
    assert read(source_work / 'complete.json')['status'] == 'complete'
    upstream = Path(cfg['upstream'])
    gate = read(upstream / 'config.json')['visual_quality_gate']
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'selection.sqlite'
    if target.exists():
        raise FileExistsError('Use a new output directory; never overwrite an existing selection')
    db = sqlite3.connect(target, uri=True)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('ATTACH DATABASE ? AS inputs', (f'file:{source_work}/inputs.sqlite?mode=ro',))
    db.execute('ATTACH DATABASE ? AS stats', (f'file:{source_work}/tables/sample_stats.sqlite?mode=ro',))
    db.executescript('''CREATE TABLE receipts(jid TEXT PRIMARY KEY, path TEXT, sha256 TEXT,
        source_sha256 TEXT, clips INTEGER);
      CREATE TABLE decisions(sid TEXT PRIMARY KEY,jid TEXT,decision TEXT,source_sha256 TEXT,record TEXT);
      CREATE TABLE selection(sid TEXT PRIMARY KEY,jid TEXT,branch TEXT,status TEXT,split TEXT,
        duration REAL,quality TEXT,caption_file TEXT,source_sha256 TEXT);
      CREATE INDEX selection_status ON selection(status);
      CREATE INDEX selection_jid ON selection(jid);''')
    jobs = db.execute('SELECT jid FROM inputs.sources ORDER BY jid').fetchall()

    def receipt(jid):
        p = Path(gate['work']) / 'measure/completed' / (jid + '.json')
        raw = p.read_bytes(); r = json.loads(raw)
        assert r['job_id'] == jid and r['contract_sha256'] == gate['contract_sha256']
        assert r['execution_errors'] == 0 and len(r['decisions']) == r['clips']
        assert len({d['sample_id'] for d in r['decisions']}) == r['clips']
        counts = collections.Counter(d['decision'] for d in r['decisions'])
        assert all(counts[k] == r['counts'][k] for k in ('retain', 'exclude', 'review'))
        for d in r['decisions']:
            assert d['contract_sha256'] == gate['contract_sha256']
            assert d['decision'] in ('retain', 'exclude', 'review')
        return jid, str(p), hashlib.sha256(raw).hexdigest(), r

    # Bounded receipts only: no directory walk, no million-file media scan.
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for start in range(0, len(jobs), 128):
            with db:
                for jid, path, sha, r in pool.map(receipt, [j[0] for j in jobs[start:start + 128]]):
                    db.execute('INSERT INTO receipts VALUES (?,?,?,?,?)',
                               (jid, path, sha, r['source_result_sha256'], r['clips']))
                    db.executemany('INSERT INTO decisions VALUES (?,?,?,?,?)',
                        [(d['sample_id'], jid, d['decision'], d['source_sha256'], json.dumps(d, ensure_ascii=False)) for d in r['decisions']])
            if start % 4096 == 0:
                print(json.dumps({'stage': 'receipts', 'sources': min(start + 128, len(jobs))}), flush=True)
    assert len(jobs) == cfg['expected_upstream_sources']
    assert db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0] == cfg['expected_upstream_clips']
    assert not db.execute('SELECT c.sid FROM inputs.coverage c LEFT JOIN decisions d ON c.sid=d.sid WHERE d.sid IS NULL LIMIT 1').fetchall()
    assert not db.execute('SELECT d.sid FROM decisions d LEFT JOIN inputs.coverage c ON c.sid=d.sid WHERE c.sid IS NULL LIMIT 1').fetchall()
    # Full outer coverage is established above; now verify the actual source identity.
    counts = collections.Counter(); seconds = collections.Counter(); reasons = collections.Counter()
    cross = collections.Counter(); splits = collections.Counter(); pending = []
    query = '''SELECT c.record,d.record,r.source_sha256 FROM inputs.coverage c
               JOIN decisions d ON c.sid=d.sid JOIN receipts r ON d.jid=r.jid'''
    for i, (raw, decision_raw, source_result_sha) in enumerate(db.execute(query)):
        c, d = json.loads(raw), json.loads(decision_raw); s = c['source']
        assert c['job_id'] == s['pair'] and d['sample_id'] == c['sample_id']
        assert digest(s) == d['source_sha256'] and s['source_result_sha256'] == source_result_sha
        status = category(c['branch'], d['decision'])
        if status == 'no_face_excluded':
            held = read(c['terminal_file'])
            assert 'no_detected_face' in held.get('reason', ''), ('Unconfirmed held reason', c['sample_id'])
        counts[status] += 1; seconds[status] += s['duration_s']
        cross[(c['branch'], d['decision'])] += 1; splits[(status, s['split'])] += 1
        for reason in d['reasons'] + d['review_reasons']:
            reasons[(status, reason)] += 1
        pending.append((c['sample_id'], c['job_id'], c['branch'], status, s['split'],
                        s['duration_s'], d['decision'], c['terminal_file'] if c['branch'] == 'caption' else None, d['source_sha256']))
        if len(pending) >= 4096:
            with db: db.executemany('INSERT INTO selection VALUES (?,?,?,?,?,?,?,?,?)', pending)
            pending.clear()
        if i % 100000 == 0:
            print(json.dumps({'stage': 'source_identity', 'clips': i + 1}), flush=True)
    with db: db.executemany('INSERT INTO selection VALUES (?,?,?,?,?,?,?,?,?)', pending)
    assert sum(counts.values()) == cfg['expected_upstream_clips']
    assert not db.execute('''SELECT s.sid FROM selection s LEFT JOIN stats.samples x ON s.sid=x.sid
                          WHERE s.branch='caption' AND x.sid IS NULL LIMIT 1''').fetchall()
    db.executescript('''CREATE TABLE samples AS SELECT x.* FROM stats.samples x
      JOIN selection s ON x.sid=s.sid WHERE s.status='retained';
      CREATE UNIQUE INDEX sample_id ON samples(sid);
      CREATE INDEX source_time ON samples(pair,listener,start,end);
      CREATE TABLE behaviors AS SELECT b.* FROM stats.behaviors b JOIN samples s ON b.sid=s.sid;
      CREATE UNIQUE INDEX behavior_sample_type ON behaviors(sid,type);
      CREATE INDEX behavior_type ON behaviors(type,split);''')
    assert db.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == counts['retained']
    assert not db.execute("SELECT sid FROM samples EXCEPT SELECT sid FROM selection WHERE status='retained' LIMIT 1").fetchall()
    names = ('retained', 'quality_excluded', 'quality_review', 'no_face_excluded', 'missing_annotation_review')
    exports = {}
    for status in names:
        path = output / (status + '.jsonl'); h = hashlib.sha256(); n = 0
        q = '''SELECT s.sid,s.jid,s.split,s.duration,s.branch,s.caption_file,d.record,
                      x.file,x.offset FROM selection s JOIN decisions d ON s.sid=d.sid
                      LEFT JOIN stats.samples x ON x.sid=s.sid WHERE s.status=? ORDER BY s.sid'''
        with path.open('wb') as f:
            for sid, jid, split, duration, branch, caption, decision, stat_file, offset in db.execute(q, (status,)):
                rec = dict(sample_id=sid, pair=jid, split=split, duration_s=duration,
                    selection_status=status, original_branch=branch, caption_file=caption,
                    statistics_file=stat_file, statistics_byte_offset=offset,
                    quality= json.loads(decision), training_admitted=False,
                    selection_scope='visual_quality_only_not_training_admission')
                data = (json.dumps(rec, ensure_ascii=False) + '\n').encode(); f.write(data); h.update(data); n += 1
        assert n == counts[status]
        exports[status] = dict(path=str(path), rows=n, sha256=h.hexdigest())
    # Same definitions as the original full-scope catalog; only the selected IDs change.
    fields = ['type','sample_count','pair_count','literal_occurrences_sum','derived_occurrences_sum',
              'known_occurrence_samples','unknown_occurrence_samples','state_groups','pending_samples',
              'covered_clip_seconds','change_samples']
    catalog = []
    query = '''SELECT type,SUM(counted),COUNT(DISTINCT CASE WHEN counted=1 THEN pair END),
       SUM(CASE WHEN unit='occurrence' AND basis='literal' THEN quantity END),
       SUM(CASE WHEN unit='occurrence' AND basis='derived_from_distinct_mentions' THEN quantity END),
       SUM(CASE WHEN unit='occurrence' AND quantity IS NOT NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN unit='occurrence' AND quantity IS NULL THEN 1 ELSE 0 END),
       SUM(state),SUM(pending),SUM(duration),SUM(change) FROM behaviors GROUP BY type ORDER BY type'''
    for values in db.execute(query):
        c = dict(zip(fields, values))
        c['supported_occurrences_sum'] = sum(c[k] or 0 for k in ('literal_occurrences_sum','derived_occurrences_sum')) if c['known_occurrence_samples'] else None
        catalog.append(c)
    with (output / 'behavior_catalog.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields + ['supported_occurrences_sum'])
        writer.writeheader(); writer.writerows(catalog)
    result = dict(schema='listen-quality-refilter-v1', started_at=started, finished_at=time.time(),
        source_work=str(source_work), upstream=str(upstream), quality_gate=gate,
        total_sources=len(jobs), total_clips=sum(counts.values()), original_caption_count=original_summary['sample_count'],
        counts=dict(counts), hours={k:v/3600 for k,v in seconds.items()},
        branches=[dict(branch=a,quality=b,count=v) for (a,b),v in sorted(cross.items())],
        split_counts=[dict(status=a,split=b,count=v) for (a,b),v in sorted(splits.items())],
        reasons=[dict(status=a,reason=b,count=v) for (a,b),v in sorted(reasons.items())],
        retained_stat_status=dict(db.execute('SELECT status,COUNT(*) FROM samples GROUP BY status')),
        retained_zero_counted_types=db.execute('SELECT COUNT(*) FROM samples WHERE types=0').fetchone()[0],
        distinct_candidate_names=sum(c['sample_count']>0 for c in catalog),
        precise_taxonomy=False, action_duration_available=False, exports=exports,
        annotation_text_changed=False, reannotation=False, new_media=False, selection_is_not_training_admission=True)
    assert db.execute('PRAGMA quick_check').fetchone() == ('ok',)
    db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (output / 'complete.json').write_text(json.dumps(dict(passed=True, at=time.time(),
        full_source_identity_checked=True, full_quality_receipt_coverage=True,
        retained_count=counts['retained'], summary=str(output/'summary.json')), indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-work', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.source_work, args.output)
