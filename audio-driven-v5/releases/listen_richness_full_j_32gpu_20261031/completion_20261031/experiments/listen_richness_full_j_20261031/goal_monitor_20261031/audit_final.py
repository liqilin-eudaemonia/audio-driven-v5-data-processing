"""Read-only full-scope verification; writes audit evidence beside this script."""
from pathlib import Path
import collections, hashlib, itertools, json, sqlite3, sys, time

W = Path(__file__).resolve().parents[1]
ROOT = W.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from listen_richness.common import digest
from listen_richness.production import contract, statistic_record

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()

def main():
    started = time.time()
    cfg = read(W / 'config.json')
    q = read(W / 'queues/extract/state.json')
    assert not q['failed'] and not q['inflight']
    assert q['offset'] == (W / 'ready.jsonl').stat().st_size
    db = sqlite3.connect(f'file:{W}/inputs.sqlite?mode=ro', uri=True)
    assert db.execute('PRAGMA quick_check').fetchone() == ('ok',)
    coverage = dict(db.execute('SELECT status,COUNT(*) FROM coverage GROUP BY status'))
    assert sum(coverage.values()) == cfg['expected_upstream_clips'] == 1151805
    assert db.execute('SELECT COUNT(*) FROM sources').fetchone()[0] == cfg['expected_upstream_sources'] == 60869
    assert not db.execute("SELECT jid FROM sources WHERE status!='prepared' LIMIT 1").fetchall()
    assert not db.execute('SELECT sid FROM pending LIMIT 1').fetchall()
    expected = collections.defaultdict(set)
    for sid, chunk in db.execute("SELECT sid,chunk FROM coverage WHERE status='caption_ready'"):
        expected[chunk].add(sid)
    totals = collections.Counter()
    statuses = collections.Counter()
    nodes = collections.Counter()
    combined = hashlib.sha256()
    for index, (jid, n) in enumerate(db.execute('SELECT jid,n FROM chunks ORDER BY jid')):
        job = read(W / 'jobs' / (jid + '.json'))
        r = read(W / 'extract/completed' / (jid + '.json'))
        assert job['contract'] == r['contract'] == contract()
        assert sha(job['path']) == job['sha256']
        assert sha(r['raw_file']) == r['raw_sha256']
        assert sha(r['statistics_file']) == r['statistics_sha256']
        ids = []
        with Path(job['path']).open() as fi, Path(r['raw_file']).open() as fr, Path(r['statistics_file']).open('rb') as fs:
            for ordinal, triple in enumerate(itertools.zip_longest(fi, fr, fs)):
                a, b, c = triple
                assert a is not None and b is not None and c is not None
                request, raw, stat = json.loads(a), json.loads(b), json.loads(c)
                snap = request['snapshot']; sid = request['sample_id']; ids.append(sid)
                assert sid == raw['sample_id'] == stat['sample_id']
                assert digest(snap) == request['snapshot_sha256'] == raw['snapshot_sha256'] == stat['snapshot_sha256']
                for key in ('caption', 'source', 'caption_file_sha256', 'crop_provenance'):
                    assert stat.get(key) == snap.get(key), (sid, key)
                assert stat['extraction_record_index'] == ordinal
                assert stat['extraction_file'] == r['raw_file']
                assert raw['attempts'] and all(x['finish_reason'] == 'stop' for x in raw['attempts'])
                assert stat['human_ground_truth'] is False and stat['training_admitted'] is False
                assert stat['zero_counted_types_is_verified_absence'] is False
                for quantity in stat['behavior_quantities']:
                    if quantity['basis'] == 'unknown':
                        assert quantity['count'] is None
                if ordinal in (0, n - 1):
                    replay = statistic_record(request, raw['attempts'], r['raw_file'])
                    replay['extraction_record_index'] = ordinal
                    assert replay == stat, ('replay', sid)
                    totals['replayed_samples'] += 1
                statuses[stat['record_status']] += 1
                combined.update(c)
        assert len(ids) == n == r['clips'] and len(set(ids)) == n
        assert set(ids) == expected.pop(jid)
        assert digest(ids) == r['input_ids_sha256']
        for k in ('clips', 'strict_failed', 'incomplete_generations', 'unresolved'):
            totals[k] += r[k]
        nodes[r['node']] += n
        totals['chunks'] += 1
        if index % 100 == 0:
            print(json.dumps({'chunks': index + 1, 'clips': totals['clips'], 'seconds': time.time() - started}), flush=True)
    assert not expected
    print(json.dumps({'stage': 'chunk_audit_complete_waiting_for_merge', 'at': time.time()}), flush=True)
    while not (W / 'complete.json').exists():
        state = read(W / 'chain.json')
        assert state['stage'] != 'error', state
        assert time.time() - started < 3600, 'Merge did not complete within audit observation window'
        time.sleep(5)
    complete = read(W / 'complete.json')
    summary = read(W / 'tables/summary.json')
    db.execute('ATTACH DATABASE ? AS output', (str(W / 'tables/sample_stats.sqlite'),))
    assert db.execute('PRAGMA output.quick_check').fetchone() == ('ok',)
    assert not db.execute("SELECT c.sid FROM coverage c LEFT JOIN output.samples s ON c.sid=s.sid WHERE c.status='caption_ready' AND s.sid IS NULL LIMIT 1").fetchall()
    assert not db.execute("SELECT s.sid FROM output.samples s LEFT JOIN coverage c ON c.sid=s.sid WHERE c.sid IS NULL OR c.status!='caption_ready' LIMIT 1").fetchall()
    assert totals['clips'] == coverage['caption_ready'] == summary['sample_count'] == complete['sample_count'] == q['completed_clips']
    assert combined.hexdigest() == sha(W / 'tables/sample_stats.jsonl')
    assert summary['coverage'] == coverage
    assert not summary['source_errors'] and summary['unresolved_extractions'] == totals['unresolved'] == 0
    assert totals['incomplete_generations'] == 0
    assert dict(db.execute('SELECT status,COUNT(*) FROM output.samples GROUP BY status')) == dict(statuses)
    result = dict(passed=True, started_at=started, finished_at=time.time(), coverage=coverage,
                  totals=dict(totals), statuses=dict(statuses), by_node=dict(nodes),
                  combined_statistics_sha256=combined.hexdigest(), full_id_and_hash_audit=True,
                  replay_scope='first_and_last_record_of_every_chunk',
                  human_accuracy_verified=False, input_media_rescanned=False)
    (W / 'goal_monitor_20261031/final_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
