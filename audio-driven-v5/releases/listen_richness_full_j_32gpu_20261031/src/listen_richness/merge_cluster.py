"""Resumable SQLite aggregation; O(N log N) interval audit, no pairwise scan."""
import csv,hashlib,heapq,json,sqlite3,time
from pathlib import Path
from .common import read
from listen_benchmark.j_annotation import write,acquire


def merge(work):
    work=Path(work);lock=acquire(work/'merge.lock',False)
    if lock is None:return
    out=work/'tables';out.mkdir(exist_ok=True)
    inputs=sqlite3.connect(f'file:{work}/inputs.sqlite?mode=ro',uri=True)
    db=sqlite3.connect(out/'sample_stats.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS chunks(jid TEXT PRIMARY KEY,sha TEXT);
    CREATE TABLE IF NOT EXISTS samples(sid TEXT PRIMARY KEY,pair TEXT,listener TEXT,split TEXT,group_id TEXT,
      start REAL,end REAL,duration REAL,face_p10 REAL,face_p50 REAL,types INTEGER,status TEXT,file TEXT,offset INTEGER);
    CREATE INDEX IF NOT EXISTS source_time ON samples(pair,listener,start,end);
    CREATE TABLE IF NOT EXISTS behaviors(sid TEXT,type TEXT,pair TEXT,split TEXT,counted INTEGER,pending INTEGER,
      quantity INTEGER,basis TEXT,unit TEXT,state INTEGER,change INTEGER,duration REAL,PRIMARY KEY(sid,type));
    CREATE INDEX IF NOT EXISTS behavior_type ON behaviors(type,split);''')
    try:
        for (jid,) in inputs.execute('SELECT jid FROM chunks ORDER BY jid'):
            receipt=read(work/'extract/completed'/f'{jid}.json');path=Path(receipt['statistics_file'])
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha!=receipt['statistics_sha256']:raise ValueError('Statistics changed: '+jid)
            previous=db.execute('SELECT sha FROM chunks WHERE jid=?',(jid,)).fetchone()
            if previous:
                if previous[0]!=sha:raise ValueError('Merged chunk changed')
                continue
            count=0
            with db,path.open('rb') as f:
                while True:
                    offset=f.tell();line=f.readline()
                    if not line:break
                    r=json.loads(line);sid=r['sample_id'];s=r['source'];count+=1
                    db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (sid,r['pair'],r['listener_id'],r['split'],s['group_id'],r['source_start_s'],r['source_end_s'],r['duration_s'],
                         r['face_short_side_p10'],r['face_short_side_p50'],r['counted_type_count'],r['record_status'],str(path),offset))
                    groups={g['key']:g for g in r['groups']}
                    pending={b['raw']['modality']+':'+b['canonical'] for b in r['evidence_ledger'] if b['disposition']=='pending'}
                    for key in set(groups)|pending:
                        g=groups.get(key);q=g['quantity'] if g else {}
                        db.execute('INSERT INTO behaviors VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                            (sid,key,r['pair'],r['split'],int(g is not None),int(key in pending),q.get('count'),q.get('basis'),q.get('unit'),
                             g['state_group_count'] if g else 0,int('change' in g['forms']) if g else 0,r['duration_s'] if g else 0))
                if count!=receipt['clips']:raise ValueError('Chunk row count differs')
                db.execute('INSERT INTO chunks VALUES (?,?)',(jid,sha))
        expected=inputs.execute("SELECT COUNT(*) FROM coverage WHERE status='caption_ready'").fetchone()[0]
        actual=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        if expected!=actual:raise ValueError(f'Missing statistics: {actual}/{expected}')
        catalog=[]
        query='''SELECT type,SUM(counted),COUNT(DISTINCT CASE WHEN counted=1 THEN pair END),
        SUM(CASE WHEN unit='occurrence' AND basis='literal' THEN quantity END),
        SUM(CASE WHEN unit='occurrence' AND basis='derived_from_distinct_mentions' THEN quantity END),
        SUM(CASE WHEN unit='occurrence' AND quantity IS NOT NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN unit='occurrence' AND quantity IS NULL THEN 1 ELSE 0 END),
        SUM(state),SUM(pending),SUM(duration),SUM(change) FROM behaviors GROUP BY type ORDER BY type'''
        fields=['type','sample_count','pair_count','literal_occurrences_sum','derived_occurrences_sum',
                'known_occurrence_samples','unknown_occurrence_samples','state_groups','pending_samples','covered_clip_seconds','change_samples']
        for values in db.execute(query):
            c=dict(zip(fields,values));c['supported_occurrences_sum']=(sum(c[k] or 0 for k in ('literal_occurrences_sum','derived_occurrences_sum')) if c['known_occurrence_samples'] else None);catalog.append(c)
        with (out/'behavior_catalog.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields+['supported_occurrences_sum']);writer.writeheader();writer.writerows(catalog)
        # One linear pass over sorted intervals, with an active-end heap. Count
        # overlap pairs exactly without materializing O(N²) pair lists.
        group=None;active=[];overlaps=0
        for pair,listener,start,end in db.execute('SELECT pair,listener,start,end FROM samples ORDER BY pair,listener,start,end'):
            if (pair,listener)!=group:group=(pair,listener);active=[]
            while active and active[0]<=start:heapq.heappop(active)
            overlaps+=len(active);heapq.heappush(active,end)
        columns=[r[1] for r in db.execute('PRAGMA table_info(samples)')]
        with (out/'sample_stats.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(columns);writer.writerows(db.execute('SELECT * FROM samples ORDER BY sid'))
        # Full per-clip evidence remains in chunk files, also exported as one
        # stream for training/filter consumers. A partial export is never final.
        temp=out/'sample_stats.jsonl.tmp'
        with temp.open('wb') as target:
            for (jid,) in inputs.execute('SELECT jid FROM chunks ORDER BY jid'):
                receipt=read(work/'extract/completed'/f'{jid}.json')
                with Path(receipt['statistics_file']).open('rb') as f:
                    while block:=f.read(1024*1024):target.write(block)
        temp.replace(out/'sample_stats.jsonl')
        coverage=dict(inputs.execute('SELECT status,COUNT(*) FROM coverage GROUP BY status'))
        if (work/'config.json').exists():
            cfg=read(work/'config.json')
            expected_sources=cfg.get('expected_upstream_sources')
            if expected_sources is not None and inputs.execute('SELECT COUNT(*) FROM sources').fetchone()[0]!=expected_sources:
                raise ValueError('Full source coverage differs from the authoritative scope')
            if cfg.get('expected_upstream_clips') is not None and sum(coverage.values())!=cfg['expected_upstream_clips']:
                raise ValueError('Full sample coverage differs from the authoritative scope')
        with (out/'coverage.jsonl').open('w') as f:
            for (record,) in inputs.execute('SELECT record FROM coverage ORDER BY sid'):f.write(record+'\n')
        source_errors=inputs.execute("SELECT jid,error FROM sources WHERE status!='prepared'").fetchall()
        unresolved=db.execute("SELECT COUNT(*) FROM samples WHERE status='unresolved_extraction'").fetchone()[0]
        summary=dict(at=time.time(),sample_count=actual,total_indexed_samples=sum(coverage.values()),coverage=coverage,
                     source_errors=source_errors,unresolved_extractions=unresolved,catalog=catalog,
                     distinct_candidate_types=sum(c['sample_count']>0 for c in catalog),source_overlap_pairs=overlaps,
                     split_counts=dict(db.execute('SELECT split,COUNT(*) FROM samples GROUP BY split')),
                     caption_hours=db.execute('SELECT SUM(duration)/3600 FROM samples').fetchone()[0],
                     count_scope='sum_across_clips_not_unique_source_events',exact_event_total=None,
                     covered_time_is_clip_time_not_event_duration=True,selection_applied=False,human_ground_truth=False,
                     unresolved_is_not_zero=True,all_source_records_retained=True)
        write(out/'summary.json',summary)
        write(work/'complete.json',dict(at=time.time(),sample_count=actual,total_indexed_samples=sum(coverage.values()),
             status='complete_with_review' if unresolved or coverage.get('input_review') or source_errors else 'complete',summary=str(out/'summary.json')))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return summary
    finally:inputs.close();db.close();lock.close()
