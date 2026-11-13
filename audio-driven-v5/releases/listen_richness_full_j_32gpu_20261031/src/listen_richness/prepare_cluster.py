"""Bounded, resumable input packing from the Omni source journal, not a file scan."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib,json,os,sqlite3,time
from pathlib import Path
from .common import read,digest
from .statistics import snapshot
from .production import tokenized,contract
from listen_benchmark.j_annotation import write,acquire,result_path


def source_records(upstream,jid):
    upstream=Path(upstream);job=read(upstream/'jobs'/f'{jid}.json')
    receipt=read(upstream/'caption/completed'/f'{jid}.json')
    if receipt['job_id']!=jid or receipt['clips']!=len(job['rows']):raise ValueError('Source receipt count mismatch')
    cfg=read(upstream/'config.json');gate=cfg['visual_quality_gate']
    qr=read(Path(gate['work'])/'measure/completed'/f'{jid}.json')
    if qr['contract_sha256']!=gate['contract_sha256'] or qr['source_result_sha256']!=job['source_result_sha256']:
        raise ValueError('Quality/source contract mismatch')
    decisions={d['sample_id']:d for d in qr['decisions']};out=[]
    for row in job['rows']:
        sid=row['sample_id'];branches=[b for b in ('caption','held','visual_excluded','visual_review') if result_path(upstream,b,sid).exists()]
        if len(branches)!=1:raise ValueError('Missing/duplicate terminal branch: '+sid)
        branch=branches[0];p=result_path(upstream,branch,sid)
        base=dict(sample_id=sid,job_id=jid,source=row['source'],branch=branch,terminal_file=str(p))
        if branch=='caption':
            raw=p.read_bytes();record=json.loads(raw)
            if record.get('original_source',record['source'])!=row['source']:raise ValueError('Caption/index provenance differs')
            try:
                s=snapshot(record,p,hashlib.sha256(raw).hexdigest(),decisions.get(sid),sample_grid_tolerance=True)
                if not s['caption'].strip():raise ValueError('Empty caption, not a zero-behavior example')
                base.update(status='caption_ready',snapshot=s)
            except Exception as exc:
                base.update(status='input_review',error=repr(exc),caption=record.get('caption'),caption_file_sha256=hashlib.sha256(raw).hexdigest())
        else:base['status']=branch
        out.append(base)
    if len({r['sample_id'] for r in out})!=len(out):raise ValueError('Duplicate source samples')
    return out


def prepared_source(upstream, jid, tokenizer):
    """Overlap immutable source reads and CPU tokenization in the bounded pool."""
    records=source_records(upstream,jid)
    for base in records:
        if 'snapshot' not in base:continue
        try:base['_prepared_request']=tokenized(base['snapshot'],tokenizer)
        except Exception as exc:base['_token_error']=repr(exc)
    return records


def database(work):
    db=sqlite3.connect(Path(work)/'inputs.sqlite',timeout=60)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
    CREATE TABLE IF NOT EXISTS sources(jid TEXT PRIMARY KEY,status TEXT,attempts INTEGER DEFAULT 0,error TEXT);
    CREATE TABLE IF NOT EXISTS coverage(sid TEXT PRIMARY KEY,jid TEXT,status TEXT,chunk TEXT,record TEXT);
    CREATE INDEX IF NOT EXISTS coverage_jid ON coverage(jid);
    CREATE TABLE IF NOT EXISTS pending(sid TEXT PRIMARY KEY,weight INTEGER,data TEXT);
    CREATE INDEX IF NOT EXISTS pending_weight ON pending(weight DESC,sid);
    CREATE TABLE IF NOT EXISTS chunks(jid TEXT PRIMARY KEY,published INTEGER DEFAULT 0,n INTEGER);''')
    return db


def ingest(db,upstream):
    value=db.execute("SELECT v FROM meta WHERE k='offset'").fetchone();offset=int(value[0]) if value else 0
    with (Path(upstream)/'ready.jsonl').open() as f:
        f.seek(offset)
        for _ in range(4096):
            line=f.readline()
            if not line or not line.endswith('\n'):break
            jid=json.loads(line)['job_id'];db.execute("INSERT OR IGNORE INTO sources(jid,status) VALUES (?,'pending')",(jid,));offset=f.tell()
    db.execute("INSERT OR REPLACE INTO meta VALUES ('offset',?)",(str(offset),));db.commit()


def publish(db,work):
    work=Path(work)
    for (jid,) in db.execute('SELECT jid FROM chunks WHERE published=0').fetchall():
        # Duplicate journal lines after a crash are harmless to JobQueue.claim.
        with (work/'ready.jsonl').open('a') as f:
            f.write(json.dumps(dict(job_id=jid))+'\n');f.flush();os.fsync(f.fileno())
        db.execute('UPDATE chunks SET published=1 WHERE jid=?',(jid,));db.commit()


def flush(db,work,size,force=False):
    work=Path(work)
    while True:
        pending=db.execute('SELECT sid,data FROM pending ORDER BY weight DESC,sid LIMIT ?',(size,)).fetchall()
        if not pending or (len(pending)<size and not force):break
        data=''.join(x[1]+'\n' for x in pending);sha=hashlib.sha256(data.encode()).hexdigest();jid='chunk-'+sha[:24]
        target=work/'inputs'/f'{jid}.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise ValueError('Immutable packed input changed')
        else:
            tmp=target.with_suffix('.tmp');tmp.write_text(data);tmp.replace(target)
        write(work/'jobs'/f'{jid}.json',dict(job_id=jid,path=str(target),sha256=sha,count=len(pending),contract=contract()))
        with db:
            db.execute('INSERT OR IGNORE INTO chunks(jid,n) VALUES (?,?)',(jid,len(pending)))
            for sid,_ in pending:
                db.execute('UPDATE coverage SET chunk=? WHERE sid=?',(jid,sid));db.execute('DELETE FROM pending WHERE sid=?',(sid,))
        publish(db,work)


def pack_source(db,records,tokenizer,commit=True):
    with db if commit else nullcontext():
        for base in records:
            sid=base['sample_id']
            if db.execute('SELECT 1 FROM coverage WHERE sid=?',(sid,)).fetchone():
                raise ValueError('Duplicate sample in source journal: '+sid)
            snap=base.pop('snapshot',None);request=base.pop('_prepared_request',None);token_error=base.pop('_token_error',None)
            if token_error is not None:
                base.update(status='input_review',error=token_error,caption=snap['caption'])
            elif snap is not None and request is None:
                try:request=tokenized(snap,tokenizer)
                except Exception as exc:base.update(status='input_review',error=repr(exc),caption=snap['caption'])
            db.execute('INSERT INTO coverage VALUES (?,?,?,?,?)',(sid,base['job_id'],base['status'],None,json.dumps(base,ensure_ascii=False)))
            if request is not None:
                db.execute('INSERT INTO pending VALUES (?,?,?)',(sid,request['prompt_tokens'],json.dumps(request,ensure_ascii=False)))
        db.execute("UPDATE sources SET status='prepared',error=NULL WHERE jid=?",(records[0]['job_id'],))


def prepare(work,once=False):
    work=Path(work);cfg=read(work/'config.json');upstream=Path(cfg['upstream'])
    lock=acquire(work/'prepare.lock',False)
    if lock is None:return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True)
    db=database(work);count=0
    try:
        with ThreadPoolExecutor(cfg.get('prepare_workers',4)) as pool:
            while True:
                ingest(db,upstream);publish(db,work)
                # Only inspect known indexed receipts, with bounded concurrency.
                candidates=[jid for (jid,) in db.execute("SELECT jid FROM sources WHERE status='pending' OR (status='error' AND attempts<3) LIMIT 4096")]
                ready=[jid for jid in candidates if (upstream/'caption/completed'/f'{jid}.json').exists()][:cfg.get('prepare_workers',4)*4]
                futures=[(jid,pool.submit(prepared_source,upstream,jid,tokenizer)) for jid in ready]
                with db:
                    # Amortize shared-filesystem WAL commits across sources;
                    # savepoints still isolate any individual failed source.
                    db.execute('BEGIN')
                    for jid,f in futures:
                        db.execute('SAVEPOINT one_source')
                        try:
                            records=f.result()
                            if records:pack_source(db,records,tokenizer,commit=False)
                            else:db.execute("UPDATE sources SET status='prepared' WHERE jid=?",(jid,))
                            db.execute('RELEASE one_source')
                        except Exception as exc:
                            db.execute('ROLLBACK TO one_source');db.execute('RELEASE one_source')
                            db.execute("UPDATE sources SET status='error',attempts=attempts+1,error=? WHERE jid=?",(repr(exc),jid))
                        count+=1
                flush(db,work,cfg.get('chunk_size',512),force=not ready)
                remaining=db.execute("SELECT COUNT(*) FROM sources WHERE status!='prepared'").fetchone()[0]
                finishing=(upstream/'producer_complete.json').exists() and remaining==0
                if not ready or count%32==0 or finishing:
                    totals=dict(db.execute('SELECT status,COUNT(*) FROM coverage GROUP BY status'))
                    errors=db.execute("SELECT jid,error FROM sources WHERE status='error' AND attempts>=3").fetchall()
                    state=dict(at=time.time(),pid=os.getpid(),coverage=totals,source_errors=errors,remaining_sources=remaining,
                               packed_chunks=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0])
                    write(work/'preparation.json',state)
                offset=int(db.execute("SELECT v FROM meta WHERE k='offset'").fetchone()[0])
                if (upstream/'producer_complete.json').exists() and offset>=(upstream/'ready.jsonl').stat().st_size and remaining==0:
                    flush(db,work,cfg.get('chunk_size',512),True)
                    state['packed_chunks']=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
                    write(work/'producer_complete.json',state);break
                if once:break
                if not ready:time.sleep(10)
    finally:db.close();lock.close()
