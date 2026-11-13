"""Independent full-range SQL audit of the current frozen interval database."""
import argparse,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_ltx_events.config import DEFAULT,load
from p2_listen.contracts import read,sha,atomic_json


def audit(c):
    start=time.monotonic();labels=Path(c['temporal_labels']);source=Path(c['current_source'])
    lm=read(labels/'manifest.json');sm=read(source/'manifest.json')
    label_sha=sha(labels/'manifest.json');source_sha=sha(source/'manifest.json')
    if sha(labels/'labels.sqlite')!=lm['database_sha256'] or sha(source/'sources.sqlite')!=sm['database_sha256']:
        raise ValueError('Audit input database changed')
    db=sqlite3.connect((labels/'labels.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.execute('ATTACH DATABASE ? AS src',(str((source/'sources.sqlite').resolve()),))
    try:
        joined=db.execute('SELECT COUNT(*) FROM labels l JOIN src.samples s ON l.ordinal=s.ordinal AND l.sid=s.sid').fetchone()[0]
        silent=db.execute("SELECT l.sid FROM labels l JOIN src.samples s ON l.ordinal=s.ordinal WHERE json_extract(s.value,'$.listener_audio_identity.full_silence')=1 AND json_array_length(json_extract(l.value,'$.classes.listener_vocalization.spans'))>0").fetchall()
        duration=db.execute("SELECT COUNT(*) FROM labels l JOIN src.samples s ON l.ordinal=s.ordinal WHERE abs(json_extract(l.value,'$.duration_s')-json_extract(s.value,'$.source.duration_s'))>0.000000001").fetchone()[0]
        missing_silence=db.execute("SELECT COUNT(*) FROM src.samples WHERE json_type(value,'$.listener_audio_identity.full_silence') NOT IN ('true','false') OR json_type(value,'$.listener_audio_identity.full_silence') IS NULL").fetchone()[0]
        invalid={}
        for field in ('spans','coverage','unknown'):
            query="SELECT count(*) FROM labels l,json_each(l.value,'$.classes') cls,json_each(cls.value,?) i WHERE json_type(i.value,'$[0]') NOT IN ('integer','real') OR json_type(i.value,'$[1]') NOT IN ('integer','real') OR json_extract(i.value,'$[0]') < -0.000001 OR json_extract(i.value,'$[1]') <= json_extract(i.value,'$[0]') OR json_extract(i.value,'$[1]') > json_extract(l.value,'$.duration_s')+0.000001"
            invalid[field]=db.execute(query,('$.'+field,)).fetchone()[0]
        result=dict(success=joined==lm['samples']==sm['samples'] and not silent and not duration and not missing_silence and not any(invalid.values()),
            samples=joined,exact_ordinal_and_id_join=True,invalid_intervals=invalid,physical_silence_conflicts=len(silent),
            conflict_ids=[x[0] for x in silent],native_duration_mismatches=duration,missing_physical_silence_flags=missing_silence,
            labels=str(labels),labels_manifest_sha256=label_sha,source_manifest_sha256=source_sha,
            labels_database_sha256=lm['database_sha256'],source_database_sha256=sm['database_sha256'],
            checker_sha256=sha(__file__),seconds=time.monotonic()-start)
        if label_sha!=sha(labels/'manifest.json') or source_sha!=sha(source/'manifest.json'):
            raise ValueError('Audit manifests changed during the check')
        atomic_json(Path(c['output'])/'INDEPENDENT_INTERVAL_AUDIT.json',result)
        if not result['success']:raise ValueError('Independent interval audit failed: '+json.dumps(result))
        return result
    finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT)
    a=p.parse_args();print(json.dumps(audit(load(a.config)),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
