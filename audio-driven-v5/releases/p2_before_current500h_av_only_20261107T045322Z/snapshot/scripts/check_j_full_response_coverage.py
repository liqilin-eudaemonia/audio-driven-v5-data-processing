"""Independent set/metadata reconciliation for the full response-text export."""
import argparse
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);args=p.parse_args()
    out=args.data;summary=json.loads((out/'summary.json').read_text())
    assert summary['scope']=='full'
    db=sqlite3.connect(f'file:{out}/index.sqlite?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('ATTACH DATABASE ? AS source',('file:'+summary['source_selection_db']+'?mode=ro',))
    db.execute('ATTACH DATABASE ? AS old',(f'file:{ROOT}/experiments/listen_j_smallface_response_sft_20261101/index.sqlite?mode=ro',))
    # Unique IDs are enforced by both independent SQLite primary/unique keys.
    extra=db.execute('SELECT COUNT(*) FROM main.samples m LEFT JOIN source.samples s ON s.sid=m.sid WHERE s.sid IS NULL').fetchone()[0]
    missing=db.execute('SELECT COUNT(*) FROM source.samples s LEFT JOIN main.samples m ON s.sid=m.sid WHERE m.sid IS NULL').fetchone()[0]
    mismatched=db.execute('''SELECT COUNT(*) FROM main.samples m JOIN source.samples s ON s.sid=m.sid
        WHERE m.split!=s.split OR m.pair!=s.pair OR m.listener!=s.listener OR m.group_id!=s.group_id
        OR m.start!=s.start OR m.end!=s.end OR m.duration!=s.duration OR m.face_p10!=s.face_p10''').fetchone()[0]
    old_missing=db.execute('SELECT COUNT(*) FROM old.samples o LEFT JOIN main.samples m ON m.sid=o.sid WHERE m.sid IS NULL').fetchone()[0]
    assert extra==missing==mismatched==old_missing==0
    supplement=ROOT/'experiments/listen_j_final_quality_selection_20261101/train500h_audio_supplement_feasibility/supplement_trial.jsonl'
    ids=[json.loads(x)['sid'] for x in supplement.open()]
    db.execute('CREATE TEMP TABLE supplement(sid TEXT PRIMARY KEY)');db.executemany('INSERT INTO supplement VALUES(?)',[(x,) for x in ids])
    counts={}
    for name,query in [
        ('old_smallface', 'SELECT COUNT(*),SUM(duration)/3600 FROM old.samples'),
        ('train_bigface_500h_base', "SELECT COUNT(*),SUM(duration)/3600 FROM main.samples WHERE split='train' AND face_p10>=224"),
        ('train_500h_audio_supplement', "SELECT COUNT(*),SUM(m.duration)/3600 FROM main.samples m JOIN supplement s ON s.sid=m.sid WHERE m.split='train'"),
        ('dev_test_bigface', "SELECT COUNT(*),SUM(duration)/3600 FROM main.samples WHERE split IN ('dev','test') AND face_p10>=224")]:
        c,h=db.execute(query).fetchone();counts[name]=dict(clips=c,hours=h)
    total=db.execute('SELECT COUNT(*),SUM(duration)/3600 FROM main.samples').fetchone()
    assert counts['train_bigface_500h_base']['clips']==264054
    assert counts['train_500h_audio_supplement']['clips']==21584
    assert counts['dev_test_bigface']['clips']==7959
    assert sum(r['clips'] for r in counts.values())==total[0]==1107323
    assert abs(sum(r['hours'] for r in counts.values())-total[1])<1e-6
    cross_split=db.execute('SELECT COUNT(*) FROM (SELECT group_id FROM main.samples GROUP BY group_id HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
    assert cross_split==0
    report=dict(passed=True,scope='all_final_quality_retained_with_caption',total_clips=total[0],total_hours=total[1],
        extra_ids=extra,missing_ids=missing,source_metadata_mismatches=mismatched,old_smallface_missing=old_missing,
        original_split_group_intersections=cross_split,composition=counts,physical_media_all_decoded=False)
    (out/'coverage_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
