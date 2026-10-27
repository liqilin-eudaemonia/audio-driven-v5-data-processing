"""Build a real-source 100-clip regression view without occupying training GPUs."""
import json,sys,sqlite3,hashlib,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json
from listen_audio_full.indexing import connect
from listen_audio_full.preparation import init_view,prepare_archive
from listen_audio_postprocess.recipes import read_listener_audio
OUT=Path(__file__).resolve().parent
BASELINE=ROOT/'experiments/p2_train50h_audio_postprocess_20261026'

def manifest():
 files=[ROOT/'scripts/p2_full_audio_postprocess.py',ROOT/'scripts/compare_listener_official_six.py']
 for folder in ('listen_audio_full','listen_audio_postprocess','p2_listen','p3_listen','listen_benchmark'):
  files+=list((ROOT/'src'/folder).glob('*.py'))
 files.append(ROOT/'experiments/p2_e1_feature01_train50h_20261026/queue_known_hosts')
 return {str(p):sha(p) for p in sorted(set(files))}


def main():
 run=OUT/'validation100';run.mkdir(exist_ok=True)
 source=connect(OUT);source.row_factory=sqlite3.Row
 ids=[r['sample_id'] for r in read(BASELINE/'gpu100_acceptance.json')['records']]
 dbpath=run/'scope.sqlite';dbpath.unlink(missing_ok=True);db=sqlite3.connect(dbpath)
 for table in ('samples','archives'):
  sql=source.execute('SELECT sql FROM sqlite_master WHERE type="table" AND name=?',(table,)).fetchone()[0];db.execute(sql)
 for sid in ids:
  row=source.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone()
  if row is None:raise ValueError('Regression ID absent from full scope')
  db.execute('INSERT INTO samples VALUES('+','.join('?'*len(row))+')',tuple(row))
 for archive,count in db.execute('SELECT archive,count(*) FROM samples GROUP BY archive').fetchall():db.execute('INSERT INTO archives VALUES(?,?,?)',(digest(archive),archive,count))
 db.execute('CREATE INDEX by_archive ON samples(archive,member,ordinal)');db.commit();db.close();source.close()
 launch=read(OUT/'launch.json');launch.update(expected_samples=100,chunk_size=32,prepare_workers=8,validation_only=True)
 atomic_json(run/'launch.json',launch)
 atomic_json(run/'scope.json',dict(samples=100,source_full_scope_sha256=sha(OUT/'scope.json'),validation_only=True,sqlite_sha256=sha(dbpath)))
 atomic_json(run/'source_shards.json',dict(source_full_index_sha256=sha(OUT/'source_shards.json'),validation_only=True))
 atomic_json(run/'code_manifest.json',manifest())
 init_view(run);db=connect(run);keys=[r[0] for r in db.execute('SELECT key FROM archives')];db.close()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:reports=list(pool.map(lambda k:prepare_archive(run,k),keys))
 assert sum(r['samples'] for r in reports)==100
 comparisons=[]
 import numpy as np
 for sid in ids:
  a=read(run/'view/audio_postprocess/recipes'/f'{sid}.json');b=read(BASELINE/'view/audio_postprocess/recipes'/f'{sid}.json')
  wa,ra,_=read_listener_audio(a);wb,rb,_=read_listener_audio(b)
  assert ra==rb and np.array_equal(wa,wb),sid
  records=[read(p/'records'/f'{sid}.json') for p in (run,BASELINE)]
  assert records[0]['audio_events']==records[1]['audio_events'],sid
  old=read(BASELINE/'view/labels/canonical'/f'{sid}.json');new=read(run/'view/labels/canonical'/f'{sid}.json')
  def content(label):return [(e['type'],e['start'],e['end'],e['description'],e.get('text')) for e in label['events']]
  assert content(old)==content(new),sid
  assert new['provenance']['formal_training_admitted'] is False
  comparisons.append(dict(sample_id=sid,waveform_bit_exact=True,audio_events_exact=True,canonical_content_exact=True,admission_unchanged=True))
 result=dict(status='passed',samples=100,archives=len(keys),waveforms_bit_exact=100,canonical_contents_exact=100,training_started=False,gpu_used=False,records=comparisons)
 atomic_json(OUT/'validation100_cpu.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
