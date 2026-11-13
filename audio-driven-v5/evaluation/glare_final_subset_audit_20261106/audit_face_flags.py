"""Stream PKL archives to count detection flags; retain only tiny summary records."""
from pathlib import Path
import sqlite3,pickle,time,json,tarfile,zstandard as z,argparse
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SRC=Path('/public/0_DATA/2_Avatar/GLARE_reaction_annotation_results/nod_headshake/face_detection')
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--limit',type=int,default=500);a=p.parse_args();ds=a.dataset
with sqlite3.connect('file:'+str(ROOT/'evaluation/glare_pair_review/catalog.sqlite')+'?mode=ro',uri=True) as c:known={i:n for i,n in c.execute('SELECT id,frames FROM clips WHERE dataset=?',(ds,))}
with sqlite3.connect('file:'+str(OUT/'fer_stats.sqlite')+'?mode=ro',uri=True) as c:complete={i for i, in c.execute('SELECT id FROM fer WHERE dataset=? AND rows=reaction_frames AND consecutive_zero_based=1',(ds,))}
def fast(b):
 key=b'\x8c\x08detected\x94';i=b.find(key);flag=b[i+len(key)] if i>=0 else 0
 if flag not in [0x88,0x89]:return None
 true=b.count(b'}\x94(h\x02\x88h\x03')+(flag==0x88);false=b.count(b'}\x94(h\x02\x89h\x03')+(flag==0x89);return true+false,true
out=sqlite3.connect(OUT/f'face_flags_{ds}.sqlite');out.execute('CREATE TABLE IF NOT EXISTS faces(id TEXT PRIMARY KEY,frames INTEGER,detected INTEGER,in_reactions INTEGER,fer_complete INTEGER,method TEXT)');counts={};start=time.time();checks=0;fallback=0;members=0;good=0;badfast=0;frames=0;goodframes=0
arc=SRC/f'output_{ds}.tar.zst'
with arc.open('rb') as f,z.ZstdDecompressor(max_window_size=2**31).stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as tf:
 for m in tf:
  if not m.isfile() or not m.name.endswith('.pkl'):continue
  members+=1;ident=m.name.removeprefix('./').removesuffix('.pkl');b=tf.extractfile(m).read();result=fast(b);expected=known.get(ident);method='fast_validated_count'
  # Validate every sample in the first 100 and every 500th later; use full parser on any mismatch.
  verify=(members<=100 or members%500==0 or result is None or expected is None or result[0]!=expected)
  if verify:
   data=pickle.loads(b);truth=(len(data),sum(isinstance(x,dict) and (x.get('detected') is True or x.get('landmarks') is not None) for x in data));checks+=1
   if truth!=result:badfast+=1
   if expected is None or result!=truth or (expected is not None and result[0]!=expected):fallback+=1;result=truth;method='pickle_loaded'
  n,detected=result;kept=ident in known and ident in complete and detected==n and n==expected
  if ident in known:frames+=expected
  if kept:good+=1;goodframes+=expected
  out.execute('INSERT OR REPLACE INTO faces VALUES(?,?,?,?,?,?)',(ident,n,detected,int(ident in known),int(ident in complete),method))
  if members%100==0:
   out.commit();report=dict(dataset=ds,members=members,compressed_bytes=f.tell(),archive_bytes=arc.stat().st_size,seconds=time.time()-start,full_parser_checks=checks,fast_mismatches=badfast,full_parser_used=fallback,joint_valid=good,joint_valid_hours=goodframes/90000,matched_reaction_hours=frames/90000,scope='full archive' if not a.limit else f'leading {a.limit} PKLs; nonrandom sample');(OUT/f'face_flags_{ds}_progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
  if a.limit and members>=a.limit:break
out.commit();out.close();report.update(complete=True,members=members,seconds=time.time()-start);(OUT/f'face_flags_{ds}_result.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)
