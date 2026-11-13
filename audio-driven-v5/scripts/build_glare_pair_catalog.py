#!/usr/bin/env python3
"""Index published reaction clips for browsing; preserve missing GLARE pairing metadata."""
from pathlib import Path
import sqlite3,tarfile,zstandard as z,zlib,json,time,re,os
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'evaluation/glare_pair_review';WORK.mkdir(exist_ok=True)
SOURCE=Path('/public/0_DATA/2_Avatar/GLARE_reaction_annotation_results')
DB=WORK/'catalog.sqlite';TMP=WORK/'catalog.building.sqlite'
if TMP.exists():raise SystemExit('Existing build database: inspect its live process before restarting')
c=sqlite3.connect(TMP);c.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE clips(dataset TEXT,id TEXT,source_id TEXT,participant TEXT,start REAL,end REAL,frames INT,PRIMARY KEY(dataset,id)); CREATE TABLE annotations(dataset TEXT,id TEXT,kind TEXT,body BLOB,PRIMARY KEY(dataset,id,kind));')
order=['nodding','head_shaking','smiling','laughing','frowning','surprised']
for kind in order:
 for dataset in ['seamless','realtalk']:
  arc=SOURCE/'reaction_detector'/kind/f'output_{dataset}.tar.zst';count=0
  with arc.open('rb') as raw,z.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) as reader,tarfile.open(fileobj=reader,mode='r|') as tf:
   for m in tf:
    if not m.isfile() or not m.name.endswith('.csv'):continue
    name=m.name.removeprefix('./');clip=name[:-4];body=tf.extractfile(m).read();frames=len(body.splitlines())-1
    c.execute('INSERT INTO annotations VALUES(?,?,?,?)',(dataset,clip,kind,zlib.compress(body,1)))
    if kind=='nodding':
     if dataset=='seamless':
      match=re.fullmatch(r'(V\d+_S\d+_I\d+)_(P[^_]+)_([\d.]+)_([\d.]+)',clip)
      source,person,a,b=match.groups() if match else (clip,None,None,None)
     else:source=clip.split('/')[0];person=a=b=None
     c.execute('INSERT INTO clips VALUES(?,?,?,?,?,?,?)',(dataset,clip,source,person,float(a) if a is not None else None,float(b) if b is not None else None,frames))
    count+=1
    if count%5000==0:(WORK/'progress.json').write_text(json.dumps(dict(kind=kind,dataset=dataset,files=count,updated=time.time())))
  c.commit();print(dataset,kind,count,flush=True)
c.execute('CREATE INDEX clips_source ON clips(source_id)');c.commit();total=c.execute('SELECT count(*) FROM clips').fetchone()[0];c.close();os.replace(TMP,DB)
(WORK/'progress.json').write_text(json.dumps(dict(status='complete',clips=total,updated=time.time())))
print('COMPLETE',total,flush=True)
