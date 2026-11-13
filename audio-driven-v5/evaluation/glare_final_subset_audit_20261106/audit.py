"""Read-only audits of released annotation archives; no changes to review data."""
from pathlib import Path
import sqlite3,json,time,argparse,tarfile,zstandard as z,zlib,collections,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DB=ROOT/'evaluation/glare_pair_review/catalog.sqlite';SRC=Path('/public/0_DATA/2_Avatar/GLARE_reaction_annotation_results')
p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','fer','reactions']);args=p.parse_args()
def db():return sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True)
def save(name,value):
 dest=OUT/(name+'.json');tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(dest)
def summary(rows):
 rows=list(rows);return dict(clips=len(rows),frames=sum(r[2] for r in rows),hours=sum(r[2] for r in rows)/90000)
if args.stage=='metadata':
 with db() as d:rows=d.execute('SELECT dataset,id,frames,source_id FROM clips').fetchall()
 report={'baseline':summary(rows),'per_dataset':{ds:summary(r for r in rows if r[0]==ds) for ds in ['realtalk','seamless']},'duration_thresholds':[], 'chunk_filters':[]}
 for frames in [60,75,100,120,125,130,135,150,180,185,240,250,260,300,360,385,420]:
  keep=[r for r in rows if r[2]>=frames];report['duration_thresholds'].append(dict(minimum_frames=frames,**summary(keep)))
 for seconds in [2,5,10]:
  window=round(seconds*25)+10+125;minchunks=(window-1)//60+1;keep=[r for r in rows if r[2]//60>=minchunks];eligible_windows=sum(r[2]//60-minchunks+1 for r in keep)
  report['chunk_filters'].append(dict(current_audio_seconds=seconds,total_window_frames=window,min_60frame_chunks=minchunks,minimum_clip_frames=minchunks*60,eligible_windows=eligible_windows,**summary(keep)))
 # Diagnostic only: varying one duration threshold cannot establish author choices.
 histogram=collections.Counter(r[2] for r in rows);n=len(rows);f=sum(r[2] for r in rows);grid=[]
 for cutoff in range(min(histogram),max(histogram)+1):
  grid.append(dict(minimum_frames=cutoff,clips=n,hours=f/90000))
  n-=histogram[cutoff];f-=histogram[cutoff]*cutoff
 report['closest_minimum_duration_by_count']=min(grid,key=lambda x:abs(x['clips']-107149));report['closest_minimum_duration_by_hours']=min(grid,key=lambda x:abs(x['hours']-147))
 save('metadata',report);print(json.dumps(report,indent=2),flush=True)
elif args.stage=='fer':
 with db() as d:known={(ds,ident):n for ds,ident,n in d.execute('SELECT dataset,id,frames FROM clips')}
 out=sqlite3.connect(OUT/'fer_stats.sqlite');out.execute('CREATE TABLE IF NOT EXISTS fer(dataset TEXT,id TEXT,rows INTEGER,reaction_frames INTEGER,consecutive_zero_based INTEGER,PRIMARY KEY(dataset,id))');reports={}
 for ds in ['realtalk','seamless']:
  t=time.time();counts=collections.Counter();examples={};arc=SRC/'fer_emotion'/f'output_{ds}.tar.zst'
  with arc.open('rb') as f,z.ZstdDecompressor(max_window_size=2**31).stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as tf:
   for m in tf:
    if not m.isfile() or not m.name.endswith('.csv'):continue
    ident=m.name.removeprefix('./').removesuffix('.csv').replace('/emo_scores/','/');b=tf.extractfile(m).read();lines=b.splitlines()[1:];lines=[x for x in lines if x.strip()];n=len(lines);expected=known.get((ds,ident));continuous=all(int(line.split(b',',1)[0])==i for i,line in enumerate(lines));counts['files']+=1;counts['rows']+=n
    if expected is not None:
     counts['matched_reaction_ids']+=1;counts['reaction_frames']+=expected
     key='complete' if n==expected and continuous else 'incomplete';counts[key]+=1;counts[key+'_reaction_frames']+=expected
     if key=='incomplete' and len(examples.get(key,[]))<5:examples.setdefault(key,[]).append(dict(id=ident,fer_rows=n,reaction_frames=expected,consecutive=continuous))
    else:
     counts['without_reaction_id']+=1
     if len(examples.get('without_reaction_id',[]))<5:examples.setdefault('without_reaction_id',[]).append(ident)
    out.execute('INSERT OR REPLACE INTO fer VALUES(?,?,?,?,?)',(ds,ident,n,expected,int(continuous)))
    if counts['files']%10000==0:out.commit();save('fer_progress',dict(dataset=ds,counts=counts,seconds=time.time()-t));print(ds,counts['files'],time.time()-t,flush=True)
  out.commit();reports[ds]=dict(counts=counts,examples=examples,seconds=time.time()-t,complete_hours=counts['complete_reaction_frames']/90000);print(ds,reports[ds],flush=True);save('fer',reports)
 out.close()
elif args.stage=='reactions':
 with db() as c:known={(ds,ident):n for ds,ident,n in c.execute('SELECT dataset,id,frames FROM clips')}
 out=sqlite3.connect(OUT/'reaction_stats.sqlite');out.execute('CREATE TABLE IF NOT EXISTS reaction(dataset TEXT,id TEXT,kind TEXT,frames INTEGER,max_score REAL,positive_frames INTEGER,over_half_frames INTEGER,over_half_runs INTEGER,PRIMARY KEY(dataset,id,kind))');start=time.time();counter=0;aggregate={}
 with db() as c:
  for ds,ident,kind,body in c.execute('SELECT dataset,id,kind,body FROM annotations'):
   b=zlib.decompress(body);data=b.split(b'\n',1)[1].replace(b',',b' ');arr=np.fromstring(data.decode(),sep=' ').reshape(-1,2);scores=arr[:,1];positive=scores>.5;runs=int(np.count_nonzero(positive & ~np.r_[False,positive[:-1]]));n=len(scores)
   out.execute('INSERT OR REPLACE INTO reaction VALUES(?,?,?,?,?,?,?,?)',(ds,ident,kind,n,float(np.max(scores)),int(np.count_nonzero(scores>0)),int(np.count_nonzero(positive)),runs));counter+=1
   if counter%25000==0:out.commit();save('reaction_progress',dict(rows=counter,seconds=time.time()-start));print(counter,round(time.time()-start,1),flush=True)
 out.commit()
 rows=out.execute('SELECT dataset,id,max(max_score),sum(over_half_runs) FROM reaction GROUP BY dataset,id').fetchall()
 report={'files':counter,'seconds':time.time()-start,'per_dataset':{}}
 for ds in ['realtalk','seamless']:
  cr=[r for r in rows if r[0]==ds];report['per_dataset'][ds]={}
  for name,test in [('any_nonzero',lambda r:r[2]>0),('any_over_half',lambda r:r[2]>.5),('all_zero',lambda r:r[2]==0)]:
   keep=[(r[0],r[1],known[(r[0],r[1])]) for r in cr if test(r)];report['per_dataset'][ds][name]=summary(keep)
 report['raw_over_half_runs_by_class']=[dict(dataset=r[0],reaction=r[1],runs=r[2]) for r in out.execute('SELECT dataset,kind,sum(over_half_runs) FROM reaction GROUP BY dataset,kind')];report['note']='Raw threshold runs precede gap merging, minimum-duration filtering, dominance resolution and manual event verification; they are not paper event annotations.'
 save('reactions',report);print(json.dumps(report,indent=2),flush=True);out.close()
