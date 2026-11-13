"""Pair catalog + source metadata and bounded preview jobs for existing review server."""
import csv,io,json,sqlite3,zlib,re,tarfile,threading,concurrent.futures,subprocess,hashlib,time,sys,shlex
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'evaluation/glare_pair_review';PAGE=ROOT/'data_previews/glare_reaction_annotation';DB=WORK/'catalog.sqlite'
CROPWORK=WORK/'crop_reconstruction'
NORM=ROOT/'data_corpora/seamless_listen_local_full/normalized_pairs';POOL=concurrent.futures.ThreadPoolExecutor(max_workers=2);JOBS={};LOCK=threading.Lock()

def read(p):return json.loads(Path(p).read_text())
def connection():return sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
def clip(dataset,ident):
 with connection() as db:
  db.row_factory=sqlite3.Row;r=db.execute('SELECT * FROM clips WHERE dataset=? AND id=?',(dataset,ident)).fetchone()
 if r is None:raise ValueError('Unknown published clip')
 result=dict(r)
 if dataset=='realtalk' and ident=='-1wCGAxqT_4/0003' and (CROPWORK/'realtalk_localization.json').exists():
  evidence=read(CROPWORK/'realtalk_localization.json');result.update(start=evidence['best']['start'],end=evidence['best']['end'],participant=evidence['best']['person'],localization=evidence,time_source='inferred from published temporal features, not an author-supplied timestamp')
 return result
@lru_cache(maxsize=128)
def member(archive,name):
 with tarfile.open(archive,'r:') as tf:
  m=tf.getmember(name);return dict(archive=archive,member=name,offset=m.offset_data,byte_size=m.size)
def raw(ref):
 with open(ref['archive'],'rb') as f:f.seek(ref['offset']);return f.read(ref['byte_size'])
def uri(ref):return f"subfile,,start,{ref['offset']},end,{ref['offset']+ref['byte_size']},,:{ref['archive']}"
@lru_cache(maxsize=256)
def probe_video(archive,member_name,offset,byte_size):
 ref=dict(archive=archive,member=member_name,offset=offset,byte_size=byte_size)
 p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,duration,nb_frames','-of','json',uri(ref)],capture_output=True,text=True,timeout=30)
 if p.returncode:return {'status':'unavailable','error':p.stderr[-300:]}
 data=json.loads(p.stdout)['streams'][0]
 from fractions import Fraction
 try:data['fps']=float(Fraction(data.get('avg_frame_rate','0/1')))
 except (ValueError,ZeroDivisionError):data['fps']=None
 return data
@lru_cache(maxsize=128)
def tracks(source):
 for kind in ['improvised','naturalistic']:
  p=NORM/f'{kind}_{source}.json'
  if not p.exists():continue
  result=[]
  for item in read(p)['tracks']:
   src=item['contract']['source'];name=Path(src['member']).stem;ref=member(src['archive'],name+'.json');meta=json.loads(raw(ref))
   result.append(dict(participant=name.rsplit('_',1)[1],id=name,source=src,metadata_source=ref,asr=meta.get('metadata:transcript',[]),vad=meta.get('metadata:vad'),preview_audio=item['path'],preview_gain_db=item.get('gain_db')))
  return result
 return []
def pair_tracks(r):
 if r['dataset']=='seamless':return tracks(r['source_id'])
 if r['dataset']=='realtalk' and r['id']=='-1wCGAxqT_4/0003' and r.get('start') is not None:
  path=Path('/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4.avi');audio='/public/0_DATA/2_Avatar/realtalk/audio/-1wCGAxqT_4.mp3'
  asr_path=CROPWORK/'realtalk_reconstruction_asr.json';asr=read(asr_path) if asr_path.exists() else {'words':[]}
  result=[]
  for person in ['p0','p1']:
   words=[w for w in asr['words'] if w.get('person')==person];text=' '.join(w['word'] for w in words)
   result.append(dict(participant=person,id='realtalk_-1wCGAxqT_4_'+person,source=dict(archive=str(path),member=path.name,offset=0,byte_size=path.stat().st_size,container_type='original AVI file, not tar'),metadata_source=dict(active_speaker='/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4_annotations.json',asr=str(asr_path)),asr=[dict(start=r['start'],end=r['end'],transcript=text,words=words)],asr_note='本次重跑 Whisper small.en；按原始 RealTalk active-speaker 标注推定词语人物，不是 GLARE 发布的 ASR。',audio_note='原始混合音轨，两窗共用；没有进行人物声源分离。',preview_audio=audio))
  return result
 return []

def transcript(track,a,b):
 eps=0.0001;segments=[];words=[]
 for s in track['asr']:
  if float(s['end'])<=a+eps or float(s['start'])>=b-eps:continue
  clipped=[w for w in s.get('words',[]) if float(w['end'])>a+eps and float(w['start'])<b-eps]
  segments.append(dict(start=s['start'],end=s['end'],context=s.get('transcript',''),clip_text=' '.join(w['word'] for w in clipped),words=clipped))
  words.extend(clipped)
 return dict(text=' '.join(w['word'] for w in words),segments=segments,empty_note='此区间无 ASR 词记录，不等于确认绝对静音' if not words else None)
def detail(dataset,ident):
 r=clip(dataset,ident);r['training_crop']={'status':'not_released','box':None,'note':'未找到原片坐标中的固定训练裁脸框或 crop/resize/pad 变换。FER/MediaPipe 框是裁后检测框，不能代替。'}
 r['glare_pair_manifest']={'status':'not_released','note':'以下两人来自原始数据的同一 interaction；GLARE 未发布显式 speaker↔listener 配对清单。'}
 with connection() as db:
  r['reactions']={}
  for kind,blob in db.execute('SELECT kind,body FROM annotations WHERE dataset=? AND id=?',(dataset,ident)):
   rows=list(csv.DictReader(io.StringIO(zlib.decompress(blob).decode('utf-8-sig'))));r['reactions'][kind]=[dict(frame=int(float(x['frame_index'])),score=float(next(v for k,v in x.items() if k!='frame_index'))) for x in rows]
 r['annotation_fps']=25;r['tracks']=[]
 if pair_tracks(r):
  for t in pair_tracks(r):
   data={k:v for k,v in t.items() if k not in ['asr','vad']};data['source_video']=probe_video(**dict(archive=t['source']['archive'],member_name=t['source']['member'],offset=t['source']['offset'],byte_size=t['source']['byte_size']));geometry=CROPWORK/(t['id']+f"_{r['start']:.3f}_segment.json");data['crop_reconstruction']=read(geometry) if geometry.exists() else None;data['provisional_role']='标注对象 / 待核验 listener' if t['participant']==r['participant'] else '同对话另一人 / 推定 speaker';data['transcript']=transcript(t,r['start'],r['end']);r['tracks'].append(data)
  r['source_status']=('temporal_fingerprint_inferred' if dataset=='realtalk' else 'matched_original_interaction') if len(r['tracks'])==2 else 'original_metadata_not_located'
 else:
  r['source_status']='missing_clip_time_and_participant_mapping';r['start']=r['end']=None
  p=Path('/public/0_DATA/2_Avatar/realtalk/asr')/(r['source_id']+'.json')
  r['mixed_asr_source']=str(p) if p.exists() else None
  if p.exists():r['mixed_asr']=[dict(start=s['start'],end=s['end'],text=s['text']) for s in read(p).get('segments',[])]
  if r['source_id']=='-1wCGAxqT_4':r['source_video_url']='media/realtalk_-1wCGAxqT_4.mp4'
  r['missing_note']='000x 仅是片段编号，不是源视频秒数；没有入点或人物映射，无法将整段混合 ASR 切分成此片段的 speaker/listener ASR。'
 return r

def build_original_media(dataset,ident,key):
 r=clip(dataset,ident);out=PAGE/'media/pairs'/key;out.mkdir(parents=True,exist_ok=True);result=[]
 for t in pair_tracks(r):
  name=t['participant']+'_source8.mp4';dest=out/name
  if not dest.exists():
   part=dest.with_suffix(f'.{threading.get_ident()}.part.mp4');args=['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-ss',str(r['start']),'-i',uri(t['source']),'-ss',str(r['start']),'-i',t['preview_audio'],'-t',str(r['end']-r['start']),'-map','0:v:0','-map','1:a:0','-vf','scale=480:480:force_original_aspect_ratio=decrease:force_divisible_by=2','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','23','-threads','2','-c:a','aac','-movflags','+faststart',str(part)]
   p=subprocess.run(args,capture_output=True,text=True,timeout=600)
   if p.returncode:raise RuntimeError(p.stderr[-1800:])
   part.replace(dest)
  result.append(dict(participant=t['participant'],url=f'media/pairs/{key}/{name}',audio_note=t.get('audio_note','试听使用本站已有的统一响度音轨，保持源时间轴；不是 GLARE 发布的训练音轨。')))
 return result

def crop_record(track,r):
 p=CROPWORK/(track['id']+f"_{r['start']:.3f}_segment.json")
 return read(p) if p.exists() else None

def ensure_crops(ident,r):
 ts=pair_tracks(r)
 if all(crop_record(t,r) for t in ts):return
 cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=7','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',str(ROOT/'compare/.envs/reactface/bin/python'),str(ROOT/'scripts/glare_crop_detect.py'),'--id',ident])
 p=subprocess.run([sys.executable,str(ROOT/'compare/deployment/remote.py'),cmd],capture_output=True,text=True,timeout=600)
 if p.returncode:raise RuntimeError('Face-Alignment crop reconstruction failed: '+p.stderr[-1200:]+p.stdout[-400:])

def build_media(dataset,ident,key,variant='auto'):
 r=clip(dataset,ident);ensure_crops(ident,r);originals={v['participant']:v for v in build_original_media(dataset,ident,key)};out=PAGE/'media/pairs'/key;result=[]
 validation=read(CROPWORK/'geometry_validation.json') if (CROPWORK/'geometry_validation.json').exists() else None
 for t in pair_tracks(r):
  g=crop_record(t,r);chosen=g.get('selected','square_1p5')
  if variant!='auto':
   if variant not in g['candidates']:raise ValueError('Requested crop candidate unavailable')
   chosen=variant
  x1,y1,x2,y2=g['candidates'][chosen];w,h=g['source_size'];pad=[max(0,-x1),max(0,-y1),max(0,x2-w),max(0,y2-h)]
  digest=hashlib.sha256(json.dumps([chosen,[x1,y1,x2,y2],r['frames'],'yuv420p-v2']).encode()).hexdigest()[:10];name=t['participant']+'_crop_'+digest+'.mp4';dest=out/name
  seconds=r['frames']/25;raw_seconds=r['end']-r['start'];pad_sec=max(0,seconds-raw_seconds)+.12
  if not dest.exists():
   vf=f"trim=duration={raw_seconds},setpts=PTS-STARTPTS,pad=iw+{pad[0]+pad[2]}:ih+{pad[1]+pad[3]}:{pad[0]}:{pad[1]}:color=black,crop={x2-x1}:{y2-y1}:{x1+pad[0]}:{y1+pad[1]}:exact=1,scale=512:512:flags=bilinear,fps=25,tpad=stop_mode=clone:stop_duration={pad_sec},trim=end_frame={r['frames']}"
   af=f"atrim=duration={raw_seconds},asetpts=PTS-STARTPTS,apad=pad_dur={pad_sec},atrim=duration={seconds}"
   part=dest.with_suffix(f'.{threading.get_ident()}.part.mp4');args=['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-ss',str(r['start']),'-i',uri(t['source']),'-ss',str(r['start']),'-i',t['preview_audio'],'-map','0:v:0','-map','1:a:0','-vf',vf,'-af',af,'-t',str(seconds),'-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','18','-threads','2','-c:a','aac','-movflags','+faststart',str(part)]
   proc=subprocess.run(args,capture_output=True,text=True,timeout=600)
   if proc.returncode:raise RuntimeError(proc.stderr[-1600:])
   part.replace(dest)
  selected=dict(g,selected=chosen,box_xyxy=[x1,y1,x2,y2],padding_ltrb=pad,validation_sample=validation)
  result.append(dict(participant=t['participant'],url=f'media/pairs/{key}/{name}',original_url=originals[t['participant']]['url'],crop=selected,alignment={'output_fps':25,'output_frames':r['frames'],'source_interval_seconds':raw_seconds,'output_seconds':seconds,'tail_adjustment_seconds':seconds-raw_seconds,'note':'No time stretching. At most frame-quantization boundary is padded by repeating the last source frame; annotation frame indices are preserved.'},audio_note=originals[t['participant']]['audio_note']))
 return result

def media(dataset,ident,variant='auto'):
 r=clip(dataset,ident)
 if not pair_tracks(r) or r.get('start') is None:return dict(status='unavailable',reason='缺少片段源时间或原始双人媒体索引')
 key=hashlib.sha256((dataset+'/'+ident).encode()).hexdigest()[:20]
 with LOCK:
  job_key=key+':'+variant
  if job_key not in JOBS:JOBS[job_key]=POOL.submit(build_media,dataset,ident,key,variant)
  f=JOBS[job_key]
 if not f.done():return dict(status='preparing')
 try:return dict(status='ready',tracks=f.result())
 except Exception as e:return dict(status='error',reason=str(e))

def respond(handler,value,status=200):
 body=json.dumps(value,ensure_ascii=False).encode();handler.send_response(status);handler.send_header('Content-Type','application/json; charset=utf-8');handler.send_header('Cache-Control','no-store');handler.send_header('Content-Length',str(len(body)));handler.end_headers();handler.wfile.write(body)
def handle(handler):
 u=urlsplit(handler.path)
 if not u.path.startswith('/glare_reaction_annotation/api/'):return False
 q=parse_qs(u.query);get=lambda key,default='':q.get(key,[default])[0]
 try:
  if not DB.exists():respond(handler,dict(status='indexing',progress=read(WORK/'progress.json') if (WORK/'progress.json').exists() else {}),503);return True
  route=u.path.rsplit('/',1)[-1]
  if route=='pairs':
   query=get('q')[:150];dataset=get('dataset');offset=max(0,int(get('offset','0')));params=[];where=[]
   if dataset:where.append('dataset=?');params.append(dataset)
   if query:where.append('id LIKE ?');params.append('%'+query+'%')
   condition=' WHERE '+' AND '.join(where) if where else ''
   with connection() as db:
    db.row_factory=sqlite3.Row;total=db.execute('SELECT count(*) FROM clips'+condition,params).fetchone()[0];rows=db.execute('SELECT * FROM clips'+condition+' ORDER BY dataset,id LIMIT 40 OFFSET ?',params+[offset]).fetchall()
   respond(handler,dict(total=total,offset=offset,items=[dict(x) for x in rows]))
  elif route=='detail':respond(handler,detail(get('dataset'),get('id')))
  elif route=='media':respond(handler,media(get('dataset'),get('id'),get('variant','auto')))
  else:respond(handler,dict(error='Unknown endpoint'),404)
 except (ValueError,KeyError,sqlite3.Error,FileNotFoundError) as e:respond(handler,dict(error=str(e)),400)
 except Exception as e:respond(handler,dict(error=str(e)),500)
 return True
