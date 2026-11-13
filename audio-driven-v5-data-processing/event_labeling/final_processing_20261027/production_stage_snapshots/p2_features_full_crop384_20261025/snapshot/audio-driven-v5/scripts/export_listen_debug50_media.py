"""CPU media export for a frozen debug sample; retain source crop and time identity."""
import argparse,bisect,collections,concurrent.futures,hashlib,json,math,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import av
import numpy as np
import soundfile as sf
from listen_benchmark.full_raw import open_source
from listen_labeling.media import crop_resize

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def export_one(row,full,out):
 sid=row['sample_id'];s=row['source'];d=row['duration_s'];dest=out/'media'/sid;dest.mkdir(parents=True,exist_ok=False)
 sources={}
 for role in ['listener','speaker']:
  src=full['normalized_audio'][role];rate=src['sample_rate'];a=round(s['source_start_s']*rate);z=round(s['source_end_s']*rate)
  with sf.SoundFile(src['path']) as f:
   assert f.samplerate==rate and f.channels==1 and 0<=a<z<=len(f)
   f.seek(a);data=f.read(z-a,dtype='float32')
  assert abs(len(data)/rate-d)<=1/rate+1e-8
  sf.write(dest/(role+'.wav'),data,rate,subtype='PCM_16')
  sources[role]=dict(path=src['path'],declared_source_sha256=src['sha256'],start_sample=a,end_sample=z,sample_rate=rate)
 images=[];times=[];pts=[];origin=s['source_start_s']
 with open_source(s['source']) as (c,reader):
  stream=c.streams.video[0];stream.codec_context.thread_count=1
  assert [stream.width,stream.height]==s['video_transform']['source_size']
  c.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
  for frame in c.decode(stream):
   t=float(frame.pts*frame.time_base)-origin
   if t<0:continue
   if t>=d:break
   images.append(crop_resize(frame.to_image(),s['video_transform']));times.append(t);pts.append(frame.pts)
 assert images and times[0]<.1 and d-times[-1]<.1
 n=math.ceil(d*25-1e-9);indices=[min(bisect.bisect_left(times,i/25),len(times)-1) for i in range(n)]
 images[indices[0]].save(dest/'first_frame.png')
 movie=av.open(str(dest/'video.mp4'),'w');stream=movie.add_stream('libx264',rate=25);stream.width=512;stream.height=512;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast','threads':'2'}
 for i,k in enumerate(indices):
  frame=av.VideoFrame.from_image(images[k]);frame.pts=i;frame.time_base=Fraction(1,25)
  for packet in stream.encode(frame):movie.mux(packet)
 for packet in stream.encode():movie.mux(packet)
 movie.close()
 hashes={name:digest(dest/name) for name in ['video.mp4','listener.wav','speaker.wav','first_frame.png']}
 meta=dict(sample_id=sid,source_start_s=origin,duration_s=d,render_duration_s=n/25,fps=25,frames=n,video_transform=s['video_transform'],source_video=s['source'],reader=reader,audio_sources=sources,source_frame_times_s=times,output_to_source_indices=indices,sha256=hashes,frame_selection='first source frame at/after each 25Hz sample; last frame reused only at end',target_geometry=[512,512],training_admitted=False)
 dump(dest/'media.json',meta)
 obs=dict(schema='p2-listen-observation-v1',sample_id=sid,experiment_split='train',official_split='train',group_id=full['group_id'],grouping_ids=['session:'+full['group_id'],'participant:'+full['listener_id'].rsplit('_',1)[-1],'participant:'+full['speaker_id'].rsplit('_',1)[-1]],speaker_audio=str(dest/'speaker.wav'),listener_image=str(dest/'first_frame.png'),requested_duration_s=d,geometry=dict(fps=25,height=512,width=512),media_hashes=dict(speaker_audio=hashes['speaker.wav'],listener_image=hashes['first_frame.png']))
 target=dict(schema='p2-listen-target-v1',sample_id=sid,target_video=str(dest/'video.mp4'),target_audio=str(dest/'listener.wav'),duration_s=d,media_hashes=dict(target_video=hashes['video.mp4'],target_audio=hashes['listener.wav']),admitted=False)
 print(json.dumps(dict(sample_id=sid,frames=n)),flush=True)
 return obs,target,meta

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.run.resolve();selection=json.loads((out/'selection.json').read_text());rows=selection['samples'];wanted=collections.defaultdict(dict)
 for r in rows:wanted[r['source']['source_manifest']][r['source']['source_line_1based']]=r
 full={}
 for filename,lines in wanted.items():
  with Path(filename).open('rb') as f:
   for i,line in enumerate(f,1):
    if i in lines:
     r=lines[i];v=json.loads(line);assert v['sample_id']==r['sample_id'];assert hashlib.sha256(line).hexdigest()==r['source']['source_line_sha256'];full[v['sample_id']]=v
    if i>=max(lines):break
 assert len(full)==50;dump(out/'source_rows.json',full)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  results=list(pool.map(lambda r:export_one(r,full[r['sample_id']],out),rows))
 dump(out/'observations.json',[r[0] for r in results]);dump(out/'targets.candidate.json',[r[1] for r in results]);dump(out/'media_manifest.json',[r[2] for r in results])
 print('MEDIA_EXPORT_COMPLETE',len(results))
if __name__=='__main__':main()
