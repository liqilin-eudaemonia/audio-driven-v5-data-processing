import sys,time,json,hashlib
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(root/'src'))
import av,numpy as np
from listen_response_sft.data import resolve_example,first_frame
from listen_visual_quality.measurement import SourceBytes
from listen_benchmark.frame_crop import crop_original_frame_reused
from p2_planning_sft.reference_data import ReferencePlanningDataset
c=json.load(open(root/'configs/p2_planning_sft_filtered_3b_32gpu_20261102.json'))
d=ReferencePlanningDataset(c['data']['seal'],'train',limit=8,seed=20261102)

def candidate(src,threads,short):
 start=src['source_start_s']
 with SourceBytes(src['source']) as member,av.open(member) as container:
  stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=threads
  assert [stream.width,stream.height]==src['video_transform']['source_size']
  rate=float(stream.average_rate)
  container.seek(int(max(0,start-(1/rate+1e-6 if short else 1))/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
  previous=None;prev_t=None;frames=0
  for frame in container.decode(video=0):
   assert frame.pts is not None
   frames+=1;t=float(frame.pts*frame.time_base)
   if t+1e-8>=start:
    chosen=previous if previous is not None and abs(prev_t-start)<=abs(t-start) else frame
    actual=float(chosen.pts*chosen.time_base)
    assert abs(actual-start)<=1/rate+1e-5
    return crop_original_frame_reused(chosen,src['video_transform']),actual,frames
   previous=frame;prev_t=t
 raise ValueError('No frame')

out=[]
for i in range(len(d)):
 row=d.record(i);source=resolve_example(row,d.crop_config)['source'];a=time.monotonic();img,pts=first_frame(source);secs=time.monotonic()-a
 result=dict(sample_id=row['sample_id'],baseline_seconds=secs,pts=pts,arms=[])
 for threads,short in [(4,False),(1,True),(4,True)]:
  a=time.monotonic();got,t,n=candidate(source,threads,short);elapsed=time.monotonic()-a
  same=np.array_equal(np.asarray(img),np.asarray(got)) and t==pts
  result['arms'].append(dict(threads=threads,short_seek=short,seconds=elapsed,frames=n,pixels_and_pts_equal=same))
  assert same,result
 out.append(result);print(json.dumps(result),flush=True)
p=root/'experiments/p2_planning_sft_filtered_3b_32gpu_20261102/performance/decode_benchmark.json';p.write_text(json.dumps(out,indent=2))
print('baseline',sum(x['baseline_seconds'] for x in out));print('candidates',[sum(x['arms'][j]['seconds'] for x in out) for j in range(3)])
