#!/usr/bin/env python3
"""Restore reviewed native crops in new lossless media, preserving the original PTS and audio."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_upright_regen_20261105'
BASE=ROOT/'experiments/listen_j_reaction_vocal_omni_20261103'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def read(p):return json.loads(Path(p).read_bytes())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def sk(row):
 s=row['source']['source'];return hashlib.sha256((s['archive']+'\0'+s['member']).encode()).hexdigest()[:24]
def init_worker():
 global BOUNDED
 from listen_benchmark.quality_cpu_execution import bounded_detector_threads
 BOUNDED=bounded_detector_threads(str(ROOT));BOUNDED.__enter__()

def restore(task):
 row,review=task;sid=row['sample_id'];out=WORK/'restored_rows'/f'{sid}.json'
 if out.exists():
  previous=read(out);assert previous['parent_caption_sha256']==row['caption_file_sha256'];return previous
 import av,numpy as np
 from listen_visual_quality.measurement import SourceBytes,measure_row
 from listen_visual_quality.cli import load_policy,DEFAULT_MODEL
 from listen_visual_quality.decision import decide
 from listen_visual_quality.io import canonical_hash
 started=time.monotonic();source=row['source'];s=source['source'];c=source['video_transform']['crop'];start=source['source_start_s'];end=source['source_end_s']
 rotate=review['rotation_ccw_degrees'];assert rotate in [-90,90]
 directory=WORK/'derived_media'/sid;directory.mkdir(parents=True,exist_ok=True)
 video=directory/(source['listener_id']+'.nut');archive=directory/'upright.tar'
 inp=f"subfile,,start,{s['offset']},end,{s['offset']+s['byte_size']},,:{s['archive']}"
 vf=f"trim=start={start}:end={end},format=rgb24,crop={c['width']}:{c['height']}:{c['x']}:{c['y']},transpose={'cclock' if rotate==90 else 'clock'}"
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','1','-noautorotate','-ss',str(max(0,start-.2)),'-copyts','-i',inp,'-an','-vf',vf,'-c:v','libx264rgb','-crf','0','-preset','ultrafast','-threads','1','-bf','0','-vsync','0','-enc_time_base','-1','-avoid_negative_ts','disabled','-f','nut',str(video)]
 run=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 if run.returncode:raise RuntimeError('Media restoration failed '+sid+': '+run.stderr[-2000:])
 with av.open(str(video)) as v:
  st=v.streams.video[0];assert [st.width,st.height]==[c['height'],c['width']]
  decoded=list(v.decode(video=0));assert decoded
  pts=[float(f.pts*f.time_base) for f in decoded]
  assert start-1e-5<=pts[0]<start+.1 and end-.1<pts[-1]<end+1e-5,(sid,start,end,pts[0],pts[-1])
  idx=len(decoded)//2;targetpts=pts[idx];actual=decoded[idx].to_ndarray(format='rgb24')
 with SourceBytes(s) as member,av.open(member) as original:
  st=original.streams.video[0];st.codec_context.thread_count=1
  original.seek(int((targetpts-.1)/st.time_base),stream=st,backward=True,any_frame=False)
  closest=None;distance=1.
  for frame in original.decode(video=0):
   p=float(frame.pts*frame.time_base);delta=abs(p-targetpts)
   if delta<distance:closest=frame;distance=delta
   if p>targetpts+.02:break
  assert closest is not None and distance<1e-5,(sid,distance)
  rgb=closest.to_ndarray(format='rgb24');expected=np.rot90(rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']],1 if rotate==90 else -1)
  difference=np.abs(actual.astype(np.int16)-expected.astype(np.int16));mae=float(difference.mean());maximum=int(difference.max())
  assert mae<1.5 and maximum<=6,(sid,mae,maximum)
 provenance=dict(schema='j-upright-native-crop-v1',source_key=sk(row),rotation_ccw_degrees=rotate,
  original_source=source,original_caption_file=row['caption_file'],original_caption_sha256=row['caption_file_sha256'],
  original_review_id=review['review_id'],direction_review_sheet=review['rotation_review_sheet'],
  native_crop=c,spatial_resize=False,encoding='libx264rgb crf=0 lossless',
  restoration_code_sha256=sha(Path(__file__)),timestamps_preserved=True,first_frame_s=pts[0],last_frame_s=pts[-1],frame_count=len(pts),
  representative_pixel_check=dict(source_pts_s=targetpts,mean_abs_error=mae,max_abs_error=maximum),
  external_audio_recipe_unchanged=True,command=cmd)
 temp=archive.with_suffix('.tar.tmp')
 with tarfile.open(temp,'w',format=tarfile.USTAR_FORMAT) as tar:
  tar.add(video,arcname=video.name)
  data=(json.dumps(provenance,ensure_ascii=False)+'\n').encode();info=tarfile.TarInfo('rotation_provenance.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
 temp.replace(archive)
 with tarfile.open(archive) as tar:m=tar.getmember(video.name);media=dict(archive=str(archive),member=m.name,offset=m.offset_data,byte_size=m.size)
 provenance['derived_archive_sha256']=sha(archive);provenance['derived_media']=media
 derived=copy.deepcopy(source);derived['source']=media
 derived['video_transform']=dict(crop=dict(x=0,y=0,width=c['height'],height=c['width']),source_size=[c['height'],c['width']],output_size=[512,512],resample='bicubic')
 derived['crop_proposal']=dict(x=0,y=0,size=c['width'],status='proposal_only',visual_filter_applied=True,revision='quality-bounded-crop-v1-20261030')
 derived.pop('quality_screening',None);derived['orientation_restoration']=dict(provenance_file=str(directory/'provenance.json'),source_key=sk(row),rotation_ccw_degrees=rotate,archive_sha256=provenance['derived_archive_sha256'])
 write(directory/'provenance.json',provenance);video.unlink()
 request=dict(sample_id=sid,source=derived,split=row['split'],duration_s=row['duration_s'])
 policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json')
 record=measure_row(request,policy,str(ROOT),str(DEFAULT_MODEL));contract=read(WORK/'config.json')['visual_quality_gate']['contract_sha256']
 record.update(source_sha256=canonical_hash(derived),contract_sha256=contract,measurement_hash='j-upright-native-crop-v1-20261105',orientation_provenance=str(directory/'provenance.json'))
 mp=WORK/'quality_measurements'/f'{sid}.json';write(mp,record)
 if record['status']!='ok':raise RuntimeError('Restored measurement execution failed '+sid+': '+str(record.get('error')))
 decision=decide(record,policy,{});decision.update(measurement_file=str(mp),measurement_sha256=sha(mp),contract_sha256=contract)
 roll=[q['roll'] for q in record['focus_samples'] if q.get('roll') is not None]
 support=sum(bool(q['detected']) for q in record['raw_measurements'])/len(record['raw_measurements'])
 result=dict(sample_id=sid,parent_caption_file=row['caption_file'],parent_caption_sha256=row['caption_file_sha256'],
  original_manifest_row=row,request=dict(request,quality=decision),decision=decision,provenance_file=str(directory/'provenance.json'),
  roll_median=statistics.median(roll) if roll else None,face_support=support,elapsed_s=time.monotonic()-started)
 if decision['decision']=='review' and support>0:
  # The original pool is already retained. Preserve partial-support review if there is no clarity/face exclusion.
  result['quality_review_preserved']=decision['review_reasons'];request_decision=copy.deepcopy(decision);request_decision['decision']='retain';request_decision['review_reasons']=decision['review_reasons'];result['request']['quality']=request_decision
 result['restoration_usable']=decision['decision']!='exclude' and support>0
 write(out,result);return result

def main(workers,limit):
 sources={r['source_key']:r for r in read(WORK/'rotation_sources_confirmed.json')}
 rows=[json.loads(x) for x in (WORK/'rotation_candidates.jsonl').open()];rows=rows[:limit] if limit else rows
 done=[];started=time.monotonic()
 with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
  pending={pool.submit(restore,(r,sources[sk(r)])):r['sample_id'] for r in rows}
  for f in as_completed(pending):
   try:d=f.result()
   except Exception as e:
    write(WORK/'restoration_execution_errors'/f'{pending[f]}.json',dict(sample_id=pending[f],error=repr(e)));raise
   done.append(d)
   if len(done)%25==0 or len(done)==len(rows):
    status=dict(total=len(rows),completed=len(done),usable=sum(x['restoration_usable'] for x in done),elapsed_s=time.monotonic()-started);write(WORK/'restoration_status.json',status);print(json.dumps(status),flush=True)
 write(WORK/('restoration_summary.json' if not limit else 'restoration_smoke.json'),dict(total=len(done),usable=sum(x['restoration_usable'] for x in done),excluded=[dict(sample_id=x['sample_id'],decision=x['decision']) for x in done if not x['restoration_usable']],elapsed_s=time.monotonic()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);p.add_argument('--limit',type=int);a=p.parse_args();main(a.workers,a.limit)
