from pathlib import Path
import tarfile,zstandard as z,pickle,json,math
base=Path('/public/0_DATA/2_Avatar/GLARE_reaction_annotation_results');out=Path(__file__).resolve().parent
reports=[]
for rel in ['nod_headshake/face_detection/output_realtalk.tar.zst','nod_headshake/face_detection/output_seamless.tar.zst','fer_emotion/output_realtalk.tar.zst','fer_emotion/output_seamless.tar.zst','reaction_detector/nodding/output_realtalk.tar.zst','reaction_detector/nodding/output_seamless.tar.zst']:
 rec={'archive':rel,'samples':[]};print('Reading',rel,flush=True)
 with (base/rel).open('rb') as f,z.ZstdDecompressor(max_window_size=2**31).stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as tf:
  for member in tf:
   if not member.isfile():continue
   sample={'path':member.name,'bytes':member.size}
   if member.size>80*1024*1024:sample['skipped']='individual member larger than sample limit'
   elif member.name.endswith('.pkl'):
    obj=pickle.load(tf.extractfile(member));sample['top_type']=type(obj).__name__;sample['length']=len(obj) if hasattr(obj,'__len__') else None
    if isinstance(obj,dict):sample['top_keys']=list(obj)
    frames=obj if isinstance(obj,list) else obj.get('frames',[]) if isinstance(obj,dict) else []
    if frames:
     sample['frame_keys']=sorted(set(k for frame in frames if isinstance(frame,dict) for k in frame))
     for i,frame in enumerate(frames):
      if isinstance(frame,dict) and frame.get('detected'):
       sample['first_detected_frame']=i;sample['first_detected_values']={k:frame.get(k) for k in frame if k!='landmarks'};sample['landmarks_count']=len(frame.get('landmarks') or [])
       norm=frame.get('bbox_norm');px=frame.get('bbox_px')
       if norm and px:sample['implied_image_size']=[px[2]/norm[2] if norm[2] else None,px[3]/norm[3] if norm[3] else None]
       break
   elif member.name.endswith(('.csv','.json','.txt')):sample['head']=tf.extractfile(member).read(2500).decode('utf-8',errors='replace')
   rec['samples'].append(sample)
   if len(rec['samples'])>=3:break
 reports.append(rec);(out/'archive_samples.json').write_text(json.dumps(reports,indent=2,default=str));print(json.dumps(rec,default=str)[:7000],flush=True)
print('SAMPLE AUDIT COMPLETE',flush=True)
