from pathlib import Path
import tarfile,zstandard as z,json,time
base=Path('/public/0_DATA/2_Avatar/GLARE_reaction_annotation_results');out=Path(__file__).resolve().parent
jobs=[('nod_headshake/face_detection/output_seamless.tar.zst',{'V00_S1453_I00001162_P1124A_126.007_129.457.pkl'}),('nod_headshake/face_detection/output_realtalk.tar.zst',{'-1wCGAxqT_4/0003.pkl'})]
for rel,targets in jobs:
 found=[];n=0
 with (base/rel).open('rb') as f,z.ZstdDecompressor(max_window_size=2**31).stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as tf:
  for m in tf:
   if not m.isfile():continue
   n+=1;name=m.name.removeprefix('./')
   if name in targets:
    dest=out/('reference_'+name.replace('/','_'));dest.write_bytes(tf.extractfile(m).read());found.append(name);print('FOUND',name,n,f.tell(),flush=True)
   if n%500==0:
    (out/'reference_progress.json').write_text(json.dumps({'archive':rel,'members':n,'compressed_bytes_read':f.tell(),'found':found,'updated':time.time()}));print('PROGRESS',rel,n,f.tell(),flush=True)
   if len(found)==len(targets):break
 (out/(Path(rel).stem+'.reference_result.json')).write_text(json.dumps({'archive':rel,'found':found,'members_scanned':n}))
