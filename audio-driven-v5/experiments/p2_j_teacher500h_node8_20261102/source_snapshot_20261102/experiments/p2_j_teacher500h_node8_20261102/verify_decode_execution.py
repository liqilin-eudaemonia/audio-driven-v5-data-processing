import sys,time,json,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,'/private/liqi_workspace/project_code/audio-driven-v5/src')
from p2_j_teacher_accel import selected_frames

def run(row):
 item,override,expected,threads=row;t=time.monotonic();frames,pts=selected_frames(item,override,(224,224),threads=threads)
 assert hashlib.sha256(frames.tobytes()).hexdigest()==expected['teacher_rgb_sha256'],item['sample_id']
 assert pts==expected['source_pts'],item['sample_id']
 return dict(sample_id=item['sample_id'],seconds=time.monotonic()-t,threads=threads,pixels_pts_exact=True)

if __name__=='__main__':
 import torch
 from p2_listen.contracts import read,sha
 root=Path('/private/liqi_workspace/project_code/audio-driven-v5');out=root/'experiments/p2_j_teacher500h_node8_20261102';source=root/'experiments/p2_j_train500h_cache_20261101'
 ids=read(out/'pilot_ids.json');m=read(source/'manifest.json');rows=[];torch.set_num_threads(1)
 for sid in ids:
  item=read(source/'samples'/(sid+'.json'));expected=torch.load(out/'pilot_v3/cache'/sid/'payload.pt',weights_only=True)['audit']
  rows.append((item,m.get('video_sampling_overrides',{}).get(sid),expected,1))
 start=time.time()
 with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  results=[]
  for value in pool.map(run,rows):results.append(value);print(json.dumps(value),flush=True)
 proof=dict(success=True,count=len(results),threads=1,seconds=time.time()-start,results=results,decoder_sha256=sha(root/'src/p2_j_teacher_accel.py'),reference_teacher_protocol_sha256=sha(out/'pilot_v3/protocol.json'),reference_check='all cropped RGB hashes and source PTS match the verified per-clip teacher inputs; model, audio and preprocessor are unchanged')
 (out/'decode_execution_verified.json').write_text(json.dumps(proof,indent=2)+'\n')
