"""Persistent GPU worker: each source gets fresh v2 segmentation then listener-only separation/scoring."""
from pathlib import Path
import os,json,hashlib,subprocess,sys,time,fcntl,traceback,shutil,argparse
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
R=Path(__file__).resolve().parents[1]
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(task,w):
 source=task['source'];signature={'source':source,'archive_size':Path(source['archive']).stat().st_size,'archive_mtime':Path(source['archive']).stat().st_mtime_ns,'settings':'pilot fps25 crf20 f32mono16k first_pts0'}
 marker=w/'prepared.json'
 if marker.exists() and json.loads(marker.read_text())==signature and all((w/n).exists() for n in ['source_video.mp4','source_audio.wav']):return
 from concurrent.futures import ThreadPoolExecutor
 def video():
  p=w/'source_video.tmp.mp4';subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','2','-filter_threads','1','-i',source['source'],'-vf','fps=25:start_time=0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-movflags','+faststart',str(p)],check=True);p.replace(w/'source_video.mp4')
 def audio():
  p=w/'source_audio.tmp.wav';subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',source['source'],'-vn','-af','aresample=16000:async=1:first_pts=0','-ac','1','-ar','16000','-c:a','pcm_f32le',str(p)],check=True);p.replace(w/'source_audio.wav')
 with ThreadPoolExecutor(2) as pool:list(pool.map(lambda f:f(),[video,audio]))
 atomic(w/'source_metadata.json',source);atomic(marker,signature)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only');a=ap.parse_args()
 from asr import WordASR
 from engine import Engine
 from worker import process_video
 codehash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((R/'code').glob('*.py')))).hexdigest()
 tasks=json.loads((R/'source_queue.json').read_text());tasks.sort(key=lambda t:(t['video_id']!='-1wCGAxqT_4',-t['source_frames']))
 if a.only:tasks=[t for t in tasks if t['video_id']==a.only]
 asr=None;engine=None
 for task in tasks:
  vid=task['video_id'];lock=(R/'locks'/f'{vid}.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();continue
  try:
   statefile=R/'state'/f'{vid}.json';prev=json.loads(statefile.read_text()) if statefile.exists() else {}
   if prev.get('code_hash')==codehash and (prev.get('stage')=='complete' or prev.get('attempt',0)>=3):continue
   attempt=prev.get('attempt',0)+1 if prev.get('code_hash')==codehash else 1;started=time.time();w=R/'videos'/vid
   for d in [w,w/'logs',w/'asr']:d.mkdir(parents=True,exist_ok=True)
   def status(stage,**extra):atomic(statefile,dict(video_id=vid,stage=stage,attempt=attempt,code_hash=codehash,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started_at=started,updated_at=time.time(),**extra))
   try:
    status('preparing');prepare(task,w)
    status('transcribing');wordfile=w/'words.json'
    if not wordfile.exists() or json.loads(wordfile.read_text()).get('source_sha256')!=digest(w/'source_audio.wav'):
     if asr is None:asr=WordASR()
     asr.run(w)
    status('segmenting');env=dict(os.environ,REALTALK_VIDEO_WORK=str(w),REALTALK_VIDEO_ID=vid);subprocess.run([sys.executable,str(R/'code/segment.py')],env=env,check=True)
    manifest=R/'manifests'/f'{vid}.json';shutil.copy2(w/'manifest.json',manifest);m=json.loads(manifest.read_text());status('separating_and_scoring',pairs=len(m['pairs']),candidate_seconds=m['summary']['candidate_seconds'])
    if m['pairs']:
     if engine is None:engine=Engine(f'/tmp/realtalk_v2_{os.getpid()}')
     inferencehash=hashlib.sha256((codehash+json.dumps(engine.model_fingerprints,sort_keys=True)).encode()).hexdigest()
     process_video({'manifest':str(manifest),'manifest_sha256':digest(manifest),'video_id':vid,'pairs':len(m['pairs'])},engine,R,inferencehash)
     result=json.loads((R/'video_status'/f'{vid}.json').read_text())
     if result['status']!='complete':raise RuntimeError(f"retryable_pairs:{result.get('retryable_pairs')}")
    status('complete',pairs=len(m['pairs']),candidate_seconds=m['summary']['candidate_seconds'],elapsed_seconds=time.time()-started)
   except Exception as e:status('error',error=str(e),traceback=traceback.format_exc());print('SOURCE_ERROR',vid,traceback.format_exc(),flush=True)
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
