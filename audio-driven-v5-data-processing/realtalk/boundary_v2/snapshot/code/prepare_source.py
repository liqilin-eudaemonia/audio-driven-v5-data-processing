from pathlib import Path
import json,subprocess,hashlib,concurrent.futures
R=Path(__file__).resolve().parents[1];BASE=R.parent/'full_20261023';m=json.loads((BASE/'manifests/-1wCGAxqT_4.json').read_text());src=m['source']['source'];(R/'source_metadata.json').write_text(json.dumps(m['source'],indent=2))
def video():
 subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','4','-i',src,'-vf','fps=25:start_time=0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-movflags','+faststart',str(R/'source_video.mp4')],check=True)
def audio():
 subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',src,'-vn','-af','aresample=16000:async=1:first_pts=0','-ac','1','-ar','16000','-c:a','pcm_f32le',str(R/'source_audio.wav')],check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(lambda f:f(),[video,audio]))
print('SOURCE_READY',flush=True)
