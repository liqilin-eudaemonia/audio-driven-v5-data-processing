from pathlib import Path
import sys,json,shutil,time
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base/'DyStream'))
import torch,soundfile as sf
from PIL import Image
import app
assert torch.cuda.device_count()==1
print('GPU',torch.cuda.get_device_name(),flush=True)
x,sr=sf.read('wav_files/11.wav');clip=base/'deployment/smoke_audio.wav';sf.write(clip,x[:sr*2],sr)
t=time.time()
result=app.run_inference(Image.open('img_files/11.png'),str(clip),None,5,0.5,0.5,0,1.0)
print('RESULT',result,flush=True)
for item in (result if isinstance(result,tuple) else [result]):
 if isinstance(item,str) and Path(item).suffix=='.mp4' and Path(item).exists():shutil.copy2(item,base/'deployment/dystream_smoke.mp4')
(base/'deployment/logs/dystream_smoke_result.json').write_text(json.dumps({'result':str(result),'seconds':time.time()-t,'gpu':torch.cuda.get_device_name(),'peak_memory_bytes':torch.cuda.max_memory_allocated()},indent=2))
