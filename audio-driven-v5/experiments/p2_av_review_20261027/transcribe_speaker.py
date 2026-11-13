import json,sys,hashlib,os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/listen_benchmark_deps/py310'))
from qwen_asr import Qwen3ASRModel
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
MODEL='/private/ckpts/Qwen3-ASR-1.7B'
model=Qwen3ASRModel.from_pretrained(MODEL,dtype=torch.bfloat16,device_map='cuda:0',max_inference_batch_size=1,max_new_tokens=512)
(OUT/'speaker_asr').mkdir(exist_ok=True)
for kind in ['train','test']:
 src=ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027/observations.json'
 for obs in json.loads(src.read_text()):
  path=Path(obs['speaker_audio']);h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==obs['media_hashes']['speaker_audio'];dest=OUT/'speaker_asr'/(h+'.json')
  if dest.exists():continue
  wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(axis=1);g=math.gcd(sr,16000)
  if sr!=16000:wave=resample_poly(wave,16000//g,sr//g).astype(np.float32)
  result=model.transcribe(audio=(wave,16000),language=None)[0]
  record=dict(sample_id=obs['sample_id'],speaker_audio=str(path),sha256=h,model=MODEL,text=result.text,language=result.language,source='ASR of exact speaker input WAV',used_for_training=False,human_verified=False)
  tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');tmp.replace(dest);print(kind,obs['sample_id'],result.text,flush=True)
print('speaker ASR complete',flush=True)
