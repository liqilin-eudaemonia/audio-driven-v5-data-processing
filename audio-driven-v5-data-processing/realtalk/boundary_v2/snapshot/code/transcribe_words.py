from pathlib import Path
import os,json,time,hashlib
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
import numpy as np,soundfile as sf,torch
from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq,pipeline
R=Path(__file__).resolve().parents[1];MODEL='/private/codes/exp/boxian_workspace/models/benchmark/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1'
torch.set_num_threads(2)
while not (R/'source_audio.wav').exists():time.sleep(2)
wave,sr=sf.read(R/'source_audio.wav',dtype='float32');assert sr==16000;digest=hashlib.sha256((R/'source_audio.wav').read_bytes()).hexdigest();model=AutoModelForSpeechSeq2Seq.from_pretrained(MODEL,torch_dtype=torch.float16,local_files_only=True,attn_implementation='eager').to('cuda:0').eval();model.config.forced_decoder_ids=None;model.generation_config.forced_decoder_ids=None;proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);recognizer=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device='cuda:0',torch_dtype=torch.float16);words=[];chunks=[]
for i,start in enumerate(np.arange(0,len(wave)/sr,28.)):
 lo=max(0.,start-1);hi=min(len(wave)/sr,start+29);file=R/'asr'/f'{i:03d}.json'
 if file.exists() and json.loads(file.read_text()).get('source_sha256')==digest:r=json.loads(file.read_text())
 else:
  with torch.inference_mode():out=recognizer({'raw':wave[round(lo*sr):round(hi*sr)],'sampling_rate':sr},return_timestamps='word',generate_kwargs={'language':'english','task':'transcribe','max_new_tokens':400,'num_beams':1,'do_sample':False})
  r={'source_sha256':digest,'model':MODEL,'window':[lo,hi],'ownership':[start,min(start+28,len(wave)/sr)],**out};file.write_text(json.dumps(r,ensure_ascii=False,indent=2))
 chunks.append(r)
 for w in r['chunks']:
  a,b=w['timestamp']
  if a is None or b is None:continue
  a+=lo;b+=lo
  if start<=(a+b)/2<min(start+28,len(wave)/sr) and 0<=a<b<=len(wave)/sr+.04:words.append({'start':a,'end':b,'text':w['text'].strip(),'window':i})
 print('ASR_WINDOW',i,len(words),flush=True)
(R/'words.json').write_text(json.dumps({'source_sha256':digest,'model':MODEL,'words':sorted(words,key=lambda w:(w['start'],w['end'])),'word_timestamps_are_model_estimates':True},ensure_ascii=False,indent=2));print('ASR_DONE',len(words),flush=True)
