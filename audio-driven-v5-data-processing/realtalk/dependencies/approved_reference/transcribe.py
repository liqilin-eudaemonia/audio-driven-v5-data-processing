"""Fresh word-timestamp ASR on bounded pilot windows; original ASR is not trusted."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq,pipeline
ROOT=Path(__file__).resolve().parent
MODEL='/private/codes/exp/boxian_workspace/models/benchmark/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1'
torch.set_num_threads(2)
model=AutoModelForSpeechSeq2Seq.from_pretrained(MODEL,torch_dtype=torch.float16,local_files_only=True,attn_implementation='eager').to('cuda:0').eval()
model.config.forced_decoder_ids=None;model.generation_config.forced_decoder_ids=None
processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
recognizer=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device='cuda:0',torch_dtype=torch.float16)
print('model ready',flush=True)
while not (ROOT/'windows.json').exists():time.sleep(2)
rows=json.loads((ROOT/'windows.json').read_text())
for r in rows:
    out=ROOT/(r['window_id']+'_asr.json')
    if out.exists():continue
    wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','1','-ss',str(r['start']),'-i',str(ROOT.parent/(r['video_id']+'.avi')),'-t',str(r['end']-r['start']),'-vn','-ac','1','-ar','16000','-f','f32le','-']),np.float32).copy()
    with torch.inference_mode():
        result=recognizer({'raw':wave,'sampling_rate':16000},return_timestamps='word',generate_kwargs=dict(language='english',task='transcribe',max_new_tokens=400,num_beams=1,do_sample=False))
    out.write_text(json.dumps(dict(window=r,model=MODEL,**result),ensure_ascii=False,indent=2)+'\n')
    print(r['window_id'],r['video_id'],round(r['start'],2),result['text'],flush=True)
