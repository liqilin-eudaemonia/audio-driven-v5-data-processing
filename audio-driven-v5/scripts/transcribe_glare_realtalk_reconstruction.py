from pathlib import Path
import json,subprocess,collections,torch,whisper
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';loc=json.loads((W/'realtalk_localization.json').read_text());a,b=loc['best']['start'],loc['best']['end'];audio=W/'realtalk_0003_inferred.wav'
subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(a),'-i','/public/0_DATA/2_Avatar/realtalk/audio/-1wCGAxqT_4.mp3','-t',str(b-a),'-ar','16000','-ac','1',str(audio)],check=True)
print('loading Whisper small.en on',torch.cuda.get_device_name(),flush=True);model=whisper.load_model('small.en',device='cuda',download_root=str(W/'whisper_models'));res=model.transcribe(str(audio),word_timestamps=True,language='en',fp16=True,condition_on_previous_text=False)
ann=json.loads(Path('/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4_annotations.json').read_text());words=[]
for seg in res['segments']:
 for word in seg.get('words',[]):
  x,y=a+word['start'],a+word['end'];counts=collections.Counter(ann.get(str(i),{}).get('current_speaker') for i in range(int(x*25),max(int(x*25)+1,int(y*25))));valid={k:v for k,v in counts.items() if k in ['p0','p1']};person=max(valid,key=valid.get) if valid else None
  words.append(dict(word=word['word'].strip(),start=x,end=y,person=person,probability=word['probability'],active_speaker_votes=dict(counts)))
result={'source_interval':[a,b],'model':'openai-whisper small.en','kind':'new ASR computed for the inferred source interval; not GLARE-released ASR','speaker_assignment':'Original RealTalk frame-wise current_speaker majority inside each word interval; uncertain where absent or mixed','text':res['text'],'words':words,'role_votes':dict(collections.Counter(ann.get(str(i),{}).get('current_speaker') for i in range(round(a*25),round(b*25))))}
(W/'realtalk_reconstruction_asr.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
