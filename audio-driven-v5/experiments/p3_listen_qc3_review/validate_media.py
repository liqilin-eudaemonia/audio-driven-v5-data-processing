from pathlib import Path
import json,hashlib,sys
import av,soundfile as sf
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'data_previews/seamless_listen_qc3_review'
d=json.loads((out/'samples.json').read_text());rows=d['samples'];unchanged=0
for r in rows:
 folder=out/r['preview']
 for role in ['speaker','listener']:
  with sf.SoundFile(folder/(role+'.wav')) as f:
   assert f.samplerate==48000 and f.channels==1 and len(f)==round(r['duration_s']*48000)
  if r.get('reuse_root'):
   original=Path(r['reuse_root'])/r['reuse_preview']/(role+'.wav')
   assert hashlib.sha256(original.read_bytes()).hexdigest()==hashlib.sha256((folder/(role+'.wav')).read_bytes()).hexdigest()
 if r.get('reuse_root'):unchanged+=1
 with av.open(str(folder/'both.mp4')) as c:
  c.streams.video[0].thread_count=1;frames=list(c.decode(video=0));assert len(frames)==r['frames'];assert all((f.width,f.height)==(512,512) for f in frames)
 with av.open(str(folder/'both.mp4')) as c:
  assert c.streams.audio[0].codec_context.channels==2;assert len(list(c.decode(audio=0)))>0
q=json.loads((out/'annotation_queue.json').read_text())['rows'];assert len(q)==240 and all(r['annotation'] is None for r in q)
for r in q:
 with sf.SoundFile(out/r['audio']) as f:
  assert f.samplerate==48000 and f.channels==1 and len(f)==round(r['context_end']*48000)-round(r['context_start']*48000)
contract=json.loads((ROOT/'experiments/p3_listen_qc3_review/frozen_contract.json').read_text())
assert all(hashlib.sha256((ROOT/'scripts'/p).read_bytes()).hexdigest()==h for p,h in contract['code_sha256'].items())
im=Image.new('RGB',(1000,((len(rows)+3)//4)*290),'white');draw=ImageDraw.Draw(im)
for i,r in enumerate(rows):
 x=i%4*250;y=i//4*290;a=Image.open(out/r['preview']/'reference.jpg');a.thumbnail((246,246));im.paste(a,(x,y));draw.text((x+3,y+248),str(i+1)+' '+r['effective_decision']+' '+r['display_group'],fill='black');draw.text((x+3,y+265),r['pair'].split('_',1)[-1],fill='black')
im.save(out/'contact_sheet.jpg')
report=dict(samples=len(rows),reused_audio_bit_identical=unchanged,decoded_media=True,exact_wav_samples=True,annotation_audio=240,annotation_labels_all_empty=True,frozen_code_unchanged=True)
(out/'media_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
main=json.loads((out/'screened_candidates.json').read_text())['rows'];print('Main keep voiced:',sum(r['effective_decision']=='keep' and r['listener_human_evidence'] for r in main))
print('Displayed groups:',[(r['sample_id'],r['effective_decision'],r['listener_human_evidence']) for r in rows])
