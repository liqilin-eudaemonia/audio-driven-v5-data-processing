"""Current full-pool SFT references: paired speaker audio + listener first frame."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def ensure(ok,message):
 if not ok:raise ValueError(message)

def resolve(row):
 from p2_j_current_qwen_hidden import item_from_row
 p=row['provenance']
 with open(p['labels_file'],'rb') as f:f.seek(p['labels_byte_offset']);raw=f.readline()
 ensure(hashlib.sha256(raw).hexdigest()==p['labels_line_sha256'],'Frozen label row changed')
 r=json.loads(raw);ensure(r['sample_id']==row['sample_id'] and r['split']==row['split'],'Label identity changed')
 ensure(row['target']['text']==r['caption'],'Caption body changed')
 ensure(row['target']['source_file']==r['caption_file'] and row['target']['source_sha256']==r['caption_file_sha256'],'Caption file changed')
 ensure(row['schema']=='j-upright-response-sft-reference-v1','Unknown schema')
 ensure(p['processed_listener_waveform_nonzero'] is (not r['listener_audio_is_silent']),'Wrong waveform gate')
 item=item_from_row(r);a=row['inputs']['speaker_audio'];fr=row['inputs']['listener_first_frame'];s=item['source']
 ensure(a['participant_id']==s['speaker_id'] and fr['participant_id']==s['listener_id'],'Wrong participant roles')
 ensure(a['source_result_file']==s['source_result'] and a['source_result_sha256']==s['source_result_sha256'] and a['clip_index']==s['source_clip_index'],'Wrong paired source')
 ensure(a['start_s']==s['source_start_s'] and a['end_s']==s['source_end_s'] and fr['source_time_s']==s['source_start_s'],'Wrong media time')
 ensure(fr['source']==s['source'] and fr['video_transform']==s['video_transform'],'Wrong final video geometry')
 ensure(row['duration_s']==r['duration_s'],'Wrong duration')
 return item

def load_example(row):
 import numpy as np
 import soundfile as sf
 from listen_benchmark.j_omni import check_file
 from listen_response_sft.data import first_frame
 item=resolve(row);s=item['source'];a=item['speaker'];check_file(a['path'],a['sha256'])
 with sf.SoundFile(a['path']) as f:
  sr=f.samplerate;ensure(sr==a['sample_rate'] and f.channels==1,'Speaker format changed');lo=round(s['source_start_s']*sr);hi=round(s['source_end_s']*sr)
  ensure(0<=lo<hi<=len(f),'Speaker bounds invalid');f.seek(lo);wave=f.read(hi-lo,dtype='float32')
 ensure(len(wave)==hi-lo and np.isfinite(wave).all(),'Incomplete speaker audio')
 ensure(abs(len(wave)/sr-row['duration_s'])<=1/sr,'Duration mismatch')
 image,pts=first_frame(s)
 return dict(sample_id=row['sample_id'],split=row['split'],inputs=dict(speaker_audio=wave,speaker_sample_rate=sr,listener_first_frame=image),target_text=row['target']['text'],audit=dict(source_frame_time_s=pts,future_listener_media_in_inputs=False,speaker_audio_path=a['path'],speaker_audio_sha256=a['sha256']))

class Reader:
 def __init__(self,root):
  self.root=Path(root);self.db=sqlite3.connect(f'file:{self.root}/index.sqlite?mode=ro',uri=True);self.db.row_factory=sqlite3.Row
 def row(self,sid):
  ix=self.db.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone();ensure(ix is not None,'Unknown accepted id')
  with (self.root/ix['file']).open('rb') as f:f.seek(ix['offset']);raw=f.read(ix['bytes'])
  r=json.loads(raw);ensure(r['sample_id']==sid and r['split']==ix['split'],'Index differs');return r
 def load(self,sid):return load_example(self.row(sid))

def audit(final,canaries):
 # Resolve the input strata even if a canary was rejected; also decode one
 # accepted sample per split. No GPU and no future listener media in inputs.
 source=Reader(final.parent/'source');accepted=Reader(final);ids=list(dict.fromkeys(canaries))
 accepted_ids=[]
 for s in ('train','dev','test'):
  q=accepted.db.execute('SELECT sid FROM samples WHERE split=? ORDER BY sid LIMIT 1',(s,)).fetchone()
  if q:accepted_ids.append(q[0])
 records=[]
 for sid in dict.fromkeys(ids+accepted_ids):
  row=source.row(sid);value=load_example(row);im=value['inputs']['listener_first_frame']
  ensure(im.size==(512,512),'Wrong first-frame size')
  records.append(dict(sample_id=sid,split=row['split'],image_size=list(im.size),audio_samples=len(value['inputs']['speaker_audio']),sample_rate=value['inputs']['speaker_sample_rate'],audit=value['audit']))
 return dict(passed=True,cases=records,accepted_ids=accepted_ids,at=__import__('time').time())
