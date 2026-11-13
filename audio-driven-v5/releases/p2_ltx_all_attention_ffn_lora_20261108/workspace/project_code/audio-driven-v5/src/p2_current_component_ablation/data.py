"""Use current mixed caches; open/read only branches enabled in this arm."""
from pathlib import Path
import threading
from p2_listen.contracts import read,sha,digest
from p2_current_av_only.data import Dataset as AVDataset,checked_index,data_config
from p2_current_av_only.data import data_identity as av_identity
from p2_current_av_feature.data import open_teacher
from p2_current_av_whisper.data import open_speaker
from .config import flags

def event_readers(c,ids,verify=True):
 from p2_clip_events.labels import LabelReader
 from p2_clip_events.model import classes
 from p2_ltx_events.current_data import SourceReader
 from p2_ltx_events.numeric_labels import NumericLabels
 source=SourceReader(c['current_source'],verify);labels=LabelReader(c['clip_labels'],verify=verify)
 m=source.manifest;lm=labels.manifest
 if (m['training_ids']!=ids or labels.ids()!=ids or sha(c['current_cache_index'])!=m['current_cache_index_sha256']
     or lm['cache_manifest_sha256']!=sha(Path(c['current_source'])/'manifest.json')):raise ValueError('Current Event membership/source mismatch')
 numeric=NumericLabels(c['clip_numeric_labels'],labels,classes(c['event_supervision']),ids,verify=verify)
 return source,labels,numeric

class Dataset(AVDataset):
 def __init__(self,c,verify=True):
  super().__init__(c,verify);self.components=flags(c);self.ids=[r['sample_id']for r in self.rows]
  self.main_aux=self.open_aux(verify)
  if self.components['feature']:
   from p2_listen.config import extractor_config
   self.cfg=extractor_config(data_config(c))
  if self.components['event']:self.labels=self.main_aux['event'][1]

 def open_aux(self,verify=False):
  result={};f=self.components
  if f['feature']:
   result['teacher']=open_teacher(self.index,verify)
   if result['teacher'].h.protocol['sample_ids']!=self.ids:raise ValueError('Teacher member/order mismatch')
  if f['whisper']:result['whisper']=open_speaker(self.spec,self.index)
  if f['event']:result['event']=event_readers(self.spec,self.ids,verify)
  return result

 def aux(self):
  if threading.get_ident()==self.owner:return self.main_aux
  if not hasattr(self.local,'aux'):self.local.aux=self.open_aux(False)
  return self.local.aux

 def sample(self,index,device='cpu'):
  from p2_j_teacher.core import validate_geometry
  value=super().sample(index,device);sid=value['sample_id'];aux=self.aux()
  if self.components['feature']:
   t=aux['teacher'].load(sid,device);validate_geometry(t,value['layout'],self.cfg.target_dim)
   value['teacher']={k:t[k]for k in ('video','audio','video_valid','audio_valid')}
  if self.components['whisper']:
   value['whisper']=aux['whisper'].load(sid,device);value['whisper'].validate()
  if self.components['event']:
   from p2_ltx_events.layout import make_layout
   source,labels,numeric=aux['event'];item=source.row(sid);row=self._reader().source.row(sid)
   if item['selected_row_sha256']!=digest(row)or item['split']!=value['source_split']:raise ValueError('Stale current caption/Event binding')
   value['events']=numeric.events(index,sid,digest(item),device)
   value['hidden_event_layout']=make_layout(value['av']['av'],value['av']['support'],row['duration_s'])
  return value

def data_identity(c):
 value=av_identity(c);index=checked_index(c);f=flags(c)
 for flag,name in [('feature','listener_teacher'),('whisper','speaker_whisper')]:
  if f[flag]:
   cache=index['caches'][name]
   if sha(cache['complete'])!=cache['complete_sha256']or sha(Path(cache['work'])/'protocol.json')!=cache['protocol_sha256']:raise ValueError('Active auxiliary cache changed')
   value['active_caches'][name]=cache
 if f['event']:
  source=Path(c['current_source']);label=Path(c['clip_labels']);numeric=Path(c['clip_numeric_labels'])
  value['event_sources']={str(p):sha(p)for p in [source/'manifest.json',label/'manifest.json',numeric/'manifest.json']}
 value.update(objective=c['component_arm'],components=f)
 return value

def to_device(value,device):
 from p2_listen.cache import tree_to
 result=tree_to(dict(value,condition=None,whisper=None),device)
 result['condition']=value['condition'].to(device)
 if value.get('whisper')is not None:result['whisper']=value['whisper'].to(device)
 else:result.pop('whisper',None)
 return result
