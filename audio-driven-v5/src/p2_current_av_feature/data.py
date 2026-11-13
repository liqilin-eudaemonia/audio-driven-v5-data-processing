"""Current Qwen/Gemma/AV plus existing mixed teacher; never open Event/Whisper."""
from pathlib import Path
from p2_listen.contracts import read,sha
from p2_current_av_only.data import Dataset as AVDataset,checked_index,data_config,to_device

def open_teacher(index,verify=True):
 import p2_j_current_teacher_cache as teacher
 from p2_j_current_qwen_hidden import Reader as HReader
 from p2_whisper_event_feature.data import CompatibleTeacherReader
 target=teacher.Reader.__new__(teacher.Reader)
 target.work=Path(index['caches']['listener_teacher']['work'])
 target.new=teacher.IncrementReader(target.work,verify=verify);p=target.new.protocol
 if sha(teacher.OLD/'protocol.json')!=p['old_protocol_sha256'] or sha(teacher.OLD/'COMPLETE.json')!=p['old_complete_sha256']:
  raise ValueError('Accepted old teacher changed')
 target.old=CompatibleTeacherReader(teacher.OLD);target.h=HReader(teacher.SOURCE,verify=False)
 done=read(target.work/'MIXED_COMPLETE.json')
 if (not done['success'] or done['samples']!=index['samples']
     or done['protocol_sha256']!=sha(target.work/'protocol.json')
     or done['increment_complete_sha256']!=sha(target.work/'COMPLETE.json')
     or p['selection_sha256']!=index['selection_sha256']):raise ValueError('Incomplete current teacher')
 if verify:
  mapped={row[0]for row in target.new.conn.execute('SELECT sid FROM samples')}
  if mapped!=set(target.h.ids) or not mapped-target.new.ids<=target.old.ids:
   raise ValueError('Teacher mixed mapping misses current members')
 return target

class Dataset(AVDataset):
 def __init__(self,c,verify=True):
  super().__init__(c,verify)
  from p2_listen.config import extractor_config
  self.cfg=extractor_config(data_config(c));self.teacher=open_teacher(self.index,verify)
  if self.teacher.h.protocol['sample_ids']!=self.reader.protocol['sample_ids']:
   raise ValueError('Teacher membership/order differs')
 def sample(self,index,device='cpu'):
  import threading
  from p2_j_teacher.core import validate_geometry
  value=super().sample(index,device)
  if threading.get_ident()==self.owner:r=self.teacher
  else:
   if not hasattr(self.local,'teacher'):self.local.teacher=open_teacher(self.index,False)
   r=self.local.teacher
  t=r.load(value['sample_id'],device);validate_geometry(t,value['layout'],self.cfg.target_dim)
  value['teacher']={k:t[k]for k in ('video','audio','video_valid','audio_valid')}
  return value

def data_identity(c):
 from p2_current_av_only.data import data_identity as av_identity
 index=checked_index(c);cache=index['caches']['listener_teacher'];work=Path(cache['work'])
 if sha(cache['complete'])!=cache['complete_sha256']or sha(work/'protocol.json')!=cache['protocol_sha256']:
  raise ValueError('Teacher receipt/protocol changed')
 value=av_identity(c);value['active_caches']['listener_teacher']=cache
 value['objective']='video_av+audio_av+.025*video_feature+.025*audio_feature'
 return value
