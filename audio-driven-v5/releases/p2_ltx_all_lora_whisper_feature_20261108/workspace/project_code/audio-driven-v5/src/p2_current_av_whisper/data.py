"""Read final Qwen/Gemma/AV plus speaker Whisper, without Event/teacher readers."""
from pathlib import Path
import threading
from p2_listen.contracts import read,sha
from p2_current_av_only.data import Dataset as AVDataset,checked_index,data_config

def open_speaker(c,index):
 from p2_j_current_whisper_cache import Reader
 r=Reader(c['whisper_cache'],require_complete=True);p=r.new.protocol
 done=read(r.work/'MIXED_COMPLETE.json');enc=p['encoder']
 if (r.sample_ids()!=read(Path(c['j_cache'])/'protocol.json')['sample_ids']
     or p['selection_sha256']!=index['selection_sha256'] or p['source']!=c['j_cache']
     or Path(enc['path']).resolve()!=Path(c['whisper']['model_path']).resolve()
     or enc['input_dim']!=c['whisper']['input_dim'] or enc['dtype']!='float32'
     or not done['success'] or done['audio_source']!='speaker' or done['samples']!=index['samples']
     or done['protocol_sha256']!=sha(r.work/'protocol.json')
     or done['increment_complete_sha256']!=sha(r.work/'COMPLETE.json')
     or not r.ids-r.new.ids<=r.old.ids):raise ValueError('Whisper source/encoder/coverage mismatch')
 return r

class Dataset(AVDataset):
 def __init__(self,c,verify=True):
  super().__init__(c,verify);self.speaker=open_speaker(c,self.index)
 def sample(self,index,device='cpu'):
  value=super().sample(index,device)
  if threading.get_ident()==self.owner:r=self.speaker
  else:
   if not hasattr(self.local,'speaker'):self.local.speaker=open_speaker(self.spec,self.index)
   r=self.local.speaker
  value['whisper']=r.load(value['sample_id'],device);value['whisper'].validate()
  return value

def data_identity(c):
 from p2_current_av_only.data import data_identity as av_identity
 index=checked_index(c);cache=index['caches']['speaker_whisper'];work=Path(cache['work'])
 if sha(cache['complete'])!=cache['complete_sha256']or sha(work/'protocol.json')!=cache['protocol_sha256']:
  raise ValueError('Whisper complete/protocol changed')
 v=av_identity(c);v['active_caches']['speaker_whisper']=cache;v['objective']='video_av+audio_av with Qwen+speaker Whisper'
 return v

def to_device(value,device):
 from p2_listen.cache import tree_to
 result=tree_to(dict(value,condition=None,whisper=None),device)
 result.update(condition=value['condition'].to(device),whisper=value['whisper'].to(device))
 return result
