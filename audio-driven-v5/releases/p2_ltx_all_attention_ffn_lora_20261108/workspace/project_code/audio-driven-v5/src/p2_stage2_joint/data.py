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
 """Target cache reuse; frozen Qwen payloads are never opened by sample()."""
 def __init__(self,joint,verify=True):
  from p2_listen.config import extractor_config
  self.joint=joint;c=joint['generator'];super().__init__(c,verify)
  self.components=flags(c);self.ids=[r['sample_id']for r in self.rows]
  self.cfg=extractor_config(data_config(c))
  eligibility=Path(joint['planning_training_ids'])
  if sha(eligibility)!=joint['planning_training_ids_sha256']:raise ValueError('Planning eligibility changed')
  self.planning_ids=set(read(eligibility))
  hashes=Path(joint['planning_caption_hashes'])
  if sha(hashes)!=joint['planning_caption_hashes_sha256']:raise ValueError('Planning caption eligibility changed')
  self.planning_hashes=read(hashes)
  self.main_aux=self.open_aux(verify)
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

 def processor(self):
  from transformers import Qwen2_5OmniProcessor
  if not hasattr(self.local,'processor'):
   self.local.processor=Qwen2_5OmniProcessor.from_pretrained(self.cfg.omni_path,local_files_only=True)
  return self.local.processor

 def sample(self,index,device='cpu'):
  import numpy as np
  from PIL import Image
  from p2_j_current_qwen_hidden import item_from_row,input_media
  from p2_j_cache.data import observation
  from p2_listen.features import make_observation,image_grid
  from p2_listen.layout import make_layout
  from p3_listen.features import prepare_prefix
  from p2_planning_sft.data import PlanningCollator
  from omni_av.data import read_audio
  sid=self.ids[index];r=self._reader();row=r.source.row(sid);item=item_from_row(row)
  if sid in self.planning_ids and self.planning_hashes[sid]!=row['caption_file_sha256']:
   raise ValueError('Planning target no longer matches approvedSFT caption')
  media,audit=input_media(r.source,row,item);obs=observation(item,media)
  processor=self.processor()
  native=prepare_prefix(self.cfg,processor,make_observation(obs))
  obs['speaker_audio'].seek(0);wave=read_audio(obs['speaker_audio'],end_s=row['duration_s'])
  with Image.open(obs['listener_image'])as im:image=im.convert('RGB')
  collator=PlanningCollator(self.cfg.omni_path,max_length=32768,max_pixels=self.cfg.vision_max_pixels)
  collator.processor=processor
  planning=collator.source_inputs([dict(duration=row['duration_s'],image=image,audio=wave)])
  _,_,grid=image_grid(obs,processor,self.cfg)
  layout=make_layout(row['duration_s'],grid,self.cfg.vision_fps,self.cfg.audio_hz)
  text=dict(r.text.fixed(device),positive=r.text.load(sid,device)['positive'])
  if text['positive']['prompt']!=item['prompt']:raise ValueError('Latest full caption/Gemma differs')
  av=r.load(sid,device)
  value=dict(sample_id=sid,source_split=row['split'],condition=None,layout=layout,text=text,
       av=dict(av=av['av'],support=av['support']),teacher=None,qwen_generation_inputs=native,
       generation_prompt=item['prompt'],qwen_planning_inputs=planning,planning_target=row['caption'],
       planning_valid=sid in self.planning_ids,input_audit=audit)
  tokens=processor.tokenizer.encode(row['caption'],add_special_tokens=False)
  if not tokens or set(tokens)&set(processor.tokenizer.all_special_ids):raise ValueError('Invalid planning target')
  value['planning_tokens']=len(tokens)+1 if value['planning_valid']else 0
  aux=self.aux()
  if self.components['feature']:
   from p2_j_teacher.core import validate_geometry
   teacher=aux['teacher'].load(sid,device);validate_geometry(teacher,layout,self.cfg.target_dim)
   value['teacher']={k:teacher[k]for k in ('video','audio','video_valid','audio_valid')}
  if self.components['whisper']:value['whisper']=aux['whisper'].load(sid,device)
  if self.components['event']:
   from p2_ltx_events.layout import make_layout as hidden_layout
   source,labels,numeric=aux['event'];entry=source.row(sid)
   if entry['selected_row_sha256']!=digest(row):raise ValueError('Stale Event caption binding')
   value['events']=numeric.events(index,sid,digest(entry),device)
   value['hidden_event_layout']=hidden_layout(av['av'],av['support'],row['duration_s'])
  return value


def to_device(value,device):
 from p2_listen.cache import tree_to
 result=tree_to(dict(value,whisper=None),device)
 if value.get('whisper')is not None:result['whisper']=value['whisper'].to(device)
 else:result.pop('whisper',None)
 return result


def data_identity(c):
 from p2_current_component_ablation.data import data_identity as previous
 # The original assembler's identity requires an arm flag, but all actual
 # readers in this module use explicit component booleans and never H payloads.
 g=c['generator'];value=av_identity(g);index=checked_index(g)
 value['active_caches'].pop('qwen_hidden')
 value['qwen_source_metadata']=dict(protocol_sha256=index['caches']['qwen_hidden']['protocol_sha256'],hidden_payloads_read=False)
 for flag,name in [('feature','listener_teacher'),('whisper','speaker_whisper')]:
  if flags(g)[flag]:
   cache=index['caches'][name]
   if sha(cache['complete'])!=cache['complete_sha256']or sha(Path(cache['work'])/'protocol.json')!=cache['protocol_sha256']:raise ValueError('Auxiliary cache changed')
   value['active_caches'][name]=cache
 if flags(g)['event']:
  value['event_sources']={k:sha(Path(g[k])/'manifest.json')for k in ('clip_labels','clip_numeric_labels','current_source')}
 for key in ('planning_training_ids','planning_caption_hashes','validation_manifest'):
  if sha(c[key])!=c[key+'_sha256']:raise ValueError('Stage2 data binding changed: '+key)
  value[key+'_sha256']=c[key+'_sha256']
 value.update(objective='online_joint_alignment',components=flags(g))
 return value

