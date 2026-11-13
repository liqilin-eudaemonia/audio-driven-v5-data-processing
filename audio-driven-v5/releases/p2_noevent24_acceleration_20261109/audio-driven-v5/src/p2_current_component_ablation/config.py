"""Freeze the final selection and existing native components; isolate each arm."""
from pathlib import Path
import copy,math
from p2_listen.contracts import ROOT,read,sha
from . import VERSION,ARMS
REFERENCE=ROOT/'experiments/p2_current500h_av_whisper31_20261107'
FULL=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106/config.json'

def flags(c):return ARMS[c['component_arm']]

def native_whisper_view(c):
 from p2_whisper_av import VERSION as WVERSION
 v=copy.deepcopy(c)
 v.update(id='guided_av_s1',loss='av',teacher_cache=None,event_head_training=False,
          whisper_experiment=WVERSION,loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025))
 return v

def validate(c,formal=True):
 if c.get('current_component_ablation')!=VERSION or c.get('component_arm')not in ARMS:raise ValueError('Unknown component ablation')
 from p2_matrix16.config import validate as matrix_validate
 view=copy.deepcopy(c);view['loss_weights']=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
 matrix_validate(view,formal=formal)
 f=flags(c);base=read(REFERENCE/'config.json');full=read(FULL);index=read(base['current_cache_index'])
 if c['component_reference']!=str(REFERENCE/'config.json')or c['component_reference_sha256']!=sha(REFERENCE/'config.json')or c['full_reference']!=str(FULL)or c['full_reference_sha256']!=sha(FULL):raise ValueError('Frozen references changed')
 if c['loss_weights']!=dict(av=1.,event=.05 if f['event']else 0.,video_feature=.025 if f['feature']else 0.,audio_feature=.025 if f['feature']else 0.):raise ValueError('Wrong active loss weights')
 expected_loss='av_event_feature'if f['event']and f['feature']else('av_event'if f['event']else'av_feature')
 if c['loss']!=expected_loss or c['id']!='guided_'+expected_loss+'_s1' or c['event_head_training']is not f['event']:raise ValueError('Wrong arm/loss identity')
 for k in ('data_config','j_cache','plan_cache','current_cache_index','sample_list','qwen_hidden_source','initialization','qwen_lora','text_lora','event_supervision','duration_sampling','execution'):
  if c[k]!=base[k]:raise ValueError('Changed current baseline '+k)
 if formal and c['connector']!=base['connector']:raise ValueError('Changed Connector architecture')
 if c['training']!=base['training'] or c['stage']!=1 or c['text_condition']!='guided' or c['parent_experiment']is not None:raise ValueError('Fresh independent matched training required')
 if c['teacher_cache']!=(index['caches']['listener_teacher']['work']if f['feature']else None):raise ValueError('Wrong teacher branch')
 if f['whisper']:
  if c.get('whisper')!=base['whisper']or c.get('whisper_cache')!=base['whisper_cache']:raise ValueError('Wrong speaker Whisper branch')
 else:
  if any(k in c for k in ('whisper','whisper_cache')):raise ValueError('Disabled Whisper must be absent')
 for k in ('clip_labels','clip_numeric_labels','current_source'):
  if c.get(k)!=(full[k]if f['event']else None):raise ValueError('Wrong Event source '+k)
 if c['hidden_event_supervision']!=dict(readout='last_block_before_output',clip_weight=1. if f['event']else 0.,temporal_weight=0.):raise ValueError('Only current LTX clip Event is supported')
 if f['event']and(c['event_supervision']!=full['event_supervision']or c['hidden_event_supervision']!=full['hidden_event_supervision']):raise ValueError('Current full-model Event policy changed')
 if Path(c['output']).resolve()in (REFERENCE.resolve(),Path(full['output']).resolve()):raise ValueError('Independent output required')
 return c

def load(path):return validate(read(path))

def source_identity():
 from p2_current_av_whisper.config import source_identity as original
 result=original()
 for p in [*Path(__file__).parent.glob('*.py'),ROOT/'scripts/p2_current500h_component_ablations.py']:
  result[str(p)]=sha(p)
 return result
