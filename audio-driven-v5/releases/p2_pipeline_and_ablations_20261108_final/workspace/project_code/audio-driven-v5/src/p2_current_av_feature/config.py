"""Original AV+Feature arm, final selection, no Event or Whisper."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha
from p2_current_av_only.config import BASE,REFERENCE
from . import VERSION
FEATURE_BASE=ROOT/'configs/p2_j_guided_av_feature_base025_20261102.json'

def validate(c):
 from p2_current_av_only.config import validate as av_validate
 from p2_current_av_only import VERSION as AV_VERSION
 if (c.get('current_av_feature_experiment')!=VERSION or c['id']!='guided_av_feature_s1'
     or c['loss']!='av_feature' or c['loss_weights']!=dict(av=1.,event=0.,video_feature=.025,audio_feature=.025)
     or c['teacher_cache']!=read(c['current_cache_index'])['caches']['listener_teacher']['work']
     or c['feature_reference']!=str(FEATURE_BASE) or sha(FEATURE_BASE)!=c['feature_reference_sha256']):
  raise ValueError('Require original AV+Feature weights and current teacher')
 original=read(FEATURE_BASE)
 for k in ('connector','text_lora','qwen_lora','event_supervision','data_config'):
  if c[k]!=original[k]:raise ValueError('Original Feature architecture changed: '+k)
 view=copy.deepcopy(c)
 view.update(id='guided_av_s1',loss='av',teacher_cache=None,current_av_only_experiment=AV_VERSION,
             loss_weights=dict(av=1.,event=0.,video_feature=0.,audio_feature=0.))
 av_validate(view)
 return c

def load(path):return validate(read(path))

def source_identity():
 from p2_current_av_only.config import source_identity as original
 result=original()
 for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_current500h_av_feature.py']:
  result[str(p)]=sha(p)
 return result
