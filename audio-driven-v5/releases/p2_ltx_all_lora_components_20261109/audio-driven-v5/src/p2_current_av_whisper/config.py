"""Only replace the old Qwen+Whisper AV arm's data with final500h."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha
from p2_current_av_only.config import BASE,REFERENCE
from . import VERSION
WHISPER_BASE=ROOT/'configs/p2_j_qwen_whisper_av_20261104.json'

def native_view(c):
 from p2_whisper_av import VERSION as NATIVE_VERSION
 v=copy.deepcopy(c)
 v.update(whisper_experiment=NATIVE_VERSION,loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025))
 return v

def validate(c):
 from p2_current_av_only.config import validate as av_validate
 from p2_current_av_only import VERSION as AV_VERSION
 from p2_whisper_av.config import validate as native_validate
 original=read(WHISPER_BASE)
 if (c.get('current_av_whisper_experiment')!=VERSION or c['whisper']!=original['whisper']
     or c['whisper_reference']!=str(WHISPER_BASE) or sha(WHISPER_BASE)!=c['whisper_reference_sha256']
     or c['whisper_cache']!=read(c['current_cache_index'])['caches']['speaker_whisper']['work']
     or c['baseline_output']!=original['baseline_output']):raise ValueError('Changed original Whisper branch or current cache')
 native_validate(native_view(c))
 v=copy.deepcopy(c)
 for k in ('whisper','whisper_cache'):v.pop(k)
 v['current_av_only_experiment']=AV_VERSION
 av_validate(v)
 return c

def load(path):return validate(read(path))

def source_identity():
 from p2_current_av_only.config import source_identity as original
 result=original()
 for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_current500h_av_whisper.py']:
  result[str(p)]=sha(p)
 return result
