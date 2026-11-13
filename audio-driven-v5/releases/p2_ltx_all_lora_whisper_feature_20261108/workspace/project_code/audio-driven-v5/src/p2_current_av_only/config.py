"""Explicit zero auxiliary weights; frozen original architecture and latest caches."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha
from . import VERSION
BASE=ROOT/'configs/p2_j_av_only_20261101.json'
REFERENCE=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'


def validate(c):
 from p2_matrix16.config import validate as native
 view=copy.deepcopy(c);view['loss_weights']=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
 native(view,formal=True)
 if (c.get('current_av_only_experiment')!=VERSION or c['id']!='guided_av_s1' or c['loss']!='av'
     or c['stage']!=1 or c['event_head_training'] is not False or c['teacher_cache'] is not None
     or c['clip_labels'] is not None or c['qwen_lora']['enabled']
     or c.get('qwen_hidden_source')!='base_pre_sft' or c.get('initialization')!='pretrained_only'
     or c['loss_weights']!=dict(av=1.,event=0.,video_feature=0.,audio_feature=0.)
     or any(k in c for k in ('whisper','whisper_cache','clip_numeric_labels','resume_execution','hidden_event_supervision'))):
  raise ValueError('Require original Qwen-only conditioning and strictly pure AV loss')
 if c['training']['global_batch'] not in (24,31,32) or c['training']['gradient_checkpointing']!='all' or c['training']['save_every']!=500:
  raise ValueError('Require explicit24/31/32 batch, all recomputation and500 save cadence')
 if (c['j_cache']!=c['plan_cache'] or sha(c['baseline_config'])!=c['baseline_config_sha256']
     or Path(c['output']).resolve()==Path(read(c['baseline_config'])['output']).resolve()):
  raise ValueError('Changed original AV reference or condition cache')
 original=read(c['baseline_config'])
 for k in ('qwen_lora','text_lora','connector','event_supervision','data_config'):
  if c[k]!=original[k]:raise ValueError('Original AV architecture changed: '+k)
 for k,v in original['training'].items():
  if k not in ('global_batch','steps','save_every','gradient_checkpointing') and c['training'][k]!=v:
   raise ValueError('Original AV optimizer/seed changed: '+k)
 if c['execution']!=dict(prefetch_batches=2,gradient_reduction='streaming_fp64',gradient_bucket_mb=512,
                         optimizer_state_cpu_offload=False,longest_sample_check=True):
  raise ValueError('Unreviewed AV execution change')
 return c


def load(path):return validate(read(path))


def source_identity():
 from p2_ltx_events.config import source_identity as current
 result=current()
 for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_current500h_av_only.py',ROOT/'scripts/p2_j_av_only.py']:
  result[str(p)]=sha(p)
 return result
