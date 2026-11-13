"""Fixed scope, model provenance and pure scheduling helpers."""
import importlib.util,json,math,time,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,PYTHON,METRIC_PYTHON
ARM='qwen7500_ltx6800';SEEDS=(137,238);COUNTS={'seamless':6624,'responsenet':586}
ORDER=[(d,s) for d in COUNTS for s in SEEDS]
SOURCE256={137:ROOT/'experiments/p2_joint_qwen_ltx_test256_16gpu_20261113',238:ROOT/'experiments/p2_joint_qwen_ltx_test256_seed238_16gpu_20261113'}
INPUTS={'seamless':ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112','responsenet':ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111/conditioning'}
TRUSTED=ROOT/'experiments/p2_ltx15_test64_sft2752_20261113'
RECORDED_CACHE=SOURCE256[137]/'cache_recorded_truncation.py'
GUARD=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
CAPS={'cache':80,'render':96,'decode':38,'metric':32}
def dataset_work(d):return BASE/d
def conditions(d):return dataset_work(d)/'conditioning'
def work(d,s):return dataset_work(d)/f'seed{s}'
def import_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def link(src,dst):
 src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():assert dst.resolve()==src.resolve(),str(dst)
 else:dst.symlink_to(src,target_is_directory=src.is_dir())
def slots():
 c=read(BASE/'cluster.json');assert {n['name']:n['gpus'] for n in c['nodes']}=={n:list(range(8)) for n in ['node12','node8','node11']} and len(c['nodes'])==3
 return [dict(rank=i,node=n,gpu=g) for i,(n,g) in enumerate((n,g) for n in c['nodes'] for g in n['gpus'])]
def stopped(d=None,s=None):return (BASE/'STOP').exists() or (d is not None and (dataset_work(d)/'STOP').exists()) or (s is not None and (work(d,s)/'STOP').exists())
def verify():
 rec=read(BASE/'PREPARED.json');assert rec['success']
 for f,h in rec['files'].items():assert sha(f)==h,('Frozen source/config changed',f)
 slots();return rec
def same_source(a,b):
 if a['key']!=b['key'] or a['geometry']!=b['geometry']:return False
 if a['dataset']=='responsenet':return a['reference']==b['reference']
 x,y=a['reference'],b['reference']
 return all(x['gt_reference'][k]==y['gt_reference'][k] for k in ['sample_id','row_sha256','crop_config_sha256','reader_contract']) and all(x[k]==y[k] for k in ['frames','fps','duration_s','listening_intervals','source_start','source_end'])
def valid_media(m,p,x,plan_sha):
 return m['step']==6800 and m['checkpoint_sha256']==p['arms'][0]['original_sha256'] and m['seed']==p['seed'] and m['generation']==p['generation'] and m['planning_adapter']==p['planner_adapter_identity'] and m['planning_sha256']==plan_sha and m['geometry']==x['geometry'] and m['export']['frames']==x['geometry']['frames'] and not m['target_description_access'] and not m['listener_future_input']
def choose_inference(active,remaining,backlog,free_gib):
 """22 persistent samplers +2 decoders, then all available GPUs help decode."""
 renders=sum(x['stage']=='render' for x in active);decoders=sum(x['stage']=='decode' for x in active)
 if backlog>decoders*4 and (decoders<2 or remaining==0) and free_gib>=CAPS['decode']+6:return 'decode'
 if remaining>renders and renders<22 and free_gib>=CAPS['render']+6:return 'render'
 return None
def cache_ready(d,i):return (conditions(d)/f'CACHE_COMPLETE_{i}.json').exists()

class CompletionIndex:
 """Verify new commits once; retain monotonic completions during a run."""
 def __init__(self,w,items):self.w=w;self.items=items;self.done=set();self.latent=set()
 def refresh(self):
  from p2_testsets_eval.runtime import ready
  for x in self.items:
   k=x['key']
   if k not in self.done and (self.w/'generated'/ARM/k/'complete.json').exists() and ready(self.w/'generated'/ARM/k):self.done.add(k);self.latent.add(k)
   if k not in self.latent and (self.w/'latents'/ARM/k/'complete.json').exists() and ready(self.w/'latents'/ARM/k):self.latent.add(k)
  return len(self.items)-len(self.latent),len(self.latent-self.done)
