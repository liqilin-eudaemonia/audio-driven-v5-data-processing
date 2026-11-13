from pathlib import Path
import sys,copy,math,types
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
OLD=W.parent/'p2_ltx_all_lora_components_20261109'
NOEVENT=W.parent/'p2_ltx_all_lora_noevent28_batch28_20261109'
TARGETS={'no_event':6573,'no_feature':6643,'no_whisper':6643}
def output(arm):return NOEVENT if arm=='no_event' else W/arm
def normalized(c):
 v=copy.deepcopy(c);v.pop('batch28_ablation');v['training']['global_batch']=31;v['training']['steps']=9169;return v
def validate(c):
 from p2_ltx_all_lora_components.config import validate as native
 native(normalized(c));assert c['batch28_ablation']=={'world':28,'reference_samples':186000}
 assert c['training']['global_batch']==28 and c['training']['steps']==10151
 return c
def source():
 from p2_ltx_all_lora_components.config import source_identity
 v=source_identity()
 for f in [W/'runtime.py',W/'runner.py']:v[str(f)]=sha(f)
 return v
class Schedule:
 def __init__(self,c):
  from p2_ltx_all_lora_components.schedule import Schedule as Native
  validate(c);old=Native(normalized(c));self.size=old.size;self.seed=old.seed;self.batch=28
  self.pairs=old.pairs[:10151*28];assert len(self.pairs)==10151*28
 def indices(self,size,step,batch,seed,rank=0,world=1):
  assert (size,batch,seed)==(self.size,28,self.seed) and world in (1,28) and 0<=rank<world
  assert 1<=step<=10151
  return self.pairs[(step-1)*28:step*28][rank::world]
def load():
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import p2_ltx_all_lora_components.training as native
 m=types.ModuleType('p2_ltx_all_lora_components._batch28');m.__package__='p2_ltx_all_lora_components'
 code=Path(native.__file__).read_text().replace('from .schedule import Schedule','# Schedule injected by batch28 runtime')
 exec(compile(code,native.__file__,'exec'),m.__dict__)
 m.validate=validate;m.source_identity=source;m.Schedule=Schedule
 from p2_ltx_all_lora_components.model import Model
 def model(c,cfg,renderer):
  v=Model(normalized(c),cfg,renderer);v.spec=c;return v
 m.QwenWhisperEventFeature=model
 return m
def noevent_runtime():
 path=W.parent/'p2_noevent28_after_full7500_20261109';sys.path.insert(0,str(path))
 import importlib.util
 s=importlib.util.spec_from_file_location('split_parent_runtime',W.parent/'p2_ltx_split24_8_batch31_20261109/runtime.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
 s=importlib.util.spec_from_file_location('resume_noevent28_runtime',path/'runtime28.py');m=importlib.util.module_from_spec(s)
 previous=sys.modules.get('runtime');sys.modules['runtime']=parent
 try:s.loader.exec_module(m)
 finally:sys.modules['runtime']=previous
 return m
