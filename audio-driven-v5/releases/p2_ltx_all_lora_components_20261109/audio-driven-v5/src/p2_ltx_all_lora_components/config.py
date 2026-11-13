"""Only expand the native LTX LoRA scope of the frozen current component arms."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha
from p2_current_component_ablation.config import validate as native_validate,flags
from p2_current_component_ablation.config import REFERENCE as CACHE_REFERENCE
from p2_ltx_all_lora.config import scope
from . import VERSION,ARMS
REFERENCE=ROOT/'experiments/p2_current_component_ablations31_20261108'
DEFAULT_OUTPUT=ROOT/'experiments/p2_ltx_all_lora_components_20261109'
META=('expanded_component_experiment','expanded_component_reference','expanded_component_reference_sha256','ltx_lora_scope')
def expected(output,arm):
 if arm not in ARMS:raise ValueError('Unknown expanded arm')
 ref=REFERENCE/arm/'config.json';c=copy.deepcopy(read(ref));s=scope()
 if arm=='no_whisper':s['whisper_full']=[]
 c.update(output=str(Path(output).resolve()),expanded_component_experiment=VERSION,
          expanded_component_reference=str(ref),expanded_component_reference_sha256=sha(ref),ltx_lora_scope=s)
 return c
def validate(c,formal=True):
 native_validate({k:v for k,v in c.items()if k not in META},formal=formal)
 if c!=expected(c['output'],c['component_arm']):raise ValueError('Final500h, latest caption, component recipe or all-LoRA scope changed')
 if Path(c['output']).resolve()in [(REFERENCE/a).resolve()for a in ARMS]:raise ValueError('Independent output required')
 return c
def load(path):return validate(read(path))
def source_identity():
 from p2_current_component_ablation.config import source_identity as original
 from p2_ltx_all_lora.config import source_identity as expanded
 result=original();result.update(expanded())
 for p in [*Path(__file__).parent.glob('*.py'),ROOT/'scripts/p2_ltx_all_lora_components_train.py']:
  result[str(p)]=sha(p)
 return result
