"""Freeze the accepted no-Event recipe; change only LTX adaptation scope."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha
from p2_current_component_ablation.config import validate as native_validate
from p2_ltx_all_lora.config import scope
from . import VERSION
REFERENCE=ROOT/'experiments/p2_current_component_ablations31_20261108/no_event/config.json'
DEFAULT_OUTPUT=ROOT/'experiments/p2_ltx_all_lora_whisper_feature31_20261108'
META=('expanded_noevent_experiment','expanded_noevent_reference','expanded_noevent_reference_sha256','ltx_lora_scope')

def expected(output):
 c=copy.deepcopy(read(REFERENCE))
 c.update(output=str(Path(output).resolve()),expanded_noevent_experiment=VERSION,
          expanded_noevent_reference=str(REFERENCE),expanded_noevent_reference_sha256=sha(REFERENCE),ltx_lora_scope=scope())
 return c

def validate(c,formal=True):
 view={k:v for k,v in c.items()if k not in META};native_validate(view,formal=formal)
 if c!=expected(c['output']):raise ValueError('Require final500h, latest caption, noEvent recipe and full native attention/FFN LoRA')
 if Path(c['output']).resolve()==REFERENCE.parent.resolve():raise ValueError('Independent expanded-LoRA output required')
 return c

def load(path):return validate(read(path))

def flags(c):return dict(whisper=True,feature=True,event=False)

from p2_current_component_ablation.config import REFERENCE as CACHE_REFERENCE

def source_identity():
 from p2_current_component_ablation.config import source_identity as original
 from p2_ltx_all_lora.config import source_identity as expanded
 result=original();result.update(expanded())
 for path in [*Path(__file__).parent.glob('*.py'),ROOT/'scripts/p2_ltx_all_lora_noevent_train.py']:
  result[str(path)]=sha(path)
 return result
