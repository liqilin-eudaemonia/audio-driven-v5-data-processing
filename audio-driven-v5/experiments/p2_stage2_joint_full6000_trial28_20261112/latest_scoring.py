"""Resolve evaluation to the latest captured workspace revision, for all stages.

Training/checkpoint identities keep their original exact training-code mapping.
The active evaluator has its own explicit source manifest and current policy.
"""
from pathlib import Path
import hashlib,importlib,importlib.util,json,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];OMNI=ROOT.parent/'OmniResponse'
def state():return json.loads((W/'LATEST_EVALUATION_ACTIVE.json').read_text())
def identity():
 s=state();snap=Path(s['snapshot']);manifest=json.loads((snap/'MANIFEST.json').read_text())
 return {str(W/'latest_scoring.py'):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  str(W/'LATEST_EVALUATION_ACTIVE.json'):hashlib.sha256((W/'LATEST_EVALUATION_ACTIVE.json').read_bytes()).hexdigest(),
  str(snap/'MANIFEST.json'):s['manifest_sha256'],**manifest['files']}

def activate():
 if getattr(activate,'done',False):return
 s=state();snap=Path(s['snapshot']);files=identity()
 for path,value in files.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value:raise ValueError('Latest evaluator snapshot changed: '+path)
 import p2_event_presence
 for name in list(sys.modules):
  if name.startswith('p2_event_presence.'):del sys.modules[name]
 p2_event_presence.__path__=[str(snap/'p2_event_presence')]
 p2_event_presence.__file__=str(snap/'p2_event_presence/__init__.py')
 media=importlib.import_module('p2_event_presence.media');media.ROOT=ROOT;media.OMNI=OMNI
 native_source=media.source_identity
 prefix='src/p2_event_presence/'
 canonical_reader=str(OMNI/'evaluation/seamless/reader.py')
 def source():
  result={k:v for k,v in native_source().items()if not k.startswith(prefix)and k!=canonical_reader}
  result.update({prefix+p.name:files[str(p)]for p in(snap/'p2_event_presence').glob('*.py')})
  result[canonical_reader]=files[str(snap/'seamless_reader.py')]
  return result
 media.source_identity=source
 sys.path.insert(0,str(OMNI));importlib.import_module('evaluation.seamless')
 name='evaluation.seamless.reader';spec=importlib.util.spec_from_file_location(name,snap/'seamless_reader.py')
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);module.DATA_ROOT=ROOT
 # Plan definitions must come from the same current reaction taxonomy.
 sys.modules.pop('p2_stage2_joint_v2.plan_evidence',None)
 activate.done=True
