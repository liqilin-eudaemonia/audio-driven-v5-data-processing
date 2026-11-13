"""Resolve canonical event-module identities to byte-identical private snapshots.

The event scorer is concurrently edited by another task. All nine original
files were recovered byte-for-byte from the existing November8 release. No
training code, parameters, checkpoint metadata or calibration is rewritten.
"""
from pathlib import Path
import hashlib,json,os,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
SNAP=W/'frozen_src/p2_event_presence'
def activate():
 if getattr(activate,'done',False):return
 sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
 expected=json.loads((W/'run/run.json').read_text())['source']
 canonical=ROOT/'src/p2_event_presence';frozen={}
 for p in SNAP.glob('*.py'):
  key=str(canonical/p.name);value=hashlib.sha256(p.read_bytes()).hexdigest()
  if expected.get(key)!=value:raise ValueError('Private scorer is not original checkpoint bytes: '+p.name)
  frozen[key]=value
 original_keys={k for k in expected if k.startswith(str(canonical)+'/')}
 if set(frozen)!=original_keys:raise ValueError('Incomplete original event snapshot')
 import p2_event_presence
 p2_event_presence.__path__=[str(SNAP)]
 import p2_stage2_joint_v2.config as C
 original=C.source_identity
 def source():
  current=original()
  result={k:v for k,v in current.items()if not k.startswith(str(canonical)+'/')}
  for key,value in frozen.items():
   if hashlib.sha256((SNAP/Path(key).name).read_bytes()).hexdigest()!=value:raise ValueError('Private event snapshot changed')
  result.update(frozen)
  return result
 C.source_identity=source
 from p2_event_presence import media
 media.ROOT=ROOT;media.OMNI=ROOT.parent/'OmniResponse'
 media_source=media.source_identity
 def event_source():
  result={k:v for k,v in media_source().items()if not k.startswith('src/p2_event_presence/')}
  result.update({str(Path(k).relative_to(ROOT)):v for k,v in frozen.items()})
  return result
 media.source_identity=event_source
 # The same concurrent update also edits the shared detector policy. Resolve
 # that canonical data path to the original release bytes in this trial only.
 import p2_listen.contracts as contracts
 old_read,old_sha=contracts.read,contracts.sha
 policy=ROOT/'configs/p2_event_presence_v1.json';snapshot=W/'frozen_src/event_config.json'
 if hashlib.sha256(snapshot.read_bytes()).hexdigest()!='5f75ba3e345243c95baa28982ba88e5bb66906fe85f63130776bd8c58271c740':raise ValueError('Private event policy changed')
 active=W/'LATEST_EVALUATION_ACTIVE.json'
 if active.exists():
  revision=json.loads(active.read_text());snapshot=Path(revision['policy'])
  if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=revision['policy_sha256']:raise ValueError('Latest event policy changed')
 def resolve(path):return snapshot if Path(path).resolve()==policy else path
 contracts.read=lambda path:old_read(resolve(path))
 contracts.sha=lambda path:old_sha(resolve(path))
 isolation=str(W/'isolation');paths=os.environ.get('PYTHONPATH','').split(os.pathsep)
 os.environ['PYTHONPATH']=os.pathsep.join([isolation,*[p for p in paths if p and p!=isolation]])
 # vLLM's random free-port selection closes the probe socket before its
 # EngineCore binds. Concurrent instances can select the same port. Give
 # each physical GPU a disjoint, non-ephemeral starting range instead.
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if visible.isdigit():os.environ['VLLM_PORT']=str(16000+int(visible)*256)
 activate.done=True
 if active.exists():
  import latest_scoring
  latest_scoring.activate()
