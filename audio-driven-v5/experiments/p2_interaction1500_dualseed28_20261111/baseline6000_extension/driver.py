"""Stage1500 evaluation; baseline selection is explicitly recorded before launch."""
from pathlib import Path
import sys,copy,time,fcntl,subprocess,inspect,importlib.util
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
TRAIN=ROOT/'experiments/p2_event_feature_interaction28_20261111'
LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(TRAIN),str(ROOT/'src'),str(ROOT/'scripts'),str(LEGACY)]
from extension import activate,source_identity,cluster_for,ORIGINAL
activate()
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON,partition
sys.path.insert(0,str(WORK))
NAMES=('baseline','event020_feature010');SEEDS=(137,238)
import adapter
adapter.install()
import stage_driver
_verify_source=inspect.getsource(stage_driver.verify_scoring)
_verify_source=_verify_source.replace("name=p['checkpoint_identity']['experiment']['loss_search']['candidate']", "name=p['checkpoint_identity']['experiment'].get('loss_search',{}).get('candidate', 'historical31')")
_verify_source=_verify_source.replace("checkpoint.parent.name!=name", "(checkpoint.parent.name!=name if name!='historical31' else str(checkpoint)!=read(BASELINE_SELECTION)['checkpoint'] or arm['name']!='losssearch_baseline_1500')")
_verify_namespace=dict(stage_driver.__dict__,BASELINE_SELECTION=WORK/'BASELINE_SELECTION.json')
exec(compile(_verify_source,str(__file__)+'::compatible_verify','exec'),_verify_namespace)
stage_driver.verify_scoring=_verify_namespace['verify_scoring']
sys.path.insert(0,str(WORK))

def target(name):return 1500 if name=='baseline' else 1661
def folder(name,seed):return WORK/'evaluations'/f'{name}_{target(name)}_seed{seed}'
def original_driver():
 spec=importlib.util.spec_from_file_location('original_event_driver',ROOT/'experiments/p2_event4_dualseed_eval28_20261110/driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def seal_score(work):
 import reference_events
 source=inspect.getsource(reference_events.freeze)
 needle="name = p['checkpoint_identity']['experiment']['loss_search']['candidate']"
 assert source.count(needle)==1
 source=source.replace(needle,"name = p['checkpoint_identity']['experiment'].get('loss_search',{}).get('candidate')\n    if name is None:\n        assert p['arms'][0]['name']=='losssearch_baseline_1500'\n        assert p['arms'][0]['original_checkpoint']==read(BASELINE_SELECTION)['checkpoint']\n        name='baseline'")
 ns=dict(reference_events.__dict__,BASELINE_SELECTION=WORK/'BASELINE_SELECTION.json')
 exec(compile(source,str(__file__)+'::compatible_reference_freeze','exec'),ns)
 native=original_driver();seal=inspect.getsource(native.seal_score)
 assert seal.count('from reference_events import freeze')==1
 seal=seal.replace('from reference_events import freeze','')
 namespace=dict(native.__dict__,freeze=ns['freeze'])
 exec(compile(seal,str(__file__)+'::compatible_seal','exec'),namespace)
 namespace['seal_score'](work)

def report():
 m=original_driver();source=inspect.getsource(m.report);assert source.count("arm['step']==500")==1
 source=source.replace("arm['step']==500","arm['step']==target(name)")
 ns=dict(m.__dict__,WORK=WORK,NAMES=NAMES,SEEDS=SEEDS,folder=folder,target=target);exec(compile(source,str(__file__),'exec'),ns);ns['report']()
 result=read(WORK/'COMPARISON.json');result.update(baseline_selection=read(WORK/'BASELINE_SELECTION.json'),automatic_promotion=False,best_model_selected=False)
 save(WORK/'COMPARISON.json',result)
def prepare():
 import adapter
 adapter.install()
 import eval_runtime as E
 from p2_combined_whisper_eval import link
 selection=read(WORK/'BASELINE_SELECTION.json');cluster=cluster_for('event020_feature010')
 for name in NAMES:
  cp=Path(selection['checkpoint']) if name=='baseline' else TRAIN/'runs/event020_feature010/step_001661.pt'
  first=folder(name,137);first.mkdir(parents=True,exist_ok=True)
  save(first/'REQUEST.json',dict(checkpoint=str(cp),samples=64),immutable=True)
  protocol=adapter.prepare(first)
  for seed in SEEDS:
   w=folder(name,seed);w.mkdir(parents=True,exist_ok=True)
   if seed!=137:
    for d in ('export','conditions','whisper','negative'):link(first/d,w/d)
    for f in ('REQUEST.json','DATA_CONFIG.json','shared_items.json'):save(w/f,read(first/f),immutable=True)
   p=copy.deepcopy(protocol);p.update(seed=seed,workers=28,cluster=str(w/'cluster.json'),metrics_cluster=str(w/'cluster.json'),worker_cli=str(WORK/'evaluate28.py'),shared_items_manifest=str(w/'shared_items.json'))
   save(w/'cluster.json',cluster);save(w/'protocol.json',p)
   for rank,rows in enumerate(partition(read(first/'shared_items.json'),28)):save(w/'shards'/f'{rank:02d}.json',rows)
 sources=source_identity();sources.update(E.code_identity());sources.update({str(f):sha(f) for f in WORK.glob('*.py')});sources[str(WORK/'BASELINE_SELECTION.json')]=sha(WORK/'BASELINE_SELECTION.json')
 save(WORK/'PLAN.json',dict(source=sources,cluster=cluster,candidates=list(NAMES),seeds=list(SEEDS),samples=64,outputs=256))
 save(WORK/'ROLLING_PLAN.json',dict(source=sources))
def run():
 from p2_loss_search28.resources import idle
 with (WORK/'handoff.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while not (TRAIN/'runs/event020_feature010/STAGE_1661.json').exists() or not (WORK/'BASELINE_SELECTION.json').exists():
   if (WORK/'STOP').exists():return
   save(WORK/'HANDOFF_STATUS.json',dict(stage='waiting_checkpoint_or_baseline_selection',at=time.time()));time.sleep(30)
  with (ORIGINAL/'queue.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if (WORK/'STOP').exists():return
   prepare()
   while True:
    snap=idle(cluster_for('event020_feature010'))
    if snap['idle']:break
    if (WORK/'STOP').exists():return
    save(WORK/'HANDOFF_STATUS.json',dict(stage='waiting_released_28_gpus',resources=snap,at=time.time()));time.sleep(30)
   save(WORK/'TRAINING_PAUSED_ACCEPTED.json',dict(success=True,at=time.time(),resources=snap))
   subprocess.run([str(PYTHON),'-u',str(WORK/'rolling_driver.py'),'run'],check=True)
if __name__=='__main__':
 import traceback
 try:run()
 except Exception as exc:save(WORK/'HANDOFF_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
