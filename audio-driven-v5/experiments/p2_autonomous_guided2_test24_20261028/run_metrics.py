import os,sys,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=HERE/'run';REF=ROOT/'experiments/p2_reintegrated_final_eval_20261028'
sys.path[:0]=[str(ROOT/'src'),str(REF)]
os.environ['P2_EVAL_CLUSTER_WORK']=str(HERE/'cluster24')
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_autonomous_eval.protocol import open_protocol
from p2_autonomous_eval.evaluation import export_pairs
arm=sys.argv[1];p=open_protocol(WORK);pairs=export_pairs(WORK,arm);path=REF/'eval_metrics.py'
assert p['metric_adapter_sha256']==sha(path)
spec=importlib.util.spec_from_file_location('_native_autonomous_metrics',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=WORK;m.CLUSTER_WORK=HERE/'cluster24';atomic_json(WORK/'test24/observations.json',p['observations'],immutable=True)
m.evaluate(arm,'test24')
folder=pairs.parent;g=read(folder/'glare_results.json');g.update(target_annotation_access=False,gt_media_access_for_scoring=True,plan_source='qwen_autonomous');atomic_json(folder/'glare_results.json',g)
marker=read(folder/'COMPLETE.json');marker.update(annotation_oracle=False,plan_source='qwen_autonomous',generation_target_access=False,gt_media_access_for_scoring=True,protocol_sha256=digest(p),results={n:sha(folder/n) for n in ['omni/results.json','glare_results.json']});atomic_json(folder/'COMPLETE.json',marker)
