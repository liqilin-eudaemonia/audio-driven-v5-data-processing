"""GT is opened only in this post-generation scoring adapter, never by planning."""
from pathlib import Path
import importlib.util,sys
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json
from p2_glare_eval.common import artifact_valid
from .protocol import open_protocol,read_plan

REFERENCE=ROOT/'experiments/p2_reintegrated_final_eval_20261028'

def export_pairs(work,arm):
    work=Path(work);p=open_protocol(work);jobs=[j for j in p['jobs'] if j['arm']==arm]
    if len(jobs)!=2*len(p['observations']):raise ValueError('Unknown/incomplete arm')
    # Access allowed only after generation is complete; this reference is identical to the oracle comparison.
    verified=[]
    for j in jobs:
        out=Path(j['output']);marker=read(out/'complete.json');contract=marker['contract'];plan=read_plan(work,p,j['sample_id'])
        if contract.get('job')!=j or contract.get('protocol_sha256')!=digest(p) or contract.get('plan_sha256')!=digest(plan):raise ValueError('Output does not match autonomous protocol')
        if not artifact_valid(out,contract):raise ValueError('Generation incomplete')
        meta=read(out/'completed.json')
        if meta['target_annotation_access'] is not False or meta['gt_media_access'] is not False or meta['plan_source']!='qwen_autonomous':raise ValueError('Oracle output in autonomous evaluation')
        verified.append((j,out))
    reference=read(ROOT/'experiments/p2_test24_fourway_eval_20261027/E0/omni_pairs.json')['samples']
    by={r['sample_id']:r for r in reference};pairs=[]
    for j,out in verified:
        ref=by[j['sample_id']]
        pairs.append(dict(sample_id=j['sample_id'],sample_seed=j['seed'],gt_video=ref['gt_video'],gt_audio=ref['gt_audio'],
                          generated_video=str(out/'generated.mp4'),generated_audio=str(out/'listener.wav')))
    path=work/'metrics'/arm/'test24/pairs.json'
    atomic_json(path,dict(schema='p2-omni-pairs-v1',samples=pairs),immutable=True)
    return path


def evaluate(work,arm):
    work=Path(work);p=open_protocol(work);pairs=export_pairs(work,arm)
    path=REFERENCE/'eval_metrics.py'
    if p['metric_adapter_sha256']!=sha(path):raise ValueError('Pinned metric adapter changed')
    # Run in its own process. The pinned adapter owns all metric implementations;
    # its scoring/detection children use the existing shared content-addressed cache.
    sys.path.insert(0,str(REFERENCE))
    spec=importlib.util.spec_from_file_location('_autonomous_native_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=work;module.CLUSTER_WORK=work/'no_external_metric_queue'
    atomic_json(work/'test24/observations.json',p['observations'],immutable=True)
    module.evaluate(arm,'test24')
    folder=pairs.parent;g=read(folder/'glare_results.json')
    g.update(target_annotation_access=False,gt_media_access_for_scoring=True,plan_source='qwen_autonomous')
    atomic_json(folder/'glare_results.json',g)
    marker=read(folder/'COMPLETE.json');marker.update(annotation_oracle=False,plan_source='qwen_autonomous',generation_target_access=False,
          gt_media_access_for_scoring=True,protocol_sha256=digest(p),results={n:sha(folder/n) for n in ('omni/results.json','glare_results.json')})
    atomic_json(folder/'COMPLETE.json',marker);return marker
