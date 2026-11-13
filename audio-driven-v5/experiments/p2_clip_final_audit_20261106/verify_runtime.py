"""Actual five-cache CPU prefetch, native layouts, corrected labels and schedule."""
from pathlib import Path
import sys, json, random, dataclasses
from collections import Counter
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,atomic_json,digest,sha
from p2_ltx_events.data import Dataset,statistics,data_identity
from p2_ltx_events.schedule import Schedule
from p2_ltx_events.layout import _position_strategy,check_prepared
from p2_ltx_events.config import validate_baseline,source_identity
from p2_clip_events.labels import LabelReader,tensors
from p2_whisper_event_feature.execution import CPUInputStream
from ltx_trainer.timestep_samplers import UniformTimestepSampler
OUT=Path(__file__).resolve().parent;c=read(OUT/'candidate_config.json');validate_baseline(c,read(c['hidden_event_parent_config']))
torch.set_num_threads(2);d=Dataset(c);plan=Schedule(c)
from omni_av.ltx_runtime import with_text
counts=Counter(i for i,o in plan.pairs)
assert set(counts)==set(range(len(d.rows))) and min(counts.values())==1 and max(counts.values())==2
assert len(plan.pairs)==c['training']['steps']*32
# Prove the union of 32 per-rank selections reconstructs each global batch.
for step in range(1,c['training']['steps']+1):
    found=[plan.indices(len(d.rows),step,32,c['training']['seed'],rank,32)[0] for rank in range(32)]
    assert found==plan.pairs[(step-1)*32:step*32]
order=[0,min(range(len(d.rows)),key=lambda i:float(d.rows[i]['duration_s'])),max(range(len(d.rows)),key=lambda i:float(d.rows[i]['duration_s']))]
order.append(next(i for i,r in enumerate(d.rows) if r['split']=='dev'))
sid=d.source.conn.execute("SELECT sid FROM samples WHERE instr(value, '\"historical_source\":')>0 ORDER BY ordinal LIMIT 1").fetchone()[0]
order.append(next(i for i,r in enumerate(d.rows) if r['sample_id']==sid))
audit=read(OUT/'LABEL_AUDIT.json')
for values in audit['examples'].values():order.extend(v['ordinal'] for v in values[:2])
order.extend(random.Random(20261107).sample(range(len(d.rows)),24));order=list(dict.fromkeys(order))
expected={i:tensors(d.labels.row(d.rows[i]['sample_id']),d.names,d.labels.manifest['absence_weight']) for i in order}
def finite(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=='cpu'
        if value.is_floating_point():assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):finite(getattr(value,field.name))
rows=[]
with patch.object(LabelReader,'row',side_effect=AssertionError('Sparse label read in training path')):
    with CPUInputStream(d,((step,[(i,step)]) for step,i in enumerate(order,1)),depth=2) as stream:
        for i,pack in zip(order,stream,strict=True):
            sample=pack['samples'][0];assert sample['sample_id']==d.rows[i]['sample_id'];finite(sample)
            sample['condition'].validate();sample['whisper'].validate()
            for name,row in expected[i].items():
                for key in ('y','known','weight'):assert torch.equal(sample['events']['clip/'+name][key],row[key])
            with torch.random.fork_rng(devices=[]):
                prepared=_position_strategy().prepare_training_inputs(with_text(sample['av']['av'],sample['text']['positive']),UniformTimestepSampler(.2,.8))
            check_prepared(sample['hidden_event_layout'],prepared)
            for mod in ('video','audio'):
                assert torch.equal(getattr(prepared,mod+'_loss_mask')&sample['av']['support'][mod],sample['hidden_event_layout'][mod+'_token_valid'])
            rows.append(dict(sample_id=sample['sample_id'],split=sample['source_split'],duration_s=d.rows[i]['duration_s'],read_seconds=pack['read_seconds']))
            print(json.dumps(dict(checked=len(rows),total=len(order))),flush=True)
stats=statistics(c,d.labels.manifest)
result=dict(success=True,samples=rows,total_samples=len(d.rows),enabled_classes=[k for k,v in stats.items() if v['enabled']],
    schedule=dict(steps=c['training']['steps'],global_batch=32,sample_presentations=len(plan.pairs),unique_samples=len(counts),tail_repeats=len(plan.pairs)-len(counts),all_32_rank_partitions_verified=True),
    all_tensors_finite=True,native_layouts_match=True,cpu_prefetch_depth=2,sparse_label_lookup_forbidden=True,
    data_identity=data_identity(c),source_identity=source_identity(),configuration_digest=digest(c),
    config_sha256=sha(OUT/'candidate_config.json'),cuda_initialized=torch.cuda.is_initialized(),training_started=False)
atomic_json(OUT/'RUNTIME_CHECK.json',result)
print(json.dumps(dict(success=True,real_samples=len(rows),enabled_classes=len(result['enabled_classes']),schedule=result['schedule'])),flush=True)
