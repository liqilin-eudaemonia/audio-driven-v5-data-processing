from pathlib import Path
import sys, runpy, random, time, shutil
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,sha,atomic_json
from p2_clip_events.labels import LabelReader,tensors
from p2_clip_events.model import classes
from p2_ltx_events.numeric_labels import NumericLabels
from p2_ltx_events.config import validate_baseline,source_identity

out=Path(__file__).resolve().parent
c=read(ROOT/'configs/p2_ltx_hidden_clip_events_current500h_20261106.json')
c['clip_numeric_labels']=str(ROOT/'experiments/p2_clip_numeric_current500h_20261106')
c['output']=str(out)
validate_baseline(c,read(c['hidden_event_parent_config']))
labels=LabelReader(c['clip_labels']);ids=labels.ids();names=classes(c['event_supervision'])
numeric=NumericLabels(c['clip_numeric_labels'],labels,names,ids)
for name in names:
    expected=labels.manifest['statistics'].get(name,{'positive':0,'weak_negative':0})
    found=numeric.manifest['statistics'][name]
    for key in ('positive','weak_negative'):assert expected[key]==found[key],(name,key,expected,found)
indices=random.Random(20261106).sample(range(len(ids)),2048)
for i in indices:
    row=labels.row(ids[i]);old=tensors(row,names,labels.manifest['absence_weight'])
    new=numeric.events(i,ids[i],row['source_digest'])
    for name in names:
        for key in ('y','known','weight'):
            assert torch.equal(old[name][key],new['clip/'+name][key]),(ids[i],name,key)
bench=[]
for mode in ('sparse','numeric','numeric','sparse'):
    start=time.perf_counter()
    for i in indices:
        if mode=='sparse':v=tensors(labels.row(ids[i]),names,labels.manifest['absence_weight'])
        else:v=numeric.events(i,ids[i],numeric.arrays['source_digests'][i].decode('ascii'))
    bench.append(dict(mode=mode,seconds=time.perf_counter()-start,samples=len(indices)))
labels.conn.close()
cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='numeric_audit_import')
# Fail if the actual training sample path attempts a sparse label lookup.
with patch.object(LabelReader,'row',side_effect=AssertionError('Training attempted sparse label JSON lookup')):
    actual=cli['verify_cache'](c,3)
result=dict(success=True,samples=len(ids),classes=len(names),all_class_counts_match=True,
    exact_numeric_rows_checked=len(indices),real_cache_check=actual,
    sparse_label_lookup_forbidden_during_real_cache_check=True,benchmark=bench,
    benchmark_scope='CPU label read/convert only; not end-to-end training throughput',
    source_identity=source_identity(),numeric_manifest_sha256=sha(Path(c['clip_numeric_labels'])/'manifest.json'),
    cuda_initialized=torch.cuda.is_initialized(),training_started=False,
    evidence_conflict_audit='not_performed; numeric compilation preserves accepted sparse labels')
shutil.copy2('/tmp/p2_numeric_labels_tests.xml',out/'tests.xml')
atomic_json(out/'VALIDATION.json',result)
print({k:v for k,v in result.items() if k not in ('source_identity','real_cache_check')})
