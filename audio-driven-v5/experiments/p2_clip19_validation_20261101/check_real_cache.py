"""Real J cache/label packing plus production Connector auxiliary backward on CPU."""
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p2_clip_events.backend import Dataset, selected_statistics
from p2_clip_events.model import mask_classes
from p2_matrix16.config import load
from p2_matrix16.model import MatrixModel
from p2_matrix16.training import objective
from p2_listen.config import extractor_config
from p2_listen.contracts import atomic_json

torch.set_num_threads(8)
work=Path(__file__).parent
started=time.time();results=[];backwards=[]
for variant in ('fixed','guided'):
    c=load(ROOT/'configs/p2_j_clip_events_20261101'/f'{variant}_av_event_s1.json');data=Dataset(c)
    stats=selected_statistics(c,data.labels.manifest);enabled=[k for k,v in stats.items() if v['enabled']]
    model=None
    for i in range(24):
        s=data.sample(i);assert len(s['events'])==19;s['events']=mask_classes(s['events'],enabled)
        assert all(v['y'].shape==(1,) and v['known'].shape==(1,) for v in s['events'].values())
        assert s['condition'].hidden.shape[0]==1 and 'av' in s['av'] and 'support' in s['av']
        results.append(dict(sample_id=data.rows[i]['sample_id'],variant=variant,
                            positive=sum(int(v['y'].item()) for v in s['events'].values())))
        if i<2:
            if model is None:
                torch.manual_seed(20261101);model=MatrixModel(c,extractor_config(data.data))
                assert len(data.rows)==285638
                assert model.event_heads.heads['video'][-1].out_features==16
                assert model.event_heads.heads['audio'][-1].out_features==3
                assert model.connector.memory.projection.out_features==1024
                assert len(model.connector.blocks)==4
                assert all(b.cross.num_heads==16 for b in model.connector.blocks)
                assert sum(p.numel() for p in model.connector.parameters())==96688128
            optimizer=torch.optim.AdamW(model.groups());optimizer.zero_grad()
            # Isolate the new auxiliary path. Real cached AV packing is checked above;
            # this does not run the 22B LTX generator or claim a GPU training test.
            def av(u,pack,text):return u.video.float().sum()*0+u.audio.float().sum()*0,{}
            with torch.autocast('cpu',dtype=torch.bfloat16):
                value=model(s,{k:v['pos_weight'] for k,v in stats.items()},torch.Generator(),av)
                total,_,event=objective(value,c,value['event_counts'],1,1)
            total.backward()
            norms={name:sum(float(p.grad.double().square().sum()) for p in module.parameters() if p.grad is not None)**.5
                   for name,module in [('connector',model.connector),('head',model.event_heads)]}
            assert norms['connector']>0 and norms['head']>0
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            backwards.append(dict(variant=variant,sample_id=data.rows[i]['sample_id'],loss=float(event.detach()),gradient_norms=norms))
            del optimizer,value,total,event
        del s
    del model,data
result=dict(success=True,samples=24,classes=19,condition_reads=len(results),results=results,production_connector_backwards=backwards,
            connector=dict(width=1024,depth=4,heads=16,parameters=96688128),device='cpu',real_ltx_generation_training_tested=False,elapsed_s=time.time()-started)
atomic_json(work/'real_cache_check.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
