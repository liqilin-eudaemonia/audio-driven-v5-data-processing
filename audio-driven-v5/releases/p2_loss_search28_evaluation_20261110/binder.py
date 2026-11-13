from pathlib import Path
import json,copy,time
from p2_testsets_eval.protocol import ROOT,read,save,sha
from p2_testsets_eval.runtime import ready
from p2_progressive_eval_20261108 import BASELINE,write_rows
class ShardBinder:
    """Reuse sealed native GT rows, changing only generated media paths.

    Canonical full native bind is rerun and compared before final aggregation.
    This permits early scoring without silently reducing the fixed200 set.
    """
    def __init__(self,work):
        self.work=work;self.p=read(work/'protocol.json');self.arm=self.p['arms'][0]['name']
        parent=read(BASELINE/'protocol.json');a=parent['arms'][0]['name']
        self.templates={};self.refs={};self.source={}
        for ds,spec in self.p['datasets'].items():
            assert spec['references_sha256']==sha(Path(spec['prepared'])/'references.jsonl')
            f=Path(spec['prepared'])/'references.jsonl';assert sha(f)==spec['references_sha256']
            refs=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
            t=BASELINE/'bindings'/a/ds/'pairs.jsonl'
            rows=[json.loads(s) for s in t.read_text().splitlines() if s.strip()]
            rows=[x for x in rows if x['id']in {v['id']for v in refs}]
            assert len(rows)==len(refs) and {x['id'] for x in rows}=={x['id'] for x in refs}
            assert read(t.parent/'binding_audit.json')['status']=='ready_for_scoring'
            self.templates[ds]=rows;self.refs[ds]=refs;self.source[ds]=dict(path=str(t),sha256=sha(t))
        self.items=read(work/'shared_items.json');self.by_key={x['key']:x for x in self.items}
        self.shards=[read(work/'shards'/f'{rank:02d}.json') for rank in range(self.p['workers'])]
        self.published=set();self.committed=set()
    def refresh(self):
        for key in self.by_key.keys()-self.committed:
            if ready(self.work/'generated'/self.arm/key):self.committed.add(key)
    def rows(self,ds,ids):
        from evaluation.listener_metrics.suite import enrich
        chosen=[];mapping=[]
        for original in self.templates[ds]:
            if original['id'] not in ids:continue
            key=ds+'/'+original['id'];assert key in self.committed
            item=self.by_key[key];folder=self.work/'generated'/self.arm/key;m=read(folder/'metadata.json');g=item['geometry']
            assert m['step']==self.p['arms'][0]['step'] and m['checkpoint_sha256']==self.p['arms'][0]['original_sha256']
            assert m['planning_adapter']==self.p['planner_adapter_identity']
            assert not m['listener_future_input'] and not m['target_description_access']
            e=m['export'];assert e['frames']==g['frames'] and abs(e['fps']-g['fps'])<1e-3
            assert (e['width'],e['height'])==(g['width'],g['height'])==(512,512)
            row=copy.deepcopy(original);row['generated']=str(folder/'generated.mp4');chosen.append(row)
            mapping.append(dict(id=row['id'],generated_audio=str(folder/'listener.wav')))
        assert {x['id'] for x in chosen}==set(ids)
        return enrich(chosen,self.refs[ds],mapping,self.work)
    def publish_ready(self):
        for rank,items in enumerate(self.shards):
            if rank in self.published or not all(x['key'] in self.committed for x in items):continue
            for ds in self.p['datasets']:
                ids={x['id'] for x in items if x['dataset']==ds};assert ids
                path=self.work/'metric_manifests'/self.arm/ds/f'{rank:02d}.jsonl'
                write_rows(path,self.rows(ds,ids))
            save(self.work/'progressive_bind'/f'{rank:02d}.json',dict(at=time.time(),rank=rank,
                complete_shard=True,ids=[x['key'] for x in items],gt_templates=self.source,
                no_test_subset_reduction=True))
            self.published.add(rank)
    def smoke(self):
        paths=[]
        for ds in self.p['datasets']:
            selected=[x['id'] for x in self.templates[ds] if ds+'/'+x['id'] in self.committed][:2]
            if len(selected)!=2:return False
            paths.append((ds,selected))
        for ds,ids in paths:
            write_rows(self.work/'smoke_metric_manifests'/self.arm/ds/'00.jsonl',self.rows(ds,set(ids)))
        save(self.work/'PROGRESSIVE_SMOKE_INPUTS.json',dict(at=time.time(),samples=dict(paths),
             rule='First two committed IDs in sealed reference order per dataset; validation only'))
        return True
    def canonical_audit(self):
        from p2_testsets_eval.cluster import bind
        assert len(self.committed)==self.p['total_samples']
        previous={}
        for rank in range(self.p['workers']):
            for ds in self.p['datasets']:
                f=self.work/'metric_manifests'/self.arm/ds/f'{rank:02d}.jsonl'
                previous[str(f)]=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
        bind(self.work)
        for path,expected in previous.items():
            actual=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
            assert actual==expected,'Progressive rows differ from full native binding: '+path
        save(self.work/'PROGRESSIVE_BIND_AUDIT.json',dict(success=True,at=time.time(),rows=self.p['total_samples'],
             all_manifests_equal_to_native_full_bind=True,templates=self.source))
