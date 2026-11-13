"""Common strict-published member subset; all original AV/event/teacher identities remain intact."""
from pathlib import Path
import copy
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json,EVENTS
from p2_matrix16.config import data_config,feature_enabled,source_identity as native_source
from p2_matrix16.data import Dataset as NativeDataset,metadata_targets


def source_identity():
    value=native_source()
    for path in [Path(__file__),ROOT/'scripts/p2_guided_chain_train.py']:value[str(path)]=sha(path)
    return value


def data_identity(c):
    from p2_matrix16.training import data_identity as native
    view=Path(c['guided_view']);manifest=read(view/'COMPLETE.json')
    return dict(**native(c),selected_observations=sha(view/'observations.json'),selected_manifest=sha(view/'COMPLETE.json'),samples=manifest['samples'])


class Dataset(NativeDataset):
    def __init__(self,c):
        from p2_plan_cache.reader import PlanCacheReader
        self.spec=c;self.data=data_config(c);self.plans=PlanCacheReader(c['plan_cache'])
        if self.plans.config!=self.data:raise ValueError('Plan/data view mismatch')
        view=Path(c['guided_view']);manifest=read(view/'COMPLETE.json');self.rows=read(view/'observations.json')
        if sha(view/'observations.json')!=manifest['observation_sha256'] or len(self.rows)!=manifest['samples'] or {o['sample_id'] for o in self.rows}!=set(self.plans.protocol['sample_ids']):raise ValueError('Selected members changed')
        original={o['sample_id']:o for o in read(Path(self.data['output'])/'manifests/observations.json')}
        if any(original[o['sample_id']]!=o for o in self.rows):raise ValueError('Selected observations altered')
        self.targets=metadata_targets(self.data,self.rows);self._negative=None;self.processor=None
        if c['stage']==2:
            from p2_listen.config import extractor_config
            from transformers import AutoProcessor
            self.cfg=extractor_config(self.data);self.processor=AutoProcessor.from_pretrained(self.cfg.omni_path,local_files_only=True)


def initialize_stage2(path,model,c,did):
    import torch
    from p2_matrix16.checkpoint import checked
    saved=checked(path);parent=saved['identity'];pc=parent['experiment']
    if c['stage']!=2 or pc['stage']!=1 or pc['id']!=c['parent_experiment'] or pc['text_condition']!=c['text_condition'] or pc['loss']!=c['loss']:raise ValueError('Wrong paired Stage1 parent')
    if saved['step']!=pc['training']['steps']:raise ValueError('Stage1 parent must have completed its full budget')
    if parent['source']!=source_identity() or parent['data_identity']!=did:raise ValueError('Parent source/member/cache identity mismatch')
    for key in ['loss_weights','text_lora','data_config','plan_cache','teacher_cache','guided_view']:
        if pc[key]!=c[key]:raise ValueError('Paired stage condition differs: '+key)
    state=model.state_dict();expected={k for k in state if not k.startswith('qwen_lora.')}
    if set(saved['model'])!=expected:raise ValueError('Parent architecture mismatch')
    for k,v in saved['model'].items():
        if state[k].shape!=v.shape:raise ValueError('Parent tensor shape mismatch')
        state[k]=v
    model.load_state_dict(state,strict=True)
    for adapter in model.qwen_lora:
        if torch.count_nonzero(adapter.b.weight):raise ValueError('Stage2 Qwen LoRA must start at zero output')
    return dict(checkpoint=str(Path(path).resolve()),sha256=sha(path),step=saved['step'],experiment=pc['id'])


def preflight(c,statistics):
    from p2_plan_cache.protocol import open_protocol
    p=open_protocol(c['plan_cache']);m=read(Path(c['guided_view'])/'COMPLETE.json')
    if len(p['entries'])!=m['samples']:raise ValueError('Plan member mismatch')
    if c['training']['steps']!=m['steps_per_stage'] or c['training']['global_batch']!=24:raise ValueError('One-epoch budget changed')
    if set(statistics)!=set(EVENTS):raise ValueError('Incomplete event statistics')
    value=dict(schema='p2-matrix-preflight-v1',data_identity=data_identity(c),experiment_mode=[c['text_condition'],c['loss'],c['stage']],samples=m['samples'],statistics=statistics,source=source_identity())
    atomic_json(Path(c['output'])/'preflight.json',value);return value
