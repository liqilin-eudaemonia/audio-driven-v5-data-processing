"""Compose per-sample conditions, postprocessed AV and optional new teacher."""
from pathlib import Path
from p2_listen.contracts import read,digest
from p2_listen.cache import load_part,tree_to
from .config import data_config,feature_enabled

def metadata_targets(data, observations):
    """Accepted manifest metadata; payload hashes are verified at cache reads."""
    from p2_listen.contracts import target
    raw={r['sample_id']:r for r in read(Path(data['output'])/'manifests/targets.json')}
    return {o['sample_id']:target(raw[o['sample_id']],o,verify=False) for o in observations}


class Dataset:
    def __init__(self,c):
        from p2_plan_cache.reader import PlanCacheReader
        self.spec=c;self.data=data_config(c);self.plans=PlanCacheReader(c['plan_cache'])
        if self.plans.config!=self.data:raise ValueError('Plan cache/data view mismatch')
        self.rows=read(Path(self.data['output'])/'manifests/observations.json')
        if set(self.plans.protocol['sample_ids'])!={o['sample_id'] for o in self.rows}:raise ValueError('Plan cache must cover the entire selected training subset')
        self.targets=metadata_targets(self.data,self.rows)
        self._negative=None
        self.processor=None
        if c['stage']==2:
            from p2_listen.config import extractor_config
            from transformers import AutoProcessor
            self.cfg=extractor_config(self.data)
            self.processor=AutoProcessor.from_pretrained(self.cfg.omni_path,local_files_only=True)

    def sample(self,index,device='cpu'):
        from p2_plan_cache.protocol import sample as plan_sample
        from p2_listen.features import load_text,image_grid
        from p2_listen.layout import make_layout
        from listen_audio_postprocess.overlay import sample_contract,compose
        from p2_listen import features
        o=self.rows[index];sid=o['sample_id'];target=self.targets[sid]
        guided=self.spec['text_condition']=='guided'
        if self.spec['stage']==1:
            value=self.plans.training_sample(sid,target,'A2' if guided else 'A0',device)
        else:
            # No input/input_plan payload is accessed on the online LoRA path.
            item=plan_sample(self.plans.work,self.plans.protocol,sid)
            text=self.plans.part(sid,'text',device)['positive'] if guided else load_text(self.data,device)['positive']
            root,folder,expected,recipe,label,ident=sample_contract(self.data,o,target)
            av=compose(load_part(folder,expected,device),load_part(root/'cache/audio_postprocess'/sid,ident,device))
            events=load_part(features.folder(self.data,'events',sid),features.identity(self.data,'events',o,label=label),device)
            # Coordinates are source-only and identical to the accepted event cache layout.
            _,_,grid=image_grid(o,self.processor,self.cfg)
            layout=tree_to(make_layout(o['requested_duration_s'],grid,self.cfg.vision_fps,self.cfg.audio_hz),device)
            value=dict(condition=None,layout=layout,text=dict(positive=text),av=av,events=events,
                observation=o,plan=item['guidance']['prompt'] if guided else None,teacher=None)
        if feature_enabled(self.spec):
            from .teacher import identity
            root=Path(self.spec['teacher_cache'])
            teacher=load_part(root/'cache'/sid,identity(self.data,o,target,root),device)
            value['teacher']={k:v for k,v in teacher.items() if k in ('video','audio','video_valid','audio_valid')}
        return value
