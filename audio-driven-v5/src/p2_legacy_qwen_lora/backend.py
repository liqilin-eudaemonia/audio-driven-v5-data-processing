"""Original unprocessed data and audited legacy parameter migration, never plan caches."""
from pathlib import Path
import torch
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json
from p2_listen.cache import load_part,tree_to
from p2_matrix16.config import feature_enabled,event_enabled

ARM_LOSS={'E0':'av','E1':'av_event','E0F':'av_feature','E1F':'av_event_feature'}

def parent_record(c):
    value=read(Path(c['legacy_parent'])/'run.json')
    if c['legacy_arm'] not in ARM_LOSS or ARM_LOSS[c['legacy_arm']]!=c['loss']:
        raise ValueError('Wrong historical arm')
    if c['text_condition']!='fixed' or c['stage']!=2 or c.get('plan_cache') is not None or c.get('teacher_cache') is not None:
        raise ValueError('Legacy continuation is fixed text, online Qwen and original caches only')
    if value['arm']!=('E1' if event_enabled(c) else 'E0') or value['feature_loss_enabled']!=feature_enabled(c):
        raise ValueError('Parent loss branch mismatch')
    if value['seed']!=c['training']['seed'] or value['budget']!=1268 or value['s0']:
        raise ValueError('Expected formal matched Stage1 parent')
    if value['text_lora']!={k:v for k,v in c['text_lora'].items() if k!='enabled'}:
        raise ValueError('LTX adapter configuration changed')
    return value

def data_config(c):
    from p2_listen.config import load,activate
    d=load(c['data_config']);parent=parent_record(c)
    if d!=parent['config'] or d['data']['size']!=512 or d['data'].get('audio_postprocess'):
        raise ValueError('Must use the exact original 50h data/config, without postprocessing')
    if d['training']['feature_loss_enabled']!=feature_enabled(c):raise ValueError('Teacher branch mismatch')
    activate(d);return d

def source_identity():
    from p2_matrix16.config import source_identity as shared
    result=shared()
    for p in [*Path(__file__).parent.glob('*.py'),ROOT/'scripts/p2_legacy_qwen_lora.py']:
        result[str(p)]=sha(p)
    return result

def audit_historical_sources(c):
    parent=parent_record(c)
    mapping=read(ROOT/'experiments/p2_train24_fourway_eval_20261027/source_restoration.json')[c['legacy_arm']]
    resolved={}
    for original,want in parent['source'].items():
        p=Path(original)
        if not p.is_file() or sha(p)!=want:
            item=mapping.get(original)
            if item is None or item['sha256']!=want:raise ValueError('Missing exact parent source: '+original)
            p=Path(item['path'])
        if sha(p)!=want:raise ValueError('Historical source checksum changed')
        resolved[original]=dict(path=str(p),sha256=want)
    return resolved

def data_identity(c):
    d=data_config(c);root=Path(d['output']);p=parent_record(c)
    return dict(schema='p2-original50h-data-v1',data_config_sha256=sha(c['data_config']),
        observations_file=sha(root/'manifests/observations.json'),targets_file=sha(root/'manifests/targets.json'),
        observations=p['observations'],targets=p['targets'],labels=digest(p['labels']),
        original_audio=True,feature_teacher='original_frozen_qwen' if feature_enabled(c) else None,
        cache_roots={part:str((root/'cache'/part).resolve()) for part in ['vae','events','text']+(['teacher'] if feature_enabled(c) else [])})

class Dataset:
    def __init__(self,c):
        from p2_matrix16.data import metadata_targets
        from p2_listen.config import extractor_config
        from p2_listen.features import load_text
        from transformers import Qwen2_5OmniProcessor
        self.spec=c;self.data=data_config(c);self.parent=parent_record(c);self.cfg=extractor_config(self.data)
        self.rows=read(Path(self.data['output'])/'manifests/observations.json')
        self.targets=metadata_targets(self.data,self.rows)
        if len(self.rows)!=30415 or digest(self.rows)!=self.parent['observations'] or digest(self.targets)!=self.parent['targets']:
            raise ValueError('Original 30,415-member data/order changed')
        self.processor=Qwen2_5OmniProcessor.from_pretrained(self.cfg.omni_path,local_files_only=True)
        self.text=load_text(self.data)

    def sample(self,index,device='cpu'):
        from p2_listen import features
        from p2_listen.layout import make_layout
        from p2_listen.annotations import validate,validate_contract
        o=self.rows[index];sid=o['sample_id'];target=self.targets[sid]
        label=read(Path(self.data['output'])/'labels/canonical'/(sid+'.json'))
        if digest(label)!=self.parent['labels'][index]:raise ValueError('Label changed since original training')
        validate(label,o);validate_contract(label,self.data)
        av=load_part(features.folder(self.data,'vae',sid),features.identity(self.data,'vae',o,target),device)
        events=load_part(features.folder(self.data,'events',sid),features.identity(self.data,'events',o,label=label),device)
        _,_,grid=features.image_grid(o,self.processor,self.cfg)
        layout=tree_to(make_layout(o['requested_duration_s'],grid,self.cfg.vision_fps,self.cfg.audio_hz),device)
        teacher=None
        if feature_enabled(self.spec):
            teacher=load_part(features.folder(self.data,'teacher',sid),features.identity(self.data,'teacher',o,target),device)
        return dict(condition=None,observation=o,plan=None,layout=layout,text=tree_to(self.text,device),av=av,events=events,teacher=teacher)

def map_parent_state(old,current):
    """Explicit rename only; require every existing module tensor and no extras."""
    mapped={}
    prefixes={'extractor.connector.':'connector.','extractor.video_decoder.':'decoders.video.',
              'extractor.audio_decoder.':'decoders.audio.','event_heads.':'event_heads.','text_lora.':'text_lora.'}
    for key,value in old.items():
        match=[p for p in prefixes if key.startswith(p)]
        if len(match)!=1:raise ValueError('Unmapped parent tensor: '+key)
        prefix=match[0];new=prefixes[prefix]+key[len(prefix):]
        if new in mapped:raise ValueError('Duplicate mapped tensor')
        mapped[new]=value
    expected={k for k in current if not k.startswith('qwen_lora.')}
    if set(mapped)!=expected:raise ValueError('Incomplete parent architecture migration')
    for key,value in mapped.items():
        if value.shape!=current[key].shape or not torch.isfinite(value).all():raise ValueError('Parent tensor shape/nonfinite mismatch')
    return {**current,**mapped}

def initialize_stage2(path,model,c,did):
    from p2_listen.training import checked_checkpoint
    from p2_listen.features import weights_identity
    path=Path(path).resolve()
    if path.parent!=Path(c['legacy_parent']).resolve() or path.name!='step_001268.pt':raise ValueError('Wrong historical checkpoint')
    saved=checked_checkpoint(path);parent=parent_record(c)
    if saved['identity']!=parent or saved['step']!=1268:raise ValueError('Parent run/checkpoint mismatch')
    if did!=data_identity(c):raise ValueError('Data identity mismatch')
    audit_historical_sources(c)
    if parent['assets']!={k:weights_identity(data_config(c),k) for k in ('omni','ltx','gemma')}:
        raise ValueError('Base assets differ from Stage1')
    if parent['text_lora_targets']!=model.text_targets:raise ValueError('LTX LoRA targets changed')
    model.load_state_dict(map_parent_state(saved['model'],model.state_dict()),strict=True)
    for adapter in model.qwen_lora:
        if torch.count_nonzero(adapter.b.weight):raise ValueError('New Qwen LoRA must initially preserve base outputs')
    return dict(checkpoint=str(path),sha256=sha(path),step=1268,experiment=c['legacy_arm'],
                migration='explicit legacy key rename; all existing module tensors preserved; new Qwen adapters only')

def preflight(c):
    """Validate unchanged labels/markers; reuse audited parent statistics, not full AV reads."""
    import time
    from p2_listen import features
    d=data_config(c);p=parent_record(c);root=Path(d['output']);sources=audit_historical_sources(c)
    from p2_matrix16.data import metadata_targets
    rows=read(root/'manifests/observations.json');targets=metadata_targets(d,rows)
    if len(rows)!=30415 or digest(rows)!=p['observations'] or digest(targets)!=p['targets']:raise ValueError('Parent data differs')
    # Historical arms have exactly the same ordered labels. Reuse their full audit
    # only when the physical label/cache roots and expected signatures agree.
    key=digest(dict(observations=p['observations'],targets=p['targets'],labels=p['labels'],
                    label_root=str((root/'labels/canonical').resolve()),
                    event_root=str((root/'cache/events').resolve()),vae_root=str((root/'cache/vae').resolve())))
    shared=Path(c['audit_root'])/('labels-'+key+'.json')
    if shared.exists():
        checked=read(shared)
        if checked['key']!=key or not checked['success']:raise ValueError('Invalid shared admission')
    else:
        for i,o in enumerate(rows):
            sid=o['sample_id'];label=read(root/'labels/canonical'/(sid+'.json'))
            if digest(label)!=p['labels'][i]:raise ValueError('Parent label changed: '+sid)
            for part in ('vae','events'):
                marker=read(root/'cache'/part/sid/'complete.json')
                expected=features.identity(d,part,o,targets[sid],label=label)
                if marker['identity']!=digest(expected):raise ValueError('Original cache identity changed: '+sid+'/'+part)
            if (i+1)%2000==0:print('legacy admission',i+1,len(rows),flush=True)
        atomic_json(shared,dict(key=key,success=True,samples=len(rows),at=time.time()))
    if feature_enabled(c):
        for o in rows:
            sid=o['sample_id'];marker=read(root/'cache/teacher'/sid/'complete.json')
            if marker['identity']!=digest(features.identity(d,'teacher',o,targets[sid])):raise ValueError('Original teacher changed')
    # Reject current assets/text changes before allocating any large model.
    if p['assets']!={k:features.weights_identity(d,k) for k in ('omni','ltx','gemma')}:raise ValueError('Pretrained assets changed')
    features.load_text(d)
    result=dict(schema='p2-matrix-preflight-v1',data_identity=data_identity(c),
        experiment_mode=[c['text_condition'],c['loss'],c['stage']],samples=len(rows),statistics=p['statistics'],
        source=source_identity(),historical_sources=sources,reused_parent_statistics=True)
    atomic_json(Path(c['output'])/'preflight.json',result);return result
