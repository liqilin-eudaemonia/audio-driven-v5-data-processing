"""Independent restartable planner, Gemma and full-sequence Qwen caches."""
from pathlib import Path
from p2_listen.contracts import atomic_json,digest,observation,sha
from .protocol import open_protocol,selected,lock,plan_identity,read_plan
from .planning import generate_attempt


def plan(work,shard_index=0,shards=1,device='cuda:0'):
    from p2_listen.config import extractor_config
    from p2_listen.features import unload
    from p2_plan_cache.runtime import dependencies
    work=Path(work);p=open_protocol(work);dep=dependencies(p['config'],'input_plan');cfg=extractor_config(p['config'])
    runtime=processor=None;rows=[]
    try:
        for obs0 in selected(p,shard_index,shards):
            sid=obs0['sample_id'];folder=work/'plans'/sid
            try:
                with lock(work/'locks/plan'/(sid+'.lock')):
                    obs=observation(obs0);expected=plan_identity(p,obs,dep)
                    if (folder/'complete.json').exists():
                        read_plan(work,p,sid,expected);rows.append(dict(sample_id=sid,status='reused'));continue
                    if runtime is None:
                        from omni_av_p2.data import load_omni
                        runtime,processor=load_omni(cfg,device)
                    error=None;attempts=[];result=None
                    for i in range(p['planner']['max_attempts']):
                        result,audit=generate_attempt(runtime,processor,obs,p['planner'],i,error)
                        attempts.append(audit);atomic_json(folder/'attempts.json',attempts)
                        if result is not None:break
                        error=audit['error']
                    if result is None:raise ValueError('Planner validation failed; no GT/annotation fallback: '+str(error))
                    atomic_json(folder/'plan.json',result)
                    (folder/'prompt.txt').write_text(result['prompt'])
                    atomic_json(folder/'complete.json',dict(identity=expected,protocol_sha256=digest(p),
                        files={n:sha(folder/n) for n in ('plan.json','prompt.txt','attempts.json')}))
                    rows.append(dict(sample_id=sid,status='planned',prompt_sha256=result['prompt_sha256']))
            except Exception as exc:
                row=dict(sample_id=sid,status='failed',error_type=type(exc).__name__,error=str(exc));rows.append(row)
                atomic_json(work/'errors/plan'/(sid+'.json'),row)
    finally:
        del runtime,processor;unload()
    result=dict(success=all(r['status']!='failed' for r in rows),results=rows,shard_index=shard_index,shards=shards)
    atomic_json(work/'reports'/f'plan.{shard_index}-of-{shards}.json',result);return result


def part_identity(p,plan,part,dependencies):
    return dict(schema='p2-autonomous-conditions-v1',protocol_sha256=digest(p),plan_sha256=digest(plan),
                prompt_sha256=plan['prompt_sha256'],part=part,dependencies=dependencies,
                target_annotation_access=False,hidden_scope='all_input_and_plan_tokens')


def cache(work,part,shard_index=0,shards=1,device='cuda:0'):
    from p2_listen.cache import load_part,save_part
    from p2_listen.config import extractor_config
    from p2_listen.features import unload
    from p2_plan_cache.runtime import dependencies,validate_payload
    from p2_plan_cache.encoding import encode_text,extract_planned_input
    if part not in ('text','input_plan'):raise ValueError('Unknown cache part')
    work=Path(work);p=open_protocol(work);c=p['config'];dep=dependencies(c,part);planner_dep=dependencies(c,'input_plan');cfg=extractor_config(c)
    model=processor=None;rows=[]
    try:
        for obs0 in selected(p,shard_index,shards):
            sid=obs0['sample_id']
            try:
                with lock(work/'locks'/part/(sid+'.lock')):
                    obs=observation(obs0);result=read_plan(work,p,sid,plan_identity(p,obs,planner_dep))
                    ident=part_identity(p,result,part,dep);folder=work/'cache'/part/sid
                    value=load_part(folder,ident,missing_ok=True)
                    if value is None:
                        if model is None:
                            if part=='text':
                                from omni_av.ltx_runtime import NativeTextConditioner
                                model=NativeTextConditioner(cfg,device)
                            else:
                                from omni_av_p2.data import load_omni
                                model,processor=load_omni(cfg,device)
                        if part=='text':
                            positive,meta=encode_text(model,result['prompt'],p['planner']['max_text_tokens'])
                            value=dict(positive=positive,prompt_sha256=result['prompt_sha256'],**meta)
                        else:
                            value=extract_planned_input(c,obs,result['prompt'],model,processor,p['planner']['max_qwen_tokens'])
                        validate_payload(value,dict(guidance=result['guidance']),part);save_part(folder,ident,value)
                    validate_payload(value,dict(guidance=result['guidance']),part)
                    rows.append(dict(sample_id=sid,status='ready',prompt_sha256=result['prompt_sha256']))
                    del value
            except Exception as exc:
                row=dict(sample_id=sid,status='failed',error_type=type(exc).__name__,error=str(exc));rows.append(row)
                atomic_json(work/'errors'/part/(sid+'.json'),row)
    finally:
        if part=='text' and model is not None:model.close()
        del model,processor;unload()
    result=dict(success=all(r['status']!='failed' for r in rows),part=part,results=rows)
    atomic_json(work/'reports'/f'{part}.{shard_index}-of-{shards}.json',result);return result


def assert_compatible(cache_config,native_config):
    from p2_listen.config import extractor_config
    keys=('omni_path','gemma_path','ltx_path','hidden_size','hidden_layers','video_dim','audio_dim','vision_fps','audio_hz','vision_max_pixels')
    a=extractor_config(cache_config);b=extractor_config(native_config)
    if any(getattr(a,k)!=getattr(b,k) for k in keys):raise ValueError('Native checkpoint and autonomous encoding contracts differ')
    if cache_config['generation']!=native_config['generation']:raise ValueError('Sampling configuration differs from paired evaluation')


class Reader:
    def __init__(self,work):
        self.work=Path(work);self.protocol=open_protocol(work);self.config=self.protocol['config'];self.deps={}

    def dependency(self,part):
        from p2_plan_cache.runtime import dependencies
        if part not in self.deps:self.deps[part]=dependencies(self.config,part)
        return self.deps[part]

    def plan(self,sid):
        obs=next(o for o in self.protocol['observations'] if o['sample_id']==sid)
        observation(obs)
        return read_plan(self.work,self.protocol,sid,plan_identity(self.protocol,obs,self.dependency('input_plan')))

    def part(self,sid,part,plan):
        from p2_listen.cache import load_part
        from p2_plan_cache.runtime import validate_payload
        value=load_part(self.work/'cache'/part/sid,part_identity(self.protocol,plan,part,self.dependency(part)))
        validate_payload(value,dict(guidance=plan['guidance']),part);return value

    def conditions(self,sid,native_config,device='cpu'):
        from p2_listen.cache import tree_to
        from p2_listen.features import load_text
        assert_compatible(self.config,native_config)
        from omni_av_p2.connector import ConditionInputs
        plan=self.plan(sid)
        text=self.part(sid,'text',plan);hidden=self.part(sid,'input_plan',plan)
        if hidden['prompt']!=text['positive']['prompt'] or hidden['prompt']!=plan['prompt']:
            raise ValueError('Planner/Qwen/Gemma prompt mismatch')
        # Native negative CFG remains unchanged for each checkpoint.
        negative=load_text(native_config)['negative']
        return dict(condition=ConditionInputs(**hidden['condition']).to(device),layout=tree_to(hidden['layout'],device),
                    text=tree_to(dict(positive=text['positive'],negative=negative),device),plan=plan)


def verify(work,shard_index=0,shards=1):
    reader=Reader(work);rows=[]
    for obs in selected(reader.protocol,shard_index,shards):
        sid=obs['sample_id']
        try:
            plan=reader.plan(sid);text=reader.part(sid,'text',plan);hidden=reader.part(sid,'input_plan',plan)
            if hidden['prompt']!=text['positive']['prompt']:raise ValueError('Qwen/Gemma plan differs')
            rows.append(dict(sample_id=sid,status='passed',prompt_sha256=plan['prompt_sha256']))
        except Exception as e:rows.append(dict(sample_id=sid,status='failed',error=str(e)))
    result=dict(success=all(r['status']=='passed' for r in rows),results=rows)
    atomic_json(Path(work)/'reports'/f'verify.{shard_index}-of-{shards}.json',result);return result
