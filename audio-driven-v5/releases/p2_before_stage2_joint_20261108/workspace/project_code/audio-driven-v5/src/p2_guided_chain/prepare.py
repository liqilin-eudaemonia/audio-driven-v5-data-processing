"""Validate selected cache identities once, then emit all eight immutable runs."""
from pathlib import Path
import copy,math,json
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json,EVENTS
from . import backend


def marker(folder,expected):
    folder=Path(folder);m=read(folder/'complete.json')
    if m['identity']!=digest(expected) or digest(m['contract'])!=m['identity']:raise ValueError('Cache identity differs: '+str(folder))
    if not (folder/'payload.pt').is_file():raise FileNotFoundError(folder/'payload.pt')
    return sha(folder/'complete.json')


def validate_selected(root):
    from p2_listen import features,cache
    from p2_plan_cache.protocol import open_protocol,sample
    from p2_plan_cache.runtime import dependencies,identity
    from p2_matrix16.teacher import identity as teacher_identity
    from p2_matrix16.data import metadata_targets
    from listen_audio_postprocess.overlay import sample_contract
    from p2_listen.config import load
    root=Path(root);spec=read(root/'pipeline.json');data=load(spec['data_config']);p=open_protocol(root/'plans')
    rows=read(root/'view/observations.json');targets=metadata_targets(data,rows)
    dep={k:dependencies(data,k) for k in ('text','input_plan')};totals={k:[0.,0] for k in EVENTS};markers={};memo={}
    def cached_sha(path):
        path=Path(path);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in memo:memo[key]=sha(path)
        return memo[key]
    native_weights=features.weights_identity;weights={}
    def cached_weights(c,kind):
        key=(digest(c),kind)
        if key not in weights:weights[key]=native_weights(c,kind)
        return weights[key]
    # Only cache small immutable source/asset metadata; media payload reads still
    # verify their recorded SHA when consumed by actual training.
    import p2_matrix16.teacher as teacher
    import listen_audio_postprocess.overlay as overlay
    import listen_audio_postprocess.recipes as recipes
    with ExitStack() as stack:
        for module in (features,teacher,overlay,recipes):stack.enter_context(patch.object(module,'sha',cached_sha))
        stack.enter_context(patch.object(features,'weights_identity',cached_weights))
        def check_one(o):
            sid=o['sample_id'];item=sample(root/'plans',p,sid);record={}
            for part in dep:record[part]=marker(root/'plans/cache'/part/sid,identity(p,item,part,dep[part]))
            base,folder,expected,recipe,label,ident=sample_contract(data,o,targets[sid])
            record['vae']=marker(folder,expected);record['audio']=marker(base/'cache/audio_postprocess'/sid,ident)
            ef=features.folder(data,'events',sid);ei=features.identity(data,'events',o,label=label)
            record['events']=marker(ef,ei);events=cache.load_part(ef,ei)
            counts={k:(float(events[k]['y'][events[k]['known']].sum()),int(events[k]['known'].sum())) for k in EVENTS}
            tr=Path(spec['teacher_cache']);record['teacher']=marker(tr/'cache'/sid,teacher_identity(data,o,targets[sid],tr))
            return sid,record,counts
        with ThreadPoolExecutor(max_workers=8) as workers:
            # Ordered reduction preserves the original floating-point summation order.
            for index,(sid,record,counts) in enumerate(workers.map(check_one,rows)):
                markers[sid]=record
                for k in EVENTS:
                    totals[k][0]+=counts[k][0];totals[k][1]+=counts[k][1]
                if (index+1)%100==0:
                    atomic_json(root/'validation_progress.json',dict(samples=index+1,total=len(rows),workers=8));print(json.dumps(dict(validation=index+1,total=len(rows))),flush=True)
    stats={};enabled=data['events']['K_train'];cap=data['events']['pos_weight_cap']
    for k,(pos,count) in totals.items():
        reason='not_requested' if k not in enabled else 'no_known_support' if count==0 else 'no_positive_support' if pos<=0 else 'no_negative_support' if pos>=count else None
        stats[k]=dict(positive=pos,negative=count-pos,count=count,enabled=reason is None,disabled_reason=reason,pos_weight=min((count-pos)/max(pos,1.),cap) if reason is None else 1.)
    report=dict(samples=len(rows),plan_sha256=digest(p),source=backend.source_identity(),statistics=stats,markers=markers,payload_policy='New writers validate payload and SHA; reused payload SHA checked lazily on every training read; event tensors verified in preflight')
    atomic_json(root/'CACHE_VALIDATED.json',report,immutable=True);return report


def prepare(root):
    root=Path(root);spec=read(root/'pipeline.json');manifest=read(root/'view/COMPLETE.json')
    if (root/'CACHE_VALIDATED.json').exists():
        report=read(root/'CACHE_VALIDATED.json')
        if report['source']!=backend.source_identity() or report['plan_sha256']!=sha_protocol(root):raise ValueError('Validated cache/source changed')
    else:report=validate_selected(root)
    configs=[]
    for loss in spec['losses']:
        for stage in (1,2):
            name=f'guided_{loss}_s{stage}';c=read(ROOT/'configs/p2_matrix16'/(name+'.json'))
            c.update(data_config=spec['data_config'],plan_cache=str(root/'plans'),teacher_cache=spec['teacher_cache'],guided_view=str(root/'view'),output=str(root/'runs'/name))
            c['training'].update(steps=manifest['steps_per_stage'],global_batch=24,seed=20261023,gradient_checkpointing='all')
            from p2_matrix16.config import validate
            validate(c,formal=True);path=root/'configs'/(name+'.json');atomic_json(path,c,immutable=True);backend.preflight(c,report['statistics'])
            configs.append(dict(id=name,config=str(path),stage=stage,loss=loss,parent=c['parent_experiment']))
    atomic_json(root/'configs/matrix.json',dict(samples=manifest['samples'],steps_per_stage=manifest['steps_per_stage'],experiments=configs),immutable=True)
    return configs


def sha_protocol(root):return digest(read(Path(root)/'plans/protocol.json'))
