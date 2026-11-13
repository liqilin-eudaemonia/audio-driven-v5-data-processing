"""Backbone Event training and CPU-only historical interval preparation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,atomic_json
from p2_ltx_events.config import DEFAULT,TEMPORAL_CLASSES,CURRENT_TEMPORAL_CLASSES,load,validate_baseline
from p2_ltx_events import VERSION


def emit(parent,labels,config,run_output):
    c=read(parent)
    c.update(output=str(Path(run_output).resolve()),hidden_event_experiment=VERSION,
        hidden_event_parent_config=str(Path(parent).resolve()),temporal_labels=str(Path(labels).resolve()),
        hidden_event_supervision=dict(readout='last_block_before_output',classes=TEMPORAL_CLASSES,
            clip_weight=1.,temporal_weight=1.,background_weight=.1,label_policy='historical-weak-intervals-v1'))
    validate_baseline(c,read(parent));atomic_json(config,c,immutable=True)
    return dict(config=str(config),output=c['output'],training_started=False)


def emit_current(parent,cache_index,source,labels,config,run_output,with_temporal=False):
    import csv,math
    from p2_ltx_events.current import checked_index
    from p2_j_duration_execution import make_pairs
    from p2_listen.contracts import sha
    index=checked_index(cache_index);c=read(parent);source=Path(source).resolve()
    m=read(source/'manifest.json')
    if m['current_cache_index_sha256']!=sha(cache_index):
        raise ValueError('Current label manifests do not bind the cache index')
    if with_temporal:
        if labels is None:raise ValueError('Temporal mode requires prepared temporal labels')
        tm=read(Path(labels)/'manifest.json')
        if tm['cache_manifest_sha256']!=sha(source/'manifest.json'):raise ValueError('Temporal source mismatch')
    c.update(output=str(Path(run_output).resolve()),hidden_event_experiment=VERSION,
        hidden_event_parent_config=str(Path(parent).resolve()),
        current_cache_index=str(Path(cache_index).resolve()),current_source=str(source),
        j_cache=index['caches']['qwen_hidden']['work'],plan_cache=index['caches']['qwen_hidden']['work'],
        clip_labels=str(source/'clip_labels'),teacher_cache=index['caches']['listener_teacher']['work'],
        whisper_cache=index['caches']['speaker_whisper']['work'],
        hidden_event_supervision=dict(readout='last_block_before_output',clip_weight=1.,temporal_weight=0.))
    c.pop('temporal_labels',None)
    if with_temporal:
        c['temporal_labels']=str(Path(labels).resolve())
        c['hidden_event_supervision'].update(classes=CURRENT_TEMPORAL_CLASSES,
            clip_weight=.5,temporal_weight=.5,background_weight=.1,label_policy='historical-weak-intervals-v1')
    c['training'].update(global_batch=32,steps=math.ceil(index['samples']/32))
    with (source/'sample_list.csv').open() as stream:rows=[dict(requested_duration_s=float(r['duration_s'])) for r in csv.DictReader(stream)]
    path=Path(c['output'])/'duration.schedule.json';list_sha=sha(source/'sample_list.csv')
    plan=dict(pairs=make_pairs(rows,c['training']['steps'],32,c['training']['seed'],0,'duration',512),sample_list_sha256=list_sha)
    atomic_json(path,plan,immutable=True)
    c['duration_sampling']=dict(policy='duration',window=512,start_step=0,manifest=str(path),sha256=sha(path),sample_list_sha256=list_sha)
    validate_baseline(c,read(parent));atomic_json(config,c,immutable=True)
    return dict(config=str(config),output=c['output'],samples=index['samples'],source_split_counts=index['source_split_counts'],
        steps=c['training']['steps'],global_batch=32,training_started=False)


def verify_cache(c,limit=3):
    import torch
    from p2_ltx_events.data import Dataset,data_identity
    torch.set_num_threads(2);dataset=Dataset(c);rows=[]
    order=[0]
    if c.get('current_cache_index'):
        order.append(next(i for i,r in enumerate(dataset.rows) if r['split']=='dev'))
        restored=dataset.source.conn.execute("SELECT sid FROM samples WHERE instr(value, '\"historical_source\":')>0 ORDER BY ordinal LIMIT 1").fetchone()
        if restored:
            order.append(next(i for i,r in enumerate(dataset.rows) if r['sample_id']==restored[0]))
    order=list(dict.fromkeys(order));seen=set(order)
    order.extend(i for i in range(len(dataset.rows)) if i not in seen)
    for i in order[:limit]:
        value=dataset.sample(i,'cpu');value['condition'].validate();value['whisper'].validate()
        from p2_ltx_events.layout import _position_strategy,check_prepared
        from omni_av.ltx_runtime import with_text
        from ltx_trainer.timestep_samplers import UniformTimestepSampler
        with torch.random.fork_rng(devices=[]):
            prepared=_position_strategy().prepare_training_inputs(with_text(value['av']['av'],value['text']['positive']),UniformTimestepSampler(.2,.8))
        check_prepared(value['hidden_event_layout'],prepared)
        for mod in ('video','audio'):
            actual=getattr(prepared,mod+'_loss_mask') & value['av']['support'][mod]
            if not torch.equal(actual,value['hidden_event_layout'][mod+'_token_valid']):
                raise ValueError('Current target mask differs from native generated tokens: '+mod)
        record=dict(sample_id=value['sample_id'],source_split=value.get('source_split'),known={k:int(v['known'].sum()) for k,v in value['events'].items()},
            positive={k:float(v['y'][v['known']].sum()) for k,v in value['events'].items()},
            intervals={m:value['hidden_event_layout'][m+'_intervals'].shape[1] for m in ('video','audio')},
            native_renderer_input_and_event_layout_match=True)
        rows.append(record)
    result=dict(success=True,samples=rows,total_training_samples=len(dataset.rows),data_identity=data_identity(c),
        cuda_initialized=torch.cuda.is_initialized(),training_started=False)
    atomic_json(Path(c['output'])/'CPU_CACHE_CHECK.json',result);return result


def commands(cluster,config,resume=None,stop_after=None):
    from p2_j_s1_queue import commands as native
    result=native(cluster,config,resume=resume,stop_after=stop_after)
    for command in result:
        command[command.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(Path(__file__).resolve())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['index-history','freeze-current','prepare','emit','emit-current','inspect','verify-cache','preflight','commands','train','launch'])
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--parent',type=Path,default=ROOT/'experiments/p2_j_whisper_event_feature_train31_20261105/config.json')
    p.add_argument('--source',type=Path,default=ROOT/'experiments/p2_full_audio_postprocess_crop384_20261026')
    p.add_argument('--history',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--cache-index',type=Path,default=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/TRAINING_CACHE_INDEX.json')
    p.add_argument('--evidence-from',type=Path,help='Reuse the exact verified source interval evidence for clock reprojection only')
    p.add_argument('--run-output',type=Path,default=ROOT/'experiments/p2_ltx_hidden_clip_events_current500h_20261106')
    p.add_argument('--with-temporal',action='store_true',help='Explicitly opt into the historical clip+temporal experiment')
    p.add_argument('--cluster',type=Path);p.add_argument('--processes',type=int,default=8)
    p.add_argument('--limit',type=int,default=3);p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int);p.add_argument('--logs',type=Path)
    a=p.parse_args()
    if a.processes<1 or a.limit<1:p.error('processes and limit must be positive')
    if a.command=='index-history':
        from p2_ltx_events.labels import index_history
        if a.output is None:p.error('index-history requires --output')
        result=index_history(a.source,a.output,a.processes)
        result={k:v for k,v in result.items() if k!='files'}
    elif a.command=='freeze-current':
        from p2_ltx_events.current import freeze_current
        if a.output is None:p.error('freeze-current requires --output')
        result=freeze_current(a.cache_index,a.output,a.processes)
    elif a.command=='prepare':
        from p2_ltx_events.labels import prepare
        if any(x is None for x in (a.cache,a.history,a.output)):p.error('prepare requires --cache --history --output')
        result=prepare(a.cache,a.history,a.output,a.processes,evidence_from=a.evidence_from)
    elif a.command=='emit-current':
        if a.cache is None:p.error('emit-current requires --cache current label sources')
        if a.with_temporal and a.output is None:p.error('--with-temporal requires --output temporal labels')
        result=emit_current(a.parent,a.cache_index,a.cache,a.output,a.config,a.run_output,a.with_temporal)
    elif a.command=='emit':
        if a.output is None:p.error('emit requires --output pointing at prepared temporal labels')
        result=emit(a.parent,a.output,a.config,a.run_output)
    else:
        c=load(a.config);validate_baseline(c,read(c['hidden_event_parent_config']))
        if a.command=='inspect':
            from p2_ltx_events.heads import names
            result=dict(config=str(a.config),event_classes=names(c),hidden_event_supervision=c['hidden_event_supervision'],
                temporal_labels=c.get('temporal_labels'),training_started=False)
        elif a.command=='verify-cache':result=verify_cache(c,a.limit)
        elif a.command=='commands':
            if a.cluster is None:p.error('commands requires --cluster')
            result=commands(read(a.cluster),a.config,a.resume,a.stop_after)
        else:
            from p2_ltx_events.training import preflight,train
            if a.command=='preflight':result=preflight(c,a.processes)
            elif a.command=='train':result=train(c,a.resume,a.stop_after)
            else:
                if a.cluster is None or a.logs is None:p.error('launch requires --cluster --logs')
                if (Path(c['output'])/'STOP').exists():raise ValueError('Explicit output STOP exists')
                if (Path(c['output'])/'run.json').exists() and not a.resume:raise ValueError('Explicit resume required')
                from listen_cluster import launch
                preflight(c,a.processes)
                launch(read(a.cluster),commands(read(a.cluster),a.config,a.resume,a.stop_after),a.logs)
                result=read(Path(c['output'])/'status.json')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
