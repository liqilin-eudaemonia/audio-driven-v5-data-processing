"""Prepare an isolated training/cache view, retaining original source media paths."""
import copy
from pathlib import Path
from p2_listen.contracts import read,sha,digest,atomic_json,path_at,safe_id
from p2_listen.config import load,feature_loss_enabled
from p2_listen.annotations import validate,validate_contract
from .recipes import make_recipe,SCHEMA
from .overlay import VERSION


def review_records(review_path):
    data=read(review_path)
    if data.get('schema')!='official-chinese-filtered-review100-v1':raise ValueError('Expected the approved final review format')
    source_sha=sha(review_path)
    for c in data['cases']:
        rate=c['gate']['sample_rate']
        yield dict(sample_id=c['sample_id'],source_audio_sha256=c['gate']['source_audio_sha256'],
                   official_listener=c['official']['listener'],audio_events=c['aligned_audio_events'],
                   extra_mute_intervals=[dict(start=a/rate,end=b/rate) for a,b in c['language_filter']['mute_intervals_samples']],
                   provenance=dict(review_run=data['run_id'],review_sha256=source_sha))


def adapt_canonical(original,record,recipe,obs,config):
    label=copy.deepcopy(original);validate(label,obs);validate_contract(label,config)
    if obs['experiment_split']!='train':raise ValueError('Automatic weak-label derivation is restricted to train; dev/test need reviewed references')
    current={e['event_id']:e for e in record['audio_events']}
    if len(current)!=len(record['audio_events']):raise ValueError('Duplicate candidate event IDs')
    original_ids={e['source'].get('event_id') for e in label['events'] if e['type']=='listener_vocalization'}
    if set(current)-original_ids:raise ValueError('Final audio event missing from canonical source; do not silently add events')
    result=[]
    for e in label['events']:
        if e['type']!='listener_vocalization':result.append(e);continue
        final=current.get(e['source'].get('event_id'))
        if final is None:continue
        if final['description']!=e['description'] or final.get('text')!=e.get('text'):raise ValueError('Postprocessing must preserve original text')
        for i,x in enumerate(final['intervals']):
            new=copy.deepcopy(e);new.update(start=x['start'],end=x['end'])
            if len(final['intervals'])>1:new['event_id']=e['event_id']+f'-part{i+1:03d}'
            new['source']['audio_postprocess']=dict(recipe_sha256=digest(recipe),original_training_interval=[e['start'],e['end']],
                                                    original_training_event_id=e['event_id'],support_segment=i,
                                                    source_event_id=final['event_id'],alignment=final.get('boundary_alignment'),
                                                    count_new_utterance=False)
            result.append(new)
    label['events']=result
    label['provenance'].update(audio_recipe_sha256=digest(recipe),audio_postprocess_policy=VERSION,
                               original_canonical_label_sha256=digest(original),final_audio_records_sha256=digest(record['audio_events']))
    # Keep the established weak-coverage/unknown contract; do not infer new negatives.
    validate(label,obs);validate_contract(label,config)
    return label


def prepare_view(base_config_path,records,output,workers=1,streaming_records_sha256=None):
    base=load(base_config_path)
    if feature_loss_enabled(base):raise ValueError('Prepare an AV/Event config with feature_loss_enabled=false first')
    if base['data'].get('audio_postprocess'):raise ValueError('Nested audio overlays are unsupported')
    old=path_at(base['output']);root=Path(output).resolve()
    if root==old or root.is_relative_to(old):raise ValueError('Use an independent output directory')
    obs={o['sample_id']:o for o in read(old/'manifests/observations.json')}
    targets={t['sample_id']:t for t in read(old/'manifests/targets.json')}
    root.mkdir(parents=True,exist_ok=True)
    def publish(observed,target_rows,manifest):
        atomic_json(root/'manifests/observations.json',observed,immutable=True)
        atomic_json(root/'manifests/targets.json',target_rows,immutable=True)
        policy_path=root/'audio_postprocess/policy_manifest.json'
        atomic_json(policy_path,manifest,immutable=True)
        (root/'cache').mkdir(exist_ok=True)
        for part in ('input','text'):
            source=old/'cache'/part;dest=root/'cache'/part
            if not source.is_dir():raise FileNotFoundError(source)
            if dest.is_symlink():
                if dest.resolve()!=source.resolve():raise ValueError('Unexpected cache link')
            elif dest.exists():raise FileExistsError(dest)
            else:dest.symlink_to(source,target_is_directory=True)
        config=copy.deepcopy(base);config['output']=str(root)
        config['data']['audio_postprocess']=dict(schema=VERSION,base_output=str(old),policy_manifest_sha256=sha(policy_path))
        atomic_json(root/'config.json',config,immutable=True)
    streaming_manifest=None
    if streaming_records_sha256:
        if len(streaming_records_sha256)!=64:raise ValueError('Expected complete records SHA256')
        streaming_manifest=dict(schema=VERSION,recipe_schema=SCHEMA,base_config_sha256=sha(base_config_path),base_output=str(old),
                                samples=len(obs),source_records_sha256=streaming_records_sha256,
                                sample_ids_sha256=digest(list(obs)),training_started=False)
        publish(list(obs.values()),[targets[sid] for sid in obs],streaming_manifest)
    observed=[];target_rows=[];entries=[];seen=set()
    def prepare_one(record):
        sid=safe_id(record['sample_id'])
        o,t=obs[sid],targets[sid]
        if record['source_audio_sha256']!=t['media_hashes']['target_audio']:raise ValueError('Official metadata bound to different listener audio')
        recipe=make_recipe(sid,t['target_audio'],t['media_hashes']['target_audio'],record['official_listener'],record['extra_mute_intervals'],record.get('provenance'))
        old_label=read(old/'labels/canonical'/f'{sid}.json');label=adapt_canonical(old_label,record,recipe,o,base)
        atomic_json(root/'audio_postprocess/recipes'/f'{sid}.json',recipe,immutable=True)
        atomic_json(root/'labels/canonical'/f'{sid}.json',label,immutable=True)
        return o,t,dict(sample_id=sid,recipe_sha256=digest(recipe),canonical_label_sha256=digest(label))
    def checked_records():
        for record in records:
            sid=safe_id(record['sample_id'])
            if sid in seen:raise ValueError('Duplicate sample in postprocess input')
            seen.add(sid)
            yield record
    if workers<1:raise ValueError('workers must be positive')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded batches avoid retaining the complete rich event input in memory.
        import itertools
        iterator=iter(checked_records())
        while batch:=list(itertools.islice(iterator,workers*4)):
            for o,t,entry in pool.map(prepare_one,batch):
                observed.append(o);target_rows.append(t);entries.append(entry)
    if not entries:raise ValueError('No postprocess records')
    if streaming_manifest is not None:
        if [x['sample_id'] for x in observed]!=list(obs):raise ValueError('Streaming record scope/order differs from frozen base')
        manifest=streaming_manifest
    else:
        manifest=dict(schema=VERSION,recipe_schema=SCHEMA,base_config_sha256=sha(base_config_path),base_output=str(old),
                      samples=len(entries),entries_sha256=digest(entries),training_started=False)
    atomic_json(root/'audio_postprocess/index.json',entries,immutable=True)
    publish(observed,target_rows,manifest)
    return dict(output=str(root),config=str(root/'config.json'),samples=len(entries),wav_files_written=0,video_cache_copies=0,training_started=False)
