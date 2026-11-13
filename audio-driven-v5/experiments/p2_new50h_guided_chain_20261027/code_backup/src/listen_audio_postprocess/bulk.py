"""Fixed-subset official metadata, approved filters, and auditable cache checks."""
import hashlib
import json
import os
import tarfile
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf
from p2_listen.contracts import read,sha,atomic_json,digest
from listen_benchmark.backchannel_face_stats import SourceReader
from listen_benchmark.official_audio_gate import make_gate
from listen_benchmark.official_event_alignment import align_events
from listen_benchmark.chinese_asr_filter import filter_labels


def official(raw,start,duration):
    # Use the exact conversion exercised by the approved fixed-100 review.
    import sys
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
    from compare_listener_official_six import official as convert
    for key in ('metadata:transcript','metadata:vad'):
        if key not in raw or not isinstance(raw[key],list):
            raise ValueError(f'Missing official field {key}; missing metadata is not silence')
    return convert(raw,'listener',start,duration)


def derive(job,raw,provenance):
    info=sf.info(job['target_audio'])
    if abs(info.frames/info.samplerate-job['duration_s'])>1/info.samplerate+1e-6:
        raise ValueError('Native audio duration mismatch')
    off=official(raw,job['start'],job['duration_s'])
    from compare_listener_official_six import match
    for word in off['words']:
        word['our_event_matches']=match(word,job['events'])
    _,gate=make_gate(off,info.frames,info.samplerate)
    aligned=align_events(job['events'],off,gate)
    result,_,support=filter_labels(aligned,gate)
    audio=[e for e in result['events'] if e['modality']=='audio']
    record=dict(sample_id=job['sample_id'],source_audio_sha256=job['source_audio_sha256'],
                official_listener=off,audio_events=audio,
                extra_mute_intervals=[dict(start=a/info.samplerate,end=b/info.samplerate) for a,b in result['language_filter']['mute_intervals_samples']],
                provenance=dict(official=provenance,source_integrated_sha256=job['integrated_sha256'],
                                clip_absolute_start=job['start'],clip_duration_s=job['duration_s']))
    audit=dict(sample_id=job['sample_id'],original_audio_events=len(job['events']),retained_audio_events=len(audio),
               removed_chinese_events=len(result['language_filter']['removed_event_ids']),
               retained_audio_support_s=int((support>0).sum())/info.samplerate,
               full_silence_support=not bool(support.any()),
               excluded_audio_events=result['excluded_audio_events'],language_filter=result['language_filter'],
               audio_event_alignment=result['audio_event_alignment'])
    return record,audit


def build_records(dataset,base,feature_run,run,workers=24):
    run=Path(run);dataset=Path(dataset);base=Path(base)
    obs=read(base/'manifests/observations.json');targets={x['sample_id']:x for x in read(base/'manifests/targets.json')}
    ids=[o['sample_id'] for o in obs];byid={o['sample_id']:o for o in obs}
    assert len(ids)==len(set(ids))==30415
    assert set(ids)==set((dataset/'sample_ids.txt').read_text().splitlines())
    assert obs==read(dataset/'manifests/observations.json')
    assert all(o['official_split']==o['experiment_split']=='train' for o in obs)
    groups=defaultdict(lambda:defaultdict(list));reader=SourceReader(feature_run);seen=set()
    with (dataset/'integrated_labels.jsonl').open('rb') as f:
        for line in f:
            label=json.loads(line);sid=label['sample_id'];assert sid in byid and sid not in seen;seen.add(sid)
            full=reader.get(label['source']);o=byid[sid];t=targets[sid]
            assert full['split']=='train' and abs(full['end']-full['start']-o['requested_duration_s'])<=1/48000+1e-6
            source=full['sources']['listener'];member=str(Path(source['member']).with_suffix('.json'))
            job=dict(sample_id=sid,target_audio=t['target_audio'],source_audio_sha256=t['media_hashes']['target_audio'],
                     duration_s=o['requested_duration_s'],start=full['start'],integrated_sha256=hashlib.sha256(line).hexdigest(),
                     events=[e for e in label['events'] if e.get('modality')=='audio'])
            groups[source['archive']][member].append(job)
    reader.close();assert seen==set(ids)
    atomic_json(run/'scope.json',dict(samples=len(ids),duration_s=sum(o['requested_duration_s'] for o in obs),
                                     archives=len(groups),listener_metadata_members=sum(map(len,groups.values())),
                                     sample_ids_sha256=digest(ids),all_train=True))
    done=0;totals=Counter();started=time.monotonic()
    def archive_work(item):
        archive,members=item;st=Path(archive).stat();found=set();outputs=[]
        # Plain tar seeks past video payloads; never extracts whole archives.
        with tarfile.open(archive,'r:') as tar:
            for member in tar:
                if member.name not in members:continue
                if member.name in found:raise ValueError('Duplicate requested archive member')
                raw_bytes=tar.extractfile(member).read();raw=json.loads(raw_bytes)
                provenance=dict(archive=archive,member=member.name,sha256=hashlib.sha256(raw_bytes).hexdigest(),
                                archive_size=st.st_size,archive_mtime_ns=st.st_mtime_ns)
                key=digest([archive,member.name]);atomic_json(run/'official_metadata'/f'{key}.json',dict(provenance=provenance,raw=raw),immutable=True)
                for job in members[member.name]:
                    record,audit=derive(job,raw,provenance);sid=job['sample_id']
                    atomic_json(run/'records'/f'{sid}.json',record,immutable=True)
                    atomic_json(run/'audits'/f'{sid}.json',audit,immutable=True)
                    outputs.append({k:audit[k] for k in ('original_audio_events','retained_audio_events','removed_chinese_events','retained_audio_support_s','full_silence_support')})
                found.add(member.name)
                if len(found)==len(members):break
        if found!=set(members):raise FileNotFoundError(f'{archive}: missing {set(members)-found}')
        after=Path(archive).stat()
        if (after.st_size,after.st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise ValueError('Source archive changed')
        return outputs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outputs in pool.map(archive_work,groups.items()):
            done+=len(outputs)
            for row in outputs:totals.update(row)
            atomic_json(run/'preparation_progress.json',dict(completed=done,total=len(ids),seconds=time.monotonic()-started))
    assert done==len(ids)
    temp=run/'records.jsonl.tmp'
    with temp.open('w') as out:
        for sid in ids:out.write(json.dumps(read(run/'records'/f'{sid}.json'),ensure_ascii=False)+'\n')
        out.flush();os.fsync(out.fileno())
    os.replace(temp,run/'records.jsonl')
    report=dict(samples=done,counts=dict(totals),records_sha256=sha(run/'records.jsonl'),seconds=time.monotonic()-started)
    atomic_json(run/'records_complete.json',report);return report


def cache_worker(run,rank):
    import torch
    from p2_listen.config import load,activate
    from p2_listen import features
    from p2_listen.annotations import rasterize,validate,validate_contract
    from p2_listen.cache import save_part
    from .overlay import build_one
    from .encoder import NativeAudioEncoder
    run=Path(run);c=load(run/'view/config.json');activate(c);torch.set_num_threads(2)
    jobs=read(run/'tasks'/f'{rank:03d}.json');encoder=None;results=[];started=time.monotonic()
    def get_encoder():
        nonlocal encoder
        if encoder is None:encoder=NativeAudioEncoder(c,'cuda:0')
        return encoder
    for job in jobs:
        o,t=job['observation'],job['target'];sid=o['sample_id']
        wait_start=time.monotonic()
        while not all((run/'view'/part/f'{sid}.json').exists() for part in ['audio_postprocess/recipes','labels/canonical']):
            if (run/'error-run.json').exists() or time.monotonic()-wait_start>600:
                raise RuntimeError('Preparation failed or timed out; source not replaced')
            time.sleep(.2)
        r=build_one(c,o,t,get_encoder)
        if r.get('busy'):raise RuntimeError('Unexpected duplicate worker ownership')
        label=read(run/'view/labels/canonical'/f'{sid}.json');validate(label,o);validate_contract(label,c)
        _,layout=features.load_input(c,o,'cpu')
        save_part(features.folder(c,'events',sid),features.identity(c,'events',o,label=label),rasterize(label,layout,c['events']['K_train']))
        results.append(r)
        if len(results)%10==0:
            atomic_json(run/'workers'/f'{rank:03d}.json',dict(completed=len(results),total=len(jobs),seconds=time.monotonic()-started,finished=False))
    atomic_json(run/'workers'/f'{rank:03d}.json',dict(completed=len(results),total=len(jobs),seconds=time.monotonic()-started,finished=True,
                reused=sum(x.get('reused',False) for x in results),audio_encoded=sum(x.get('audio_encoded',False) for x in results)))


def audit_all(run,workers=16):
    import torch
    torch.set_num_threads(1)
    from p2_listen.config import load,activate
    from p2_listen.cache import load_part
    from p2_listen import features
    from p2_listen.training_data import load_training_sample
    from .overlay import sample_contract
    from .recipes import masks
    run=Path(run);c=load(run/'view/config.json');activate(c)
    base=Path(c['data']['audio_postprocess']['base_output']);obs=read(run/'view/manifests/observations.json')
    targets={x['sample_id']:x for x in read(run/'view/manifests/targets.json')}
    assert obs==read(base/'manifests/observations.json')
    assert list(targets.values())==read(base/'manifests/targets.json')
    assert len(obs)==30415
    sampled=set(range(0,len(obs),max(1,len(obs)//100)));totals=Counter();rows=[]
    def one(pair):
        i,o=pair;sid=o['sample_id'];t=targets[sid]
        _,folder,expected,recipe,label,identity=sample_contract(c,o,t)
        original=read(base/'labels/canonical'/f'{sid}.json')
        assert digest(original)==label['provenance']['original_canonical_label_sha256']
        assert [e for e in label['events'] if e['type']!='listener_vocalization']==[e for e in original['events'] if e['type']!='listener_vocalization']
        assert sha(recipe['source']['path'])==recipe['source']['sha256']
        patch=load_part(run/'view/cache/audio_postprocess'/sid,identity)
        cached_events=load_part(features.folder(c,'events',sid),features.identity(c,'events',o,label=label))
        assert 'av' not in patch and 'video' not in patch
        if not patch.get('reuse_base_audio'):assert torch.isfinite(patch['audio_latents']['latents']).all()
        record=read(run/'records'/f'{sid}.json');source_events={e['event_id']:e for e in record['audio_events']};actual=defaultdict(list)
        for e in label['events']:
            if e['type']!='listener_vocalization':continue
            eid=e['source']['audio_postprocess']['source_event_id'];ref=source_events[eid]
            assert e['description']==ref['description'] and e.get('text')==ref.get('text')
            actual[eid].append(dict(start=e['start'],end=e['end']))
        assert set(actual)==set(source_events)
        assert all(actual[k]==v['intervals'] for k,v in source_events.items())
        gate,extra,_=masks(recipe);support=gate*extra
        sr=recipe['source']['sample_rate']
        for spans in actual.values():
            for x in spans:
                a,b=round(x['start']*sr),round(x['end']*sr)
                assert b>a and (support[a:b]>0).all()
        for k,v in cached_events.items():
            if isinstance(v,dict):
                for tensor in v.values():
                    if isinstance(tensor,torch.Tensor):assert torch.isfinite(tensor).all()
        if i in sampled:load_training_sample(c,o,t,'cpu')
        classes={e['type'] for e in label['events']}
        row=dict(sample_id=sid,audio_events=len(source_events),training_audio_segments=sum(len(x) for x in actual.values()),has_audio_labels=bool(actual),
                 full_silence_support=not bool(support.any()),unchanged_audio=bool(patch.get('reuse_base_audio')),
                 audio_payload_bytes=(run/'view/cache/audio_postprocess'/sid/'payload.pt').stat().st_size,
                 training_read_checked=i in sampled)
        return row,classes
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row,classes in pool.map(one,enumerate(obs)):
            rows.append(row);totals.update({k:1 for k in classes})
            if len(rows)%500==0:atomic_json(run/'audit_progress.json',dict(completed=len(rows),total=len(obs)))
    summary=dict(status='passed',samples=len(rows),duration_s=sum(o['requested_duration_s'] for o in obs),
                 retained_audio_events=sum(r['audio_events'] for r in rows),with_audio_labels=sum(r['has_audio_labels'] for r in rows),
                 without_audio_labels=sum(not r['has_audio_labels'] for r in rows),full_silence_support=sum(r['full_silence_support'] for r in rows),
                 unchanged_audio=sum(r['unchanged_audio'] for r in rows),class_sample_counts=dict(totals),
                 audio_payload_bytes=sum(r['audio_payload_bytes'] for r in rows),training_read_checked=sum(r['training_read_checked'] for r in rows),
                 all_source_audio_hashes_verified=True,all_original_text_and_visual_labels_verified=True,
                 wav_files_written=0,video_cache_copies=0,training_started=False)
    assert len(totals)==7
    atomic_json(run/'audit.json',dict(summary=summary,records=rows))
    atomic_json(run/'COMPLETE.json',dict(summary=summary,config_sha256=sha(run/'view/config.json'),audit_sha256=sha(run/'audit.json'),code_manifest_sha256=sha(run/'code_manifest.json')))
    return summary
