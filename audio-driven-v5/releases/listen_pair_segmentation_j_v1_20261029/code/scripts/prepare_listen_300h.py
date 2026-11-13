"""Build a fresh listening corpus from original Seamless pairs, with exclusive groups."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
os.environ['PATH'] = str(ROOT/'scripts/media_bin') + os.pathsep + os.environ['PATH']
sys.path.insert(0, str(ROOT / 'src'))
from p3_listen.common import digest, read, write, write_once, sha
from p3_listen.dataset import windows, validate
from p3_listen.prompts import contract
from p3_listen import subturns
import build_seamless_listening_preview as media
import build_seamless_rule_turns as rules

OUT = ROOT / 'data_corpora/seamless_listen_300h'
INDEX = Path('/private/liqi_workspace/data/pipeline/seamless/outputs_demo100/pair_index_all.json')
SEED = 20261021


def eligible_clip(clip):
    return clip['usable'] and not clip['crop_boxes']['listener'].get('wide_crop',False)


def qualified_seconds(plan):
    return sum(c['unique_training_seconds'] for c in plan.get('clips',[]) if eligible_clip(c))


def person_keys(ident):
    v, s, _, p = ident.split('_')
    # A/B suffixes must not make one participant appear independent.
    match = re.fullmatch(r'(P\d+)[A-Za-z]*', p)
    if not match: raise ValueError(ident)
    return v + '_' + s, v + '_' + match[1]


def catalog():
    pairs = read(INDEX)
    parent = list(range(len(pairs))); owners = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i,p in enumerate(pairs):
        for role in ('a', 'b'):
            for key in person_keys(p[role]['id']):
                if key in owners: parent[find(i)] = find(owners[key])
                owners[key] = i
    buckets = defaultdict(list)
    for i,p in enumerate(pairs): buckets[find(i)].append(p)
    groups, selected, excluded = [], [], Counter()
    for items in buckets.values():
        splits = sorted({p['split'] for p in items})
        if len(splits) != 1 or splits[0] == 'test':
            excluded['+'.join(splits)] += len(items); continue
        sessions = sorted({person_keys(p[k]['id'])[0] for p in items for k in ('a','b')})
        people = sorted({person_keys(p[k]['id'])[1] for p in items for k in ('a','b')})
        gid = 'component_' + digest([sessions, people])[:20]
        groups.append(dict(group_id=gid, sessions=sessions, participants=people, split=splits[0],
                           pairs=sorted(p['label']+'_'+p['interaction'] for p in items)))
        for p in items:
            if any(p[k]['split'] != p['split'] or p[k]['label'] != p['label'] for k in ('a','b')):
                excluded['cross_subset_pair'] += 1; continue
            selected.append(dict(p, pair=p['label']+'_'+p['interaction'],group_id=gid))
    train = sorted((p for p in selected if p['split']=='train'), key=lambda p:digest([SEED,p['pair']]))
    dev = sorted((p for p in selected if p['split']=='dev'), key=lambda p:digest([SEED,p['pair']]))
    result = dict(index=str(INDEX), index_sha256=sha(INDEX), groups=groups, train=train, dev=dev,
                  excluded_pairs=dict(excluded), policy='only unmixed official train/dev components; all test-touching components excluded')
    write_once(OUT/'catalog.json',result)
    print(dict(train_pairs=len(train),dev_pairs=len(dev),excluded=dict(excluded)),flush=True)
    return result


def load_person(record):
    import numpy as np
    import soundfile as sf
    with tarfile.open(record['archive'],'r:') as tar:
        by = {m.name:m for m in tar if m.isfile()}
        refs = {}
        for ext in ('.json','.mp4','.wav','.npz'):
            name=record['id']+ext; m=by[name]
            refs[ext]=dict(archive=record['archive'],member=name,offset=m.offset_data,byte_size=m.size)
        info=json.load(tar.extractfile(by[record['id']+'.json']))
        with np.load(tar.extractfile(by[record['id']+'.npz']),allow_pickle=False) as arrays:
            keypoints=arrays['boxes_and_keypoints:keypoints']
        with sf.SoundFile(tar.extractfile(by[record['id']+'.wav'])) as audio:
            audio_samples=len(audio); sample_rate=audio.samplerate
    probe=json.loads(media.run(['ffprobe','-v','error','-select_streams','v:0',
                   '-show_streams','-of','json',media.uri(refs['.mp4'])]))['streams'][0]
    a,b=map(int,probe['avg_frame_rate'].split('/'));fps=a/b
    duration=float(probe['duration']); frames=int(probe.get('nb_frames',round(duration*fps)))
    if abs(len(keypoints)-frames)>1 or abs(duration-audio_samples/sample_rate)>.08:
        raise ValueError('Source AV/keypoint length disagreement')
    annotation=dict(id=record['id'],transcript=info.get('metadata:transcript',[]),vad=info.get('metadata:vad',[]))
    if not annotation['transcript']: raise ValueError('Missing transcript')
    geometry=dict(id=record['id'],source=refs['.mp4'],duration=min(duration,audio_samples/sample_rate),
                  fps=fps,width=int(probe['width']),height=int(probe['height']),
                  arrays={'boxes_and_keypoints:keypoints':keypoints})
    source=dict(id=record['id'],source=refs['.mp4'],duration=geometry['duration'],fps=fps,
                width=geometry['width'],height=geometry['height'],audio_samples=audio_samples,sample_rate=sample_rate)
    return annotation,geometry,source


def plan_pair(p):
    path=OUT/'plans'/p['split']/(p['pair']+'.json')
    if path.exists(): return read(path)['summary']
    try:
        loaded=[load_person(p[k]) for k in ('a','b')]
        records=[x[0] for x in loaded]; geom=[x[1] for x in loaded];sources=[x[2] for x in loaded]
        annotated=rules.annotate(dict(pair=p['pair'],subset=p['label'],sources=sources),dict(records=records))
        people=[rules.prepare(r,i) for i,r in enumerate(records)]
        clips=[];rejected=[]
        for ci,c in enumerate(annotated['clips']):
            a,b=c['start'],c['end'];si=c['speaker_index'];li=c['listener_index']
            words=[w for w in people[si]['words'] if w['start']<b-.001 and w['end']>a+.001]
            if not words:continue
            a=max(a,min(w['start'] for w in words));b=min(b,max(w['end'] for w in words))
            if b-a<4:continue
            try:
                # Keep the same fixed crop over the entire parent turn.
                crop={role:rules.full_turn_crop(geom[i],a,b) for role,i in (('speaker',si),('listener',li))}
            except ValueError as e:
                rejected.append(dict(start=a,end=b,reason=str(e)));continue
            events=[dict(e,start=max(a,e['start']),end=min(b,e['end'])) for e in c['listener_events'] if e['end']>a and e['start']<b]
            # Include VAD-only feedback when selecting safe split boundaries.
            activity=events+[dict(start=max(a,x),end=min(b,y)) for x,y in people[li]['vad'] if y>a and x<b]
            for j,part in enumerate(subturns.cut_points(a,b,words,activity)):
                x,y=part['start'],part['end']
                ws=windows(x,y)
                usable=bool(y-x>=4 and ws)
                covered=ws[-1]['end']-ws[0]['start'] if usable else 0.
                clips.append(dict(start=x,end=y,duration=y-x,speaker=c['speaker'],listener=c['listener'],
                    speaker_index=si,listener_index=li,crop_boxes=crop,
                    path=f'media/{p["pair"]}/clip_{ci:04d}_sub{j:02d}',
                    parent_start=a,parent_end=b,parent_id=ci,cut=part,kind=c['kind'],
                    listener_events=[e for e in events if e['end']>x and e['start']<y],
                    usable=usable,unique_training_seconds=covered))
        # No double-counting intervals, including role switches.
        intervals=sorted((c['start'],c['end']) for c in clips)
        if any(y>x2+1e-7 for (_,y),(x2,_) in zip(intervals,intervals[1:])):
            raise ValueError('Overlapping canonical listening intervals')
        summary=dict(pair=p['pair'],split=p['split'],clips=len(clips),
                     seconds=sum(c['unique_training_seconds'] for c in clips),rejected=len(rejected),ok=True)
        speech=[media.overlap(q['vad'],0,s['duration']) for q,s in zip(people,sources)]
        write(path,dict(pair=p['pair'],group_id=p['group_id'],split=p['split'],subset=p['label'],
                        sources=sources,clips=clips,speech_seconds=speech,rejected=rejected,summary=summary))
    except Exception as e:
        summary=dict(pair=p['pair'],split=p['split'],ok=False,seconds=0,error=type(e).__name__+': '+str(e))
        write(path,dict(summary=summary))
    return summary


def run_plan(a):
    c=read(OUT/'catalog.json')
    rows=c['train'][:a.train_pairs]+c['dev'][:a.dev_pairs]
    rows=rows[a.node::3]
    counts=Counter();seconds=Counter();errors=[];begin=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(plan_pair,rows),1):
            counts[r['split']]+=1; seconds[r['split']]+=r['seconds']
            if not r['ok']:errors.append(r)
            if i%20==0 or i==len(rows):
                report=dict(node=a.node,completed=i,total=len(rows),hours={k:v/3600 for k,v in seconds.items()},
                            failures=len(errors),seconds=time.monotonic()-begin)
                write(OUT/f'plan_progress_{a.node}.json',report);print(report,flush=True)
    write(OUT/f'plan_errors_{a.node}.json',errors)


def select(a):
    c=read(OUT/'catalog.json'); chosen=[]; totals={}
    for split,target in (('train',a.hours),('dev',a.dev_hours)):
        seconds=0.
        for p in c[split]:
            path=OUT/'plans'/split/(p['pair']+'.json')
            if not path.exists():continue
            plan=read(path)
            if not plan['summary']['ok'] or qualified_seconds(plan)<=0:continue
            chosen.append(dict(pair=p['pair'],split=split,path=str(path),plan_sha256=sha(path)))
            seconds+=qualified_seconds(plan)
            if seconds>=target*3600:break
        if seconds<target*3600:raise ValueError(f'Only {seconds/3600:.3f} h {split}; need {target}')
        totals[split]=seconds/3600
    write_once(OUT/'selected.json',dict(pairs=chosen,hours=totals,normalization_reserve=True))
    print(dict(selected=len(chosen),hours=totals),flush=True)


def _normalize_pair(item):
    plan=read(item['path']);path=OUT/'normalized_pairs'/(item['pair']+'.json')
    if path.exists():return read(path)['summary']
    try:
        values=[]
        for source,speech in zip(plan['sources'],plan['speech_seconds']):
            value=subturns.normalize(plan['subset']+'_'+source['id'],source['source'],speech)
            import soundfile as sf
            info=sf.info(value['path'])
            if info.frames!=round(source['audio_samples']*48000/source['sample_rate']):
                raise ValueError('Loudness normalization changed duration')
            if speech>=2 and not value['target_reached']:raise ValueError('Loudness target not reached')
            if value['after']['input_tp'] is not None and value['after']['input_tp']>-.8:
                raise ValueError('True peak exceeds tolerance')
            values.append({k:value[k] for k in ('path','sha256','gain_db','before','after')})
        summary=dict(pair=plan['pair'],ok=True,split=plan['split'],seconds=qualified_seconds(plan),qualification_revision=2)
        write(path,dict(summary=summary,plan_sha256=sha(item['path']),tracks=values))
    except Exception as e:
        summary=dict(pair=plan['pair'],ok=False,seconds=0,error=type(e).__name__+': '+str(e))
        write(path,dict(summary=summary))
    return summary


def normalize_pair(item):
    # CephFS POSIX locks are shared by the three nodes and released on process death.
    path=OUT/'normalization_locks'/(item['pair']+'.lock')
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(pair=item['pair'],ok=True,busy=True,seconds=0)
        return _normalize_pair(item)


def extend_selection(a):
    """Replace rejected normalization hours with new planned pairs; retain every revision."""
    current=read(OUT/'selected.json');chosen=list(current['pairs']);seen={x['pair'] for x in chosen}
    qualified=Counter()
    for item in chosen:
        marker=OUT/'normalized_pairs'/(item['pair']+'.json')
        if not marker.exists():raise ValueError('Finish current normalization before extending')
        value=read(marker)['summary']
        if value['ok']:qualified[item['split']]+=qualified_seconds(read(item['path']))
    added=Counter();cat=read(OUT/'catalog.json')
    for split,target in (('train',315),('dev',3.3)):
        for p in cat[split]:
            if qualified[split]+added[split]>=target*3600:break
            path=OUT/'plans'/split/(p['pair']+'.json')
            if p['pair'] in seen or not path.exists():continue
            plan=read(path);summary=plan['summary'];seconds=qualified_seconds(plan)
            if not summary['ok'] or seconds<=0:continue
            chosen.append(dict(pair=p['pair'],split=split,path=str(path),plan_sha256=sha(path)))
            seen.add(p['pair']);added[split]+=seconds
        if qualified[split]+added[split]<target*3600:raise ValueError('Need more planned '+split+' pairs')
    if not added:raise ValueError('No replacement candidates')
    write_once(OUT/('selected_previous_'+digest(current)[:12]+'.json'),current)
    write(OUT/'selected.json',dict(pairs=chosen,normalization_reserve=True,
          previous_sha256=digest(current),qualified_hours={k:v/3600 for k,v in qualified.items()},
          replacement_hours={k:v/3600 for k,v in added.items()}))
    print(dict(replacement_hours={k:v/3600 for k,v in added.items()}),flush=True)


def run_normalize(a):
    subturns.OUT=OUT
    (OUT/'normalization_locks').mkdir(parents=True,exist_ok=True)
    all_rows=read(OUT/'selected.json')['pairs']
    # Prefer the node's shard, then help unfinished shards instead of idling at a straggler.
    rows=all_rows[a.node::3]+[r for i,r in enumerate(all_rows) if i%3!=a.node];errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(normalize_pair,rows),1):
            if not result['ok']:errors.append(result)
            if i%10==0 or i==len(rows):
                report=dict(node=a.node,completed=i,total=len(rows),failures=len(errors))
                write(OUT/f'normalize_progress_{a.node}.json',report);print(report,flush=True)
    write(OUT/f'normalize_errors_{a.node}.json',errors)


def freeze(a):
    cfg=read(a.config);run=Path(cfg['output']);run=run if run.is_absolute() else ROOT/run
    catalog_value=read(OUT/'catalog.json');selected=read(OUT/'selected.json')['pairs']
    rows=[];reviews=[];chosen=[];totals=Counter();clips=Counter()
    goals={'train':300*3600,'dev':a.dev_hours*3600}; used=defaultdict(list)
    for item in selected:
        split=item['split']
        if totals[split]>=goals[split]:continue
        path=OUT/'normalized_pairs'/(item['pair']+'.json')
        if not path.exists():raise ValueError('Normalization incomplete: '+item['pair'])
        normalized=read(path)
        if not normalized['summary']['ok']:continue
        if normalized['plan_sha256']!=sha(item['path']):raise ValueError('Stale normalized plan')
        plan=read(item['path']); included=[]
        for c in plan['clips']:
            if not eligible_clip(c) or totals[split]>=goals[split]:continue
            clip_id=plan['pair']+'_'+Path(c['path']).name
            ws=windows(c['start'],c['end']);totals[split]+=ws[-1]['end']-ws[0]['start'];clips[split]+=1
            included.append(c['path'])
            for i,w in enumerate(ws):
                row=dict(sample_id=f'{clip_id}_w{i:03d}',pair=plan['pair'],clip_id=clip_id,group_id=plan['group_id'],
                    split=split,**w,clip_start=c['start'],clip_end=c['end'],fps=25,height=512,width=512,
                    speaker_id=c['speaker'],listener_id=c['listener'],listener_crop=c['crop_boxes']['listener'],
                    sources={role:plan['sources'][c[role+'_index']]['source'] for role in ('speaker','listener')},
                    normalized_audio={role:{k:normalized['tracks'][c[role+'_index']][k] for k in ('path','sha256','gain_db')}
                                      for role in ('speaker','listener')})
                rows.append(row);reviews.append(dict(sample_id=row['sample_id'],candidate_kind=c['kind'],events=c['listener_events'],cut=c['cut']))
        if included:
            chosen.append(dict(item,clips=included));used[plan['group_id']].append(plan['pair'])
    if any(totals[s]<goals[s] for s in goals):raise ValueError(f'Insufficient qualified hours: {dict(totals)}')
    groups=[dict(g,pairs=sorted(used[g['group_id']])) for g in catalog_value['groups'] if g['group_id'] in used]
    random.Random(SEED).shuffle(rows)
    manifest=dict(version=cfg['version'],source=str(OUT),source_sha256=digest(chosen),
        official_source_split='exclusive official train/dev components; test untouched',
        seed=SEED,data_config=cfg['data'],prompts=contract(),groups=groups,windows=rows,short_clips=[],
        unique_seconds=dict(totals),canonical_clips=dict(clips),selection_policy=catalog_value['policy'],
        geometry_policy='exclude listener wide_crop (side > 3.2 * median face extent)')
    validate(manifest)
    write_once(run/'windows.json',manifest);write_once(run/'review_only.json',dict(note='Review only; never a model input',rows=reviews))
    write_once(run/'data_summary.json',dict(unique_hours={k:v/3600 for k,v in totals.items()},
        windows=dict(Counter(r['split'] for r in rows)),groups=dict(Counter(g['split'] for g in groups)),
        canonical_clips=dict(clips),manifest_sha256=sha(run/'windows.json'),official_test_used=False))
    write_once(run/'s0_selection.json',{s:[r['sample_id'] for r in rows if r['split']==s][:24] for s in ('train','dev')})
    write_once(run/'selected_pairs.json',chosen)
    print(read(run/'data_summary.json'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','plan','select','extend','normalize','freeze'))
    p.add_argument('--node',type=int,default=0,choices=(0,1,2));p.add_argument('--workers',type=int,default=16)
    p.add_argument('--train-pairs',type=int,default=14000);p.add_argument('--dev-pairs',type=int,default=160)
    p.add_argument('--hours',type=float,default=320);p.add_argument('--dev-hours',type=float,default=3)
    p.add_argument('--config',type=Path,default=ROOT/'configs/p3_listen_300h.json')
    a=p.parse_args()
    {'catalog':lambda _:catalog(),'plan':run_plan,'select':select,'extend':extend_selection,'normalize':run_normalize,'freeze':freeze}[a.command](a)


if __name__=='__main__':main()
