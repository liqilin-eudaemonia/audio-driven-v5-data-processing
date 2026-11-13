"""Audio-event model + acoustic screening of a bounded random review sample."""
import argparse,json,random,sys,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
import preview_listen_v6 as preview
from p3_listen.common import read,write,sha,digest
from p3_listen import media
from listen_audio_quality import EventModel,assess,POLICY

ROOT=preview.ROOT;OUT=ROOT/'data_previews/seamless_listen_qc_review'
EXP=ROOT/'experiments/p3_listen_qc_review';MODEL=ROOT/'models/audio_qc_ast'


@lru_cache(maxsize=128)
def metadata(archive,member):
    name=str(Path(member).with_suffix('.json'));offset,size=media.members(archive)[name]
    with open(archive,'rb') as f:f.seek(offset);return json.loads(f.read(size))


def annotations(meta,start,end):
    vad=[[max(start,v['start'])-start,min(end,v['end'])-start] for v in meta.get('metadata:vad',[])
         if v['end']>start and v['start']<end]
    words=[dict(start=w['start']-start,end=w['end']-start,score=w.get('score',0),text=w.get('word',''))
           for s in meta.get('metadata:transcript',[]) for w in s.get('words',[])
           if isinstance(w.get('start'),(int,float)) and isinstance(w.get('end'),(int,float))
           and w['end']>start and w['start']<end]
    return vad,words


def read_wave(path,start,duration):
    with sf.SoundFile(path) as f:
        if f.samplerate!=48000 or f.channels!=1:raise ValueError('Expected normalized mono 48 kHz')
        f.seek(round(start*48000));wave=f.read(round(duration*48000),dtype='float32')
    if len(wave)!=round(duration*48000):raise ValueError('Truncated source audio')
    return wave


def regression(model):
    previous=read(ROOT/'data_previews/seamless_listen_v6_audio_ab/samples.json')['samples']
    inventory=read(ROOT/'experiments/p3_listen_v6_audio_ab/source_inventory.json')
    rejected={r['sample_id'] for r in read(ROOT/'data_reviews/listener_noise_rejections.json')['rows']}
    results=[]
    for row in previous:
        if row['sample_id'] not in rejected:continue
        record=inventory[row['listener_id']]
        wave=read_wave(record['normalization']['path'],row['start'],row['duration_s'])
        vad,words=annotations(record['metadata'],row['start'],row['end'])
        result=assess(wave,48000,vad,words,model)
        results.append(dict(sample_id=row['sample_id'],expected='reject_user_confirmed_noise',
                            automatic_result=result,manual_exclusion_applied=False))
        print(json.dumps(dict(regression=row['sample_id'],accepted=result['accepted'],reasons=result['reasons'])),flush=True)
    write(EXP/'known_noise_regression.json',dict(rows=results,
        note='Known noisy examples informed design; this is a regression check, not held-out accuracy. No validated clean-laughter test set yet.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--regression-only',action='store_true');a=p.parse_args()
    model=EventModel(MODEL,batch_size=8)
    write(EXP/'model_contract.json',dict(provenance=read(MODEL/'provenance.json'),
        policy=POLICY,code_sha256=sha(ROOT/'scripts/listen_audio_quality.py'),
        human_labels=[model.labels[i] for i in model.human_ids],noise_labels=[model.labels[i] for i in model.noise_ids]))
    regression(model)
    if a.regression_only:return
    planned=read(OUT/'planned_candidates.json');rows=planned['rows'];results=[]
    for i,row in enumerate(rows):
        norm=read(preview.OLD/'normalized_pairs'/(row['pair']+'.json'));checks={}
        for role in ('listener','speaker'):
            source=row['sources'][role]
            wave=read_wave(norm['tracks'][row[role+'_index']]['path'],row['start'],row['duration_s'])
            vad,words=annotations(metadata(source['archive'],source['member']),row['start'],row['end'])
            checks[role]=assess(wave,48000,vad,words,model,role=role)
        result=dict(row,audio_qc=checks,screen_decision='keep' if all(c['accepted'] for c in checks.values()) else 'reject')
        results.append(result);write(OUT/'audio_checks'/(row['sample_id']+'.json'),result)
        print(json.dumps(dict(index=i+1,total=len(rows),sample=row['sample_id'],decision=result['screen_decision'],
            reasons={role:c['reasons'] for role,c in checks.items()})),flush=True)
    # Sample from both decision pools, with at most one video per conversation in each pool.
    rng=random.Random(planned['seed']+19);rng.shuffle(results);selected=[];used=set()
    for decision,target in [('keep',10),('reject',4)]:
        count=0
        # Ensure non-silent kept candidates are reviewable, without inventing reaction labels.
        pools=([r for r in results if r['audio_qc']['listener']['events']],
               [r for r in results if not r['audio_qc']['listener']['events']]) if decision=='keep' else (results,)
        for pi,pool in enumerate(pools):
            for row in pool:
                if row['screen_decision']!=decision or row['pair'] in used:continue
                if decision=='keep' and pi==0 and count>=3:break
                selected.append(dict(row,sample_id=row['sample_id'].replace('_v6_','_qc1_')))
                used.add(row['pair']);count+=1
                if count>=target:break
            if count>=target:break
    write(OUT/'screened_candidates.json',dict(seed=planned['seed'],policy=POLICY,rows=results,
        counts={k:sum(r['screen_decision']==k for r in results) for k in ('keep','reject')}))
    write(OUT/'publication_selection.json',dict(seed=planned['seed'],rows=selected,
        note='Seeded sample of decisions, plus up to three non-silent keep candidates. Rejected clips are explicit counterexamples, not training data.'))
    print(json.dumps(dict(selected=len(selected),keep=sum(r['screen_decision']=='keep' for r in selected),
                         reject=sum(r['screen_decision']=='reject' for r in selected))),flush=True)


if __name__=='__main__':main()
