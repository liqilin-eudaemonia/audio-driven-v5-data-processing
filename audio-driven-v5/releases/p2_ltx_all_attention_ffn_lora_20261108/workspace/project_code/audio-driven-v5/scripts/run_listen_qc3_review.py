"""QC3 on the existing 56 clips, with paired-track/background checks and honest regression."""
import argparse,json,random,time,shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
import preview_listen_v6 as preview
from p3_listen.common import read,write,sha
from screen_listen_qc_sample import read_wave
from listen_audio_quality_v3 import PannsModel,scan_track,prepare_pair,POLICY,MODEL
ROOT=preview.ROOT;OLD=ROOT/'data_previews/seamless_listen_qc2_review'
OUT=ROOT/'data_previews/seamless_listen_qc3_review';EXP=ROOT/'experiments/p3_listen_qc3_review'
SEED=2026102211


def write_contract(model):
    write(EXP/'frozen_contract.json',dict(policy=POLICY,seed=SEED,
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model=read(MODEL/'provenance.json'),
        vendor=read(MODEL/'vendor_provenance.json'),
        noise_labels=[model.labels[i] for i in model.noise_ids],human_labels=[model.labels[i] for i in model.vocal_ids],
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['listen_audio_quality_v3.py','run_listen_qc3_review.py']}))


def wave_pair(row):
    norm=read(preview.OLD/'normalized_pairs'/(row['pair']+'.json'))
    return {role:read_wave(norm['tracks'][row[role+'_index']]['path'],row['start'],row['duration_s']) for role in ['speaker','listener']}


def scan_row(row,model):
    waves=wave_pair(row);pair=prepare_pair(waves['speaker'],waves['listener']);checks={}
    for role in ['listener','speaker']:
        checks[role]=scan_track(waves[role],row['audio_qc'][role],model,paired=pair if role=='listener' else None,role=role)
    statuses=[v['status'] for v in checks.values()]
    decision='exclude' if 'exclude' in statuses else 'review' if 'review' in statuses else 'keep'
    if row['listener_screen'].get('deferred_reasons') and decision=='keep':decision='review'
    return dict(row,qc3=checks,qc2_decision=row['screen_decision'],screen_decision=decision,
                listener_human_evidence=checks['listener']['has_human_evidence'])


def regression(model):
    result=[]
    old=read(ROOT/'data_previews/seamless_listen_qc_review/samples.json')['samples']
    for row in old:
        if not row.get('human_review'):continue
        scanned=scan_row(row,model)
        result.append(dict(sample_id=row['sample_id'],human_label=row['human_review']['decision'],
                           automatic_status=scanned['screen_decision'],qc3=scanned['qc3'],source='qc1'))
    ab=read(ROOT/'data_previews/seamless_listen_v6_audio_ab/samples.json')['samples']
    lookup={r['sample_id']:r for r in ab}
    inventory=read(ROOT/'experiments/p3_listen_v6_audio_ab/source_inventory.json')
    for r in read(ROOT/'experiments/p3_listen_qc_review/known_noise_regression.json')['rows']:
        row=lookup[r['sample_id']];wave=read_wave(inventory[row['listener_id']]['normalization']['path'],row['start'],row['duration_s'])
        paired=None
        if row.get('video') and 'speaker_index' in row:
            waves=wave_pair(row);paired=prepare_pair(waves['speaker'],waves['listener'])
        c=scan_track(wave,r['automatic_result'],model,paired=paired)
        result.append(dict(sample_id=row['sample_id'],human_label='reject_noise',automatic_status=c['status'],qc3=dict(listener=c),source='audio_ab',paired_check_available=paired is not None))
    write(EXP/'known_examples_regression.json',dict(rows=result,
        note='Development examples, not held-out accuracy. Human labels never override automatic results; no waveform alteration. Four old audio-only probes lack paired-track analysis.'))
    print('REGRESSION',[(x['human_label'],x['automatic_status']) for x in result],flush=True)
    return result


def screen():
    OUT.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    model=PannsModel(batch_size=16);write_contract(model)
    known=regression(model)
    rows=read(OLD/'screened_candidates.json')['rows'];assert len(rows)==56
    manual=read(ROOT/'data_reviews/listener_noise_rejections.json')['rows'];results=[]
    for i,row in enumerate(rows):
        result=scan_row(row,model)
        hits=[r for r in manual if r['listener_id']==row['listener_id'] and r['end']>row['start']+1e-7 and r['start']<row['end']-1e-7]
        result['manual_exclusions']=hits
        result['effective_decision']='exclude' if hits else result['screen_decision']
        results.append(result);write(OUT/'audio_checks'/(row['sample_id']+'.json'),result)
        print(json.dumps(dict(index=i+1,total=len(rows),sample=row['sample_id'],previous=row['screen_decision'],automatic=result['screen_decision'],manual=bool(hits),reasons={k:v['reasons'] for k,v in result['qc3'].items()})),flush=True)
    write(OUT/'screened_candidates.json',dict(policy=POLICY,rows=results,
        automatic_counts=dict(Counter(r['screen_decision'] for r in results)),
        effective_counts=dict(Counter(r['effective_decision'] for r in results)),
        transitions=dict(Counter(r['qc2_decision']+' -> '+r['screen_decision'] for r in results))))
    known_bleed=next(r for r in results if r['sample_id']=='naturalistic_V01_S0065_I00000126_qc2_p001_00')
    known.append(dict(sample_id=known_bleed['sample_id'],human_label='reject_speaker_bleed',automatic_status=known_bleed['screen_decision'],qc3=known_bleed['qc3'],source='qc2'))
    write(EXP/'known_examples_regression.json',dict(rows=known,note='Development feedback examples; automatic decisions before manual exclusions. Same-pool QC2 comparison, not a fresh independent test.'))
    # Show all 16 previously published QC2 clips so that changes are directly reviewable.
    previous=read(OLD/'samples.json')['samples'];lookup={r['sample_id']:r for r in results}
    selected=[dict(lookup[r['sample_id']],display_group='previous_qc2',reuse_preview=r['preview'],reuse_root=str(OLD)) for r in previous]
    used={r['pair'] for r in selected};shuffled=list(results);random.Random(SEED).shuffle(shuffled)
    # Add up to four voice candidates that survive new checks, without relaxing thresholds.
    for r in shuffled:
        if len(selected)>=20:break
        if r['pair'] in used or r['effective_decision']!='keep' or not r['listener_human_evidence']:continue
        selected.append(dict(r,display_group='additional_voice_candidate'));used.add(r['pair'])
    # User-confirmed positive controls stay visibly separate, regardless of automatic decision.
    qc1root=ROOT/'data_previews/seamless_listen_qc_review'
    controls={r['sample_id']:r for r in known if r['human_label']=='retain_valid_listener_response'}
    for r in read(qc1root/'samples.json')['samples']:
        if r['sample_id'] not in controls:continue
        c=controls[r['sample_id']]
        selected.append(dict(r,qc3=c['qc3'],qc2_decision='not_in_qc2',screen_decision=c['automatic_status'],
            effective_decision=c['automatic_status'],display_group='human_positive_control',
            listener_human_evidence=c['qc3']['listener']['has_human_evidence'],
            reuse_preview=r['preview'],reuse_root=str(qc1root),manual_exclusions=[]))
    write(OUT/'publication_selection.json',dict(rows=selected,seed=SEED))
    # Build a balanced annotation queue, leaving human labels empty. Clip labels do not label every event.
    queue=[]
    for r in results:
        for kind,key in [('event','events'),('background','background')]:
            for i,e in enumerate(r['qc3']['listener'][key]):
                queue.append(dict(sample_id=r['sample_id'],pair=r['pair'],group_id=r['group_id'],kind=kind,
                    start=e['start'],end=e['end'],automatic_status=e['decision']['status'],
                    annotation=None,allowed_labels=['clean_backchannel','clean_laughter_or_nonverbal','environment_noise','speaker_bleed','reaction_mixed_with_noise','quiet','uncertain']))
    rng=random.Random(SEED);rng.shuffle(queue);pools={k:[e for e in queue if e['automatic_status']==k] for k in ('exclude','review','keep')}
    balanced=[]
    while len(balanced)<240 and any(pools.values()):
        for p in pools.values():
            if p and len(balanced)<240:balanced.append(p.pop())
    write(OUT/'annotation_queue.json',dict(rows=balanced,status='awaiting_human_event_labels',note='Model decisions are not labels. Split future training/evaluation by source group; do not train on current clip-level feedback as event labels.'))
    print('DONE',dict(Counter(r['screen_decision'] for r in results)),'publish',len(selected),flush=True)


def export_one(row):
    row=dict(row)
    if row.get('reuse_root'):
        destination=OUT/'media'/row['sample_id'];destination.parent.mkdir(parents=True,exist_ok=True)
        source=Path(row['reuse_root'])/row['reuse_preview']
        if not destination.exists():destination.symlink_to(source,target_is_directory=True)
        row.update(preview='media/'+row['sample_id']+'/',media_verified=True)
        return row
    return preview.publish(row)


def publish():
    preview.OUT=OUT;rows=read(OUT/'publication_selection.json')['rows'];assert len(rows)<=23
    with ThreadPoolExecutor(max_workers=4) as pool:samples=list(pool.map(export_one,rows))
    summary=read(OUT/'screened_candidates.json')
    write(OUT/'samples.json',dict(samples=samples,policy=POLICY,status='human_review_pending_no_training',
        source_candidates=56,automatic_counts=summary['automatic_counts'],effective_counts=summary['effective_counts'],
        transitions=summary['transitions'],regression=read(EXP/'known_examples_regression.json'),
        note='Same QC2 clips, unchanged normalized audio. Three positive controls explicitly identified. No new independent accuracy estimate.'))
    shutil.copyfile(ROOT/'scripts/listen_qc3_review.html',OUT/'index.html')
    annotation=read(OUT/'annotation_queue.json')
    corpus={r['sample_id']:r for r in read(OUT/'screened_candidates.json')['rows']}
    cache={};folder=OUT/'annotation_audio';folder.mkdir(exist_ok=True)
    for i,e in enumerate(annotation['rows']):
        sid=e['sample_id'];row=corpus[sid]
        if sid not in cache:cache[sid]=wave_pair(row)['listener']
        a=max(0,e['start']-.25);b=min(row['duration_s'],e['end']+.25)
        name=f'{i:03d}.wav';sf.write(folder/name,cache[sid][round(a*48000):round(b*48000)],48000,subtype='PCM_24')
        e.update(audio='annotation_audio/'+name,context_start=a,context_end=b,
                 source_start=row['start']+e['start'],source_end=row['start']+e['end'])
    write(OUT/'annotation_queue.json',annotation)
    shutil.copyfile(ROOT/'scripts/listen_qc3_annotation.html',OUT/'annotation.html')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_qc3'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print('Published http://127.0.0.1:8875/listen_qc3/',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','publish']);a=p.parse_args()
    {'screen':screen,'publish':publish}[a.stage]()
