"""Apply the approved 50-sample integration pilot, preserving raw and human evidence."""
import argparse,hashlib,json,shutil,sys,io
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.integration_pilot import integrate,VERSION
from listen_benchmark.response_review import alignment_track

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def corrections(source,review,observations):
    """Explicit case adjudications from this user's notes, not an automatic NLP rule."""
    feedback=json.loads(source.read_text());result={}
    for rec in feedback['records']:
        sid=rec['sample_id'];s=next(s for s in review['samples'] if s['id']==sid)
        group=next(e for e in s['events'] if e['event_id']==rec['group_id'])
        base=dict(feedback_reference=str(source)+'#'+sid,original_feedback=rec,
                  boundary_precision_s=None,interpretation_author='assistant_from_user_notes',events=[])
        def event(description,text=None,start=None,end=None,unit='vocal_response',count='human_confirmed',boundary='unknown',**extra):
            return dict(description=description,text=text,start=start,end=end,event_unit=unit,
                        status=dict(content='human_confirmed',count=count,boundary=boundary),**extra)
        # Cases are keyed by reviewed sample identity and remain separate from the algorithm.
        if sid=='naturalistic_V03_S2043_I00000338_local_32d3f86103cd':
            base['replace_observation_ids']=group['member_event_ids']
            base['events']=[event('发出“嘿嘿”式笑声',unit='laughter_bout_candidate',count='unknown',
                phonetic_hint='嘿嘿',candidate_intervals=[dict(start=.4,end=1.04,role='old_group_context_not_confirmed_boundary')],
                review_flags=['lexical_hei_reinterpreted_as_laughter','laughter_bout_count_unknown'])]
            base['summary']='已确认是笑声；“嘿嘿”作为拟声描述保留，三个词区间不等于三次笑声。'
        else:
            # Keep all previous audio evidence visible, but do not assign it to corrected occurrences.
            base['replace_observation_ids']=[e['event_id'] for e in observations[sid]['events'] if e['modality']=='audio']
            base['additional_model_audio_policy']='preserved_as_unresolved_evidence_not_rejected'
            if sid=='naturalistic_V00_S0805_I00000126_local_f7c1fb6d13ca':
                seq=['Right','right','Yeah','yeah','yeah','yeah','yeah']
                base['events']=[event('说“'+t+'”',text=t,unit='lexical_response',sequence_index=i,
                    candidate_intervals=[],review_flags=['time_not_assigned_from_three_submitted_intervals']) for i,t in enumerate(seq)]
                base['confirmed_sequence']=seq;base['sequence_context']={'start':2.,'end':5.,'role':'user_referenced_window_not_activity_mask'}
                base['summary']='已确认 Right×2、Yeah×5；三个已填区间与七个词尚未逐一对应，全部逐次边界暂留空。'
            elif sid=='improvised_V00_S0835_I00000896_local_ab89a3850544':
                base['events']=[event('说“Yeah”，随后有尾音',text='Yeah',candidate_intervals=[dict(start=3.88,end=3.96,role='core_only_tail_end_unknown')],review_flags=['discard_Ed_interpretation','tail_end_unknown'])]
                base['summary']='内容改为 Yeah 及尾音；3.88–3.96秒仅作核心参考，不沿用 Ed 的3.84–10秒长区间。'
            elif sid=='naturalistic_V02_S3358_I00000420_local_983b51838483':
                base['events']=[event('说“嗯”',text='嗯',start=e['start'],end=e['end'],boundary='human_submitted_precision_unverified',sequence_index=i) for i,e in enumerate(rec['occurrences'])]
                base['summary']='确认两次嗯；保留两个提交区间，不因首尾相接合并，±0.2秒精度未单独核验。'
            elif sid=='naturalistic_V02_S5193_I00000190_local_60bde84ffa78':
                base['events']=[event('说“'+e['description']+'”',text=e['description'],start=e['start'],end=e['end'],boundary='human_approximate',sequence_index=i,review_flags=['manual_vs_asr_boundary_conflict'] if i==1 else []) for i,e in enumerate(rec['occurrences'])]
                base['summary']='确认 Uh huh、Yeah 两次回应；分别保留3–4秒、7–8秒人工大致时间，Yeah与旧ASR边界有差异。'
            else:raise ValueError('Uninterpreted human feedback: '+sid)
        result[sid]=base
    return result

def build(source,align,feedback,out,preview,font=None):
    for p in [out,preview]:
        if p.exists() and any(p.iterdir()):raise ValueError('Use empty output directories: '+str(p))
    raw=(source/'observations.jsonl').read_bytes();original={s['sample_id']:s for s in map(json.loads,raw.splitlines())};old=json.loads((source/'review.json').read_text())
    corr=corrections(feedback,old,original);samples=[];hashes={}
    for s in old['samples']:
        tracks=[]
        for p in sorted(align.glob(s['id']+'*.json')):
            b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();tracks.append(alignment_track(json.loads(b),s['duration']))
        row=integrate(original[s['id']],s['decisions'],tracks,corr.get(s['id']))
        row.update(id=s['id'],video=s['video'],audio=s['audio'],original_events=original[s['id']]['events'])
        samples.append(row)
    all_events=[e for s in samples for e in s['events']]
    tokens=sum(len(t['tokens']) for s in samples for t in s['alignment_tracks']);clusters=sum(len(s['token_clusters']) for s in samples)
    summary=dict(samples=len(samples),original_observations=sum(len(s['events']) for s in original.values()),
        mapped_observations_before=sum(e['mapping_status']=='mapped_candidate' for s in original.values() for e in s['events']),
        independent_event_candidates=len(all_events),visual_candidates=sum(e['modality']=='visual' for e in all_events),
        audio_candidates=sum(e['modality']=='audio' for e in all_events),
        cross_window_observation_duplicates_removed=sum(e['duplicate_observations_removed'] for s in samples for e in s['pre_correction_events']),
        asr_token_observations=tokens,asr_token_clusters=clusters,asr_cross_window_duplicate_observations=tokens-clusters,
        invalid_asr_token_times=sum(not t['time_valid'] for s in samples for tr in s['alignment_tracks'] for t in tr['tokens']),
        human_corrected_samples=len(corr),events_with_unknown_time=sum(e.get('start') is None for e in all_events),
        unresolved_observations=sum(len(s['unresolved_observations']) for s in samples),
        descriptive_observations_attached_without_counting=sum(x['relation']=='descriptive_support_not_an_extra_occurrence' for s in samples for x in s['evidence_links']),
        training_admitted=False,full_corpus_applied=False,media_inference_rerun=False)
    data=dict(schema=VERSION,run_id=out.name,summary=summary,samples=samples,policy=dict(unlabeled_time='unknown',group_extent_is_activity=False,training_admitted=False,matching_tolerance_s=.2,precision_guarantee=False))
    html=(ROOT/'scripts/listen_integration_pilot.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    fontbytes=None
    if font:
        from fontTools.ttLib import TTFont
        from fontTools.subset import Options,Subsetter
        f=TTFont(str(font),fontNumber=2);o=Options();o.flavor='woff';sub=Subsetter(options=o);sub.populate(text=html);sub.subset(f);f.flavor='woff';buf=io.BytesIO();f.save(buf);fontbytes=buf.getvalue()
    else:html=html.replace('@font-face{font-family:Review;src:url(review.woff)}','')
    for p in [out,preview]:
        p.mkdir(parents=True,exist_ok=True);dump(p/'review.json',data);dump(p/'summary.json',summary);dump(p/'human_corrections.json',corr)
        (p/'observations.jsonl').write_bytes(raw);shutil.copy2(feedback,p/'human_feedback.original.json');(p/'index.html').write_text(html)
        (p/'labels.candidate.jsonl').write_text(''.join(json.dumps({k:v for k,v in s.items() if k not in ['alignment_tracks','original_events','pre_correction_events','video','audio']},ensure_ascii=False)+'\n' for s in samples))
        if fontbytes:(p/'review.woff').write_bytes(fontbytes)
    paths=['src/listen_benchmark/integration_pilot.py','src/listen_benchmark/response_review.py','scripts/build_listen_integration_pilot.py','scripts/listen_integration_pilot.html','tests/test_listen_integration_pilot.py']
    for name in paths:
        dest=out/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    dump(out/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source=str(source),alignment=str(align),feedback=str(feedback),
        raw_observations_sha256=hashlib.sha256(raw).hexdigest(),feedback_sha256=hashlib.sha256(feedback.read_bytes()).hexdigest(),alignment_sha256=hashes,
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','align','feedback','out','preview']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--font',type=Path);a=p.parse_args();build(a.source.resolve(),a.align.resolve(),a.feedback.resolve(),a.out.resolve(),a.preview.resolve(),a.font)
