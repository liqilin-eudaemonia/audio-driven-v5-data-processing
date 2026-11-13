"""Apply the approved multi-character Chinese ASR policy to the fixed 100 cases."""
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.chinese_asr_filter import filter_labels,POLICY,han_count
from listen_benchmark.backchannel_face_stats import sha

SOURCE=ROOT/'experiments/listen_official_aligned100_20261026'
RUN=ROOT/'experiments/listen_official_chinese100_20261026'
OUT=ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100'
PREV=ROOT/'data_previews/listen_benchmark_review/audio_official_aligned100'

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def main():
    for p in [RUN,OUT,RUN/'labels',OUT/'labels']:p.mkdir(parents=True,exist_ok=True)
    data=json.loads((SOURCE/'results.json').read_text());checks=[];changes=[];new_seconds=0
    for c in data['cases']:
        sid=c['sample_id'];lp=SOURCE/'labels'/f'{sid}.json';original=json.loads(lp.read_text());old_gate=copy.deepcopy(c['gate'])
        result,mask,support=filter_labels(original,old_gate);policy=result['language_filter'];affected=bool(policy['removed_event_ids'])
        prev_wave=(PREV/c['media']['processed']).resolve();oldhash=sha(prev_wave);assert oldhash==old_gate['processed_audio_sha256']
        c['media']['previous_processed']=c['media']['processed']
        sourcehash=sha(lp)
        if affected:
            dst=OUT/'media'/sid;dst.mkdir(parents=True,exist_ok=True)
            x,sr=sf.read(prev_wave,dtype='float32');assert sr==old_gate['sample_rate'] and len(x)==old_gate['samples']
            y=x*mask;sf.write(dst/'processed.wav',y,sr,subtype='FLOAT')
            raw,_=sf.read((PREV/c['media']['listener']).resolve(),dtype='float32');sf.write(dst/'removed.wav',raw-y,sr,subtype='FLOAT')
            for a,b in policy['mute_intervals_samples']:assert np.all(y[a:b]==0)
            np.testing.assert_array_equal(y[mask==1],x[mask==1])
            decoded,rate=sf.read(dst/'processed.wav',dtype='float32');np.testing.assert_array_equal(decoded,y);assert rate==sr
            c['media']['processed']=f'media/{sid}/processed.wav';c['media']['removed']=f'media/{sid}/removed.wav'
            newhash=sha(dst/'processed.wav')
            cut_s=sum(b-a for a,b in policy['mute_intervals_samples'])/sr;new_seconds+=cut_s
            changes.append(dict(sample_rank=c['sample_rank'],sample_id=sid,texts=policy['removed_texts'],mute_intervals=[dict(start=a/sr,end=b/sr) for a,b in policy['mute_intervals_samples']],mute_seconds=cut_s))
            ranges=policy['positive_intervals'];ends=[];left=0.
            for x in ranges:
                if x['start']>left:ends.append(dict(start=left,end=x['start'],text='置零区间'))
                left=x['end']
            if left<c['duration_s']:ends.append(dict(start=left,end=c['duration_s'],text='置零区间'))
            c['gate'].update(full_silence=not bool(ranges),positive_intervals_samples=[[round(x['start']*sr),round(x['end']*sr)] for x in ranges],
                             intervals=[dict(x,text='当前保留区间') for x in ranges],muted_intervals=ends,
                             positive_samples=int(np.count_nonzero(support)),zero_samples=int(np.count_nonzero(support==0)),
                             nominal_keep_samples=int(np.count_nonzero(support)),processed_audio_sha256=newhash,
                             post_language_filter=POLICY)
            # Old event/support statistics describe the previous gate, not the new one.
            for key in ['events','event_status_counts','keep_intervals_samples','core_intervals_samples','core_samples','fade_samples']:c['gate'].pop(key,None)
        else:newhash=oldhash
        result.update(schema='official-aligned-chinese-filtered-v1',source_aligned_label_sha256=sourcehash,
                      previous_processed_audio_sha256=oldhash,processed_audio_sha256=newhash,training_applied=False)
        write(RUN/'labels'/f'{sid}.json',result);write(OUT/'labels'/f'{sid}.json',result)
        assert sha(lp)==sourcehash and sha(prev_wave)==oldhash
        old_visual=[e for e in original['events'] if e['modality']!='audio'];new_visual=[e for e in result['events'] if e['modality']!='audio'];assert old_visual==new_visual
        assert len(result['events'])+len(result['excluded_audio_events'])==len(original['events'])+len(original['excluded_audio_events'])
        single_ids={e['event_id'] for e in original['events'] if e.get('text') and han_count(e['text'])==1}
        surviving={e['event_id']:e for e in result['events']};assert single_ids<=set(surviving)
        for e in original['events']:
            if e['event_id'] in surviving:
                for k in ['description','text','categories','original_descriptions']:assert e.get(k)==surviving[e['event_id']].get(k)
        c['aligned_audio_events']=[e for e in result['events'] if e['modality']=='audio'];c['alignment_audit']=result['audio_event_alignment']
        c['alignment_counts']=dict(Counter(a['method'] for a in c['alignment_audit']));c['label_group']='has_audio_labels' if c['aligned_audio_events'] else 'no_audio_labels'
        c['language_filter']=policy;c['aligned_label_url']=f'labels/{sid}.json'
        checks.append(dict(sample_id=sid,changed=affected,source_files_unchanged=True,visual_events_unchanged=True,all_events_accounted_for=True,single_han_events_retained=len(single_ids),processed_audio_sha256=newhash))
    summary=dict(samples=len(data['cases']),affected_samples=len(changes),removed_events=sum(len(x['texts']) for x in changes),
                 exact_mute_seconds=new_seconds,single_han_events_retained=sum(x['single_han_events_retained'] for x in checks),
                 with_audio_labels=sum(c['label_group']=='has_audio_labels' for c in data['cases']),without_audio_labels=sum(c['label_group']=='no_audio_labels' for c in data['cases']),
                 retained_audio_events=sum(len(c['aligned_audio_events']) for c in data['cases']),full_silence_clips=sum(c['gate']['full_silence'] for c in data['cases']),training_applied=False,changes=changes)
    data.update(schema='official-chinese-filtered-review100-v1',run_id=RUN.name,language_filter_summary=summary,language_filter_policy=POLICY,
                source_aligned_results_sha256=sha(SOURCE/'results.json'))
    data['alignment_summary'].update(with_audio_labels=summary['with_audio_labels'],without_audio_labels=summary['without_audio_labels'],retained_audio_events=summary['retained_audio_events'],
                                     methods=dict(sum((Counter(c['alignment_counts']) for c in data['cases']),Counter())),audio_reprocessed=True,
                                     boundary_changed_retained=sum(a['boundary_changed'] and a['method']!='excluded' for c in data['cases'] for a in c['alignment_audit']),
                                     original_text_with_unmatched_official_words=sum(a['text_not_literal_match'] and a['method']!='excluded' for c in data['cases'] for a in c['alignment_audit']),
                                     without_labels_but_audio_retained=sum(c['label_group']=='no_audio_labels' and not c['gate']['full_silence'] for c in data['cases']))
    for name,value in [('results.json',data),('summary.json',summary),('policy.json',POLICY),('verification.json',checks)]:write(RUN/name,value)
    write(OUT/'review.json',data)
    with (RUN/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_rank','sample_id','group','remaining_audio_events','removed_chinese','muted_intervals']);writer.writeheader()
        for c in data['cases']:writer.writerow(dict(sample_rank=c['sample_rank'],sample_id=c['sample_id'],group=c['label_group'],remaining_audio_events=len(c['aligned_audio_events']),removed_chinese=' | '.join(c['language_filter']['removed_texts']),muted_intervals=json.dumps(c['language_filter']['mute_intervals_samples'])))
    (OUT/'summary.csv').write_bytes((RUN/'summary.csv').read_bytes())
    html=(ROOT/'scripts/listener_official_chinese100.html').read_text();(OUT/'index.html').write_text(html)
    from fontTools.ttLib import TTFont
    from fontTools import subset
    font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2);sub=subset.Subsetter();sub.populate(text=html+json.dumps(data,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
