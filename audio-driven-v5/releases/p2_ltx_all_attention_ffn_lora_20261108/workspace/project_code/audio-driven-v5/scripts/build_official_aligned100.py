"""Build the fixed 100-case timing review; reference existing gated WAVs unchanged."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.official_event_alignment import align_events,POLICY
from listen_benchmark.backchannel_face_stats import sha

SOURCE=ROOT/'experiments/listen_official_gate100_20261026/results.json'
RUN=ROOT/'experiments/listen_official_aligned100_20261026'
OUT=ROOT/'data_previews/listen_benchmark_review/audio_official_aligned100'

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(RUN/'labels').mkdir(exist_ok=True);(OUT/'labels').mkdir(exist_ok=True)
    data=json.loads(SOURCE.read_text());counts=Counter(); checks=[]
    for c in data['cases']:
        sid=c['sample_id'];original=ROOT/'data_previews/listen_benchmark_review/audio_official_random100/media'/sid/'label.json'
        assert sha(original)==c['label_sha256'];events=json.loads(original.read_text())['events']
        result=align_events(events,c['official']['listener'],c['gate'])
        result.update(sample_id=sid,source_label_sha256=c['label_sha256'],source_gate_run=SOURCE.parent.name,
                      original_audio_sha256=c['gate']['source_audio_sha256'],processed_audio_sha256=c['gate']['processed_audio_sha256'])
        write(RUN/'labels'/f'{sid}.json',result);write(OUT/'labels'/f'{sid}.json',result)
        audio=[e for e in result['events'] if e['modality']=='audio']
        status=Counter(a['method'] for a in result['audio_event_alignment']);counts.update(status)
        old={e['event_id']:e for e in events}
        assert [e for e in result['events'] if e['modality']!='audio']==[e for e in events if e['modality']!='audio']
        assert len(result['events'])+len(result['excluded_audio_events'])==len(events)
        for e in audio:
            for key in ['description','text','original_descriptions','categories','description_source','description_sources']:
                assert e.get(key)==old[e['event_id']].get(key)
            assert e['source_interval']==dict(start=old[e['event_id']]['start'],end=old[e['event_id']]['end'])
            assert e['start']==e['intervals'][0]['start'] and e['end']==e['intervals'][-1]['end']
            for x in e['intervals']:
                assert 0<=x['start']<x['end']<=c['duration_s']+1e-6
                assert any(a/c['gate']['sample_rate']-1e-9<=x['start'] and x['end']<=b/c['gate']['sample_rate']+1e-9 for a,b in c['gate']['positive_intervals_samples'])
        for k,v in c['media'].items():
            if not v.startswith('../'):c['media'][k]='../audio_official_gate100/'+v
        c['aligned_audio_events']=audio;c['alignment_audit']=result['audio_event_alignment'];c['alignment_counts']=dict(status)
        c['label_group']='has_audio_labels' if audio else 'no_audio_labels';c['aligned_label_url']=f'labels/{sid}.json'
        c['gate']['label_url']='../audio_official_gate100/'+c['gate']['label_url'];c['gate']['metadata_url']='../audio_official_gate100/'+c['gate']['metadata_url']
        checks.append(dict(sample_id=sid,source_label_hash_ok=True,text_and_categories_unchanged=True,visual_events_unchanged=True,all_events_accounted_for=True,all_final_intervals_within_audio_support=True))
    summary=dict(samples=len(data['cases']),with_audio_labels=sum(c['label_group']=='has_audio_labels' for c in data['cases']),
                 without_audio_labels=sum(c['label_group']=='no_audio_labels' for c in data['cases']),
                 without_labels_but_audio_retained=sum(c['label_group']=='no_audio_labels' and not c['gate']['full_silence'] for c in data['cases']),
                 original_audio_events=sum(len(c['alignment_audit']) for c in data['cases']),retained_audio_events=sum(len(c['aligned_audio_events']) for c in data['cases']),
                 methods=dict(counts),boundary_changed_retained=sum(a['boundary_changed'] and a['method']!='excluded' for c in data['cases'] for a in c['alignment_audit']),
                 original_text_with_unmatched_official_words=sum(a['text_not_literal_match'] and a['method']!='excluded' for c in data['cases'] for a in c['alignment_audit']),
                 training_applied=False,audio_reprocessed=False)
    data.update(schema='official-aligned-review100-v1',run_id=RUN.name,alignment_policy=POLICY,alignment_summary=summary,source_gate_results_sha256=sha(SOURCE))
    write(RUN/'results.json',data);write(RUN/'summary.json',summary);write(RUN/'verification.json',checks);write(RUN/'policy.json',POLICY);write(OUT/'review.json',data)
    with (RUN/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_rank','sample_id','label_group','audio_full_silence','original_audio_events','retained_audio_events','official_words','official_vad','model_intersection','excluded'])
        writer.writeheader()
        for c in data['cases']:writer.writerow(dict(sample_rank=c['sample_rank'],sample_id=c['sample_id'],label_group=c['label_group'],audio_full_silence=c['gate']['full_silence'],original_audio_events=len(c['alignment_audit']),retained_audio_events=len(c['aligned_audio_events']),**{k:c['alignment_counts'].get(k,0) for k in ['official_words','official_vad','model_intersection','excluded']}))
    (OUT/'summary.csv').write_bytes((RUN/'summary.csv').read_bytes())
    html=(ROOT/'scripts/listener_official_aligned100.html').read_text();(OUT/'index.html').write_text(html)
    from fontTools.ttLib import TTFont
    from fontTools import subset
    font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=html+json.dumps(data,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
