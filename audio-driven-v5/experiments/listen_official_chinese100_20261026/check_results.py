import json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import sha
from listen_benchmark.chinese_asr_filter import filter_labels,han_count
RUN=Path(__file__).resolve().parent;OUT=ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100';PREV=ROOT/'data_previews/listen_benchmark_review/audio_official_aligned100'
data=json.loads((RUN/'results.json').read_text());old=json.loads((ROOT/'experiments/listen_official_aligned100_20261026/results.json').read_text());single=0
assert [c['sample_id'] for c in data['cases']]==[c['sample_id'] for c in old['cases']]
for c,before in zip(data['cases'],old['cases']):
    sid=c['sample_id'];path=ROOT/'experiments/listen_official_aligned100_20261026/labels'/f'{sid}.json';label=json.loads(path.read_text());expected,mask,support=filter_labels(label,before['gate']);actual=json.loads((RUN/'labels'/f'{sid}.json').read_text())
    assert actual['events']==expected['events'];assert actual['excluded_audio_events']==expected['excluded_audio_events']
    assert sha(path)==actual['source_aligned_label_sha256']
    prev=(PREV/before['media']['processed']).resolve();current=(OUT/c['media']['processed']).resolve()
    assert sha(prev)==before['gate']['processed_audio_sha256'];assert sha(current)==c['gate']['processed_audio_sha256']
    x,sr=sf.read(prev,dtype='float32');y,ysr=sf.read(current,dtype='float32')
    assert sr==ysr and x.shape==y.shape;np.testing.assert_array_equal(y,x*mask)
    for a,b in actual['language_filter']['mute_intervals_samples']:assert np.all(y[a:b]==0)
    np.testing.assert_array_equal(y[mask==1],x[mask==1])
    removed,_=sf.read((OUT/c['media']['removed']).resolve(),dtype='float32');raw,_=sf.read((OUT/c['media']['listener']).resolve(),dtype='float32');assert np.max(abs(y+removed-raw))<1e-7
    assert c['gate']['full_silence']==(not support.any())
    for e in label['events']:
        if e.get('text') and han_count(e['text'])==1:assert e in actual['events'];single+=1
assert single==9;assert data['language_filter_summary']['affected_samples']==3
report=dict(samples=100,changed_samples=3,single_han_events_exactly_preserved=single,all_source_hashes_unchanged=True,all_output_labels_reproduced=True,
            mute_samples_exactly_zero=True,unaffected_waveform_bit_exact=True,duration_and_rate_unchanged=True,residual_reconstruction=True)
(RUN/'full_numeric_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
