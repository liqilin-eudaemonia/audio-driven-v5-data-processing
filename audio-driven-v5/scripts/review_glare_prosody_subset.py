#!/usr/bin/env python3
"""Bounded, status-stratified source-chunk review with exact model-input FLOAT WAVs."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import ChunkCache,CHUNK_SAMPLES,read,save,sha,lock
from glare_prosody_source_runtime import OfficialRuntime,waveform_sha
from prepare_glare_prosody_subset import protocol,connect,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=32)
    a=p.parse_args();require(1<=a.limit<=64,'review limit must be1..64')
    require(not a.out.exists(),'Use a new review directory to preserve evidence')
    protocol_data=protocol(a.plan);cache=ChunkCache(protocol_data['cache_root'],protocol_data['extractor'])
    buckets=defaultdict(list)
    with lock(cache.root/'mutation.lock',False),connect(a.plan) as db:
        tracks={r['track_id']:json.loads(r['data']) for r in db.execute('SELECT * FROM tracks')}
        for r in db.execute('SELECT * FROM chunks ORDER BY track_id,chunk_index'):
            track=tracks[r['track_id']];record=cache.load(track,r['chunk_index'])
            if record:buckets[record['analysis']['status']].append((track,r['chunk_index'],record))
    selected=[]
    for offset in range(a.limit):
        for status in sorted(buckets):
            if offset<len(buckets[status]) and len(selected)<a.limit:selected.append(buckets[status][offset])
    a.out.mkdir(parents=True);runtime=OfficialRuntime(protocol_data['extractor']);wave=None;last=None
    rows=[];cards=[]
    import soundfile as sf
    for track,index,record in selected:
        if track['track_id']!=last:wave=runtime.waveform(track);last=track['track_id']
        chunk=wave[index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES]
        name=cache.chunk_id(track,index)+'.wav';audio=''
        if len(chunk)==CHUNK_SAMPLES:
            require(waveform_sha(chunk)==record['waveform_sha256'] or record['waveform_sha256'] is None,'Review waveform changed')
            sf.write(a.out/name,chunk,16000,subtype='FLOAT')
            restored,sr=sf.read(a.out/name,dtype='float32');require(sr==16000 and waveform_sha(restored)==waveform_sha(chunk),'Review WAV differs from model input')
            audio=f'<audio controls preload="none" src="{name}"></audio>'
        row=dict(track_id=track['track_id'],speaker_id=track['speaker_id'],split=track['split'],
                 chunk_index=index,source_start_s=index*2.4,source_end_s=(index+1)*2.4,
                 analysis=record['analysis'],generation=record['generation'],audio=name if audio else None)
        rows.append(row)
        cards.append('<article><h3>'+html.escape(f'{row["speaker_id"]} · {index*2.4:.3f}–{(index+1)*2.4:.3f}s · {record["analysis"]["status"]}')+'</h3>'+audio+'<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></article>')
    save(a.out/'REVIEW.json',dict(plan=str(a.plan.resolve()),protocol_sha256=sha(a.plan/'protocol.json'),
        sampling='round-robin by audited status; diagnostic, not a retention estimate',rows=rows))
    (a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GLARE source prosody review</title><style>body{max-width:1100px;margin:24px auto;font:16px sans-serif}article{border:1px solid #ccc;padding:16px;margin:16px 0}pre{white-space:pre-wrap}</style><h1>原 speaker 音轨块：韵律原始回答复核</h1><p>按状态分层抽样；声音是实际模型输入。通过格式检查不代表人工确认正确。</p>'+''.join(cards))
    print(a.out/'index.html')


if __name__=='__main__':main()
