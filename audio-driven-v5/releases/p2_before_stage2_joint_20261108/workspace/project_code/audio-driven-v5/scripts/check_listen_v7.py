"""Bounded real-source geometry, timing, original roles and audio smoke checks."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import run_listen_v7_bulk as bulk
from p3_listen.common import read,write


def main():
    out=bulk.ROOT/'experiments/p3_listen_v7_validation'
    rows=read(bulk.OUT/'catalog.json')['rows'];lookup={r['pair']:r for r in rows}
    identifiers=['naturalistic_V03_S0847_I00000064','naturalistic_V01_S0065_I00000126',
                 'naturalistic_V03_S0985_I00000432']
    chosen=[lookup[p] for p in identifiers]+random.Random(20261022).sample(rows,32)
    bulk.OUT=out;bulk.preview.OUT=out;bulk.subturns.OUT=out
    with ThreadPoolExecutor(max_workers=8) as pool:
        audits=list(pool.map(lambda row:bulk.preview.inspect_pair(row,True),chosen))
    counts=Counter();hours=Counter()
    for item,audit in zip(chosen,audits):
        intervals=[]
        for row in audit['clips']:
            assert 2-1e-7<=row['duration_s']<=10+1e-7
            assert abs(row['end']-row['start']-row['duration_s'])<1e-7
            assert row['listener_crop']['size']>=448 and not row['listener_crop']['wide_crop']
            for role in ('speaker','listener'):
                index=row[role+'_index']
                assert item['ab'[index]]['id']==row[role+'_id']
                assert Path(row['sources'][role]['member']).stem==row[role+'_id']
            intervals.append((row['start'],row['end']))
            counts[row['duration_tier']]+=1;hours[row['duration_tier']]+=row['duration_s']/3600
        intervals.sort()
        assert all(b<=c+1e-7 for (_,b),(c,_) in zip(intervals,intervals[1:]))
    assert not audits[2]['ok'] and audits[2]['rejected'][0]['reason']=='source_orientation_not_upright'
    ast=bulk.EventModel(bulk.MODEL,batch_size=16);panns=bulk.PannsModel(batch_size=16)
    rejects=read(bulk.ROOT/'data_reviews/listener_noise_rejections.json')['rows']
    scanned=[]
    for item,audit in list(zip(chosen,audits))[:2]:
        assert audit['ok'] and audit['clips']
        norm=bulk.normalize(item,audit['clips'])
        for i,track in enumerate(norm['tracks']):
            assert Path(track['contract']['source']['member']).stem==item['ab'[i]]['id']
        for row in audit['clips'][:3]:
            scanned.append(bulk.scan(row,norm,ast,panns,rejects,[]))
    write(out/'audio_smoke.json',dict(clips=scanned))
    value=dict(real_pairs=len(audits),source_ok=sum(a['ok'] for a in audits),
        clips=sum(len(a['clips']) for a in audits),duration_counts=dict(counts),hours=dict(hours),
        audio_scanned=len(scanned),audio_decisions=dict(Counter(r['effective_decision'] for r in scanned)),
        assertions_passed=True,note='Bounded engineering smoke, not a corpus yield estimate or human audio review.')
    write(out/'validation.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
