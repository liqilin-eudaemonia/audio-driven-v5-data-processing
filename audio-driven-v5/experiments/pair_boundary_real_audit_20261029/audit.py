"""Read-only production segmentation replay and diagnostic alternatives.

Alternatives use the unchanged production acoustic cutter, but are NOT QC-approved.
Run from the project root: python experiments/pair_boundary_real_audit_20261029/audit.py
"""
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import listen_local_recovery as r


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.item()) + '\n')


def covered(w, clips):
    return any(c['start'] <= w['start'] + 1e-6 and c['end'] >= w['end'] - 1e-6 for c in clips)


def analyze(d):
    ps, limit = d['people'], d['limit']
    rms = [np.array(x) for x in d['rms']]
    results = []
    for reg in d['regions']:
        si = reg['speaker_index']
        blocks = [(x['start'], x['end']) for x in d['diagnostic']['blocked'] if x['speaker_index'] == si]
        blocks += r.semantic_blocks(ps[1-si])
        blocks += [(w['start']-.15, w['end']+.15) for p in ps for w in p['words']
                   if math.isfinite(w['start']) and math.isfinite(w['end']) and w['end']-w['start'] > 1.5]
        for a, b in r.subtract([(reg['start'], reg['end'])], blocks):
            if b-a < 2:
                continue
            cs, bad = r.recover_region(ps, rms, si, a, b, limit)
            units = r.sentence_units(ps[si]['words'])

            def cut(ws, left, right):
                if not ws or ws[-1]['end']-ws[0]['start'] < 1.6:
                    return None
                if any(v['start'] < u['end']-.001 or v['start']-u['end'] > 3 for u,v in zip(ws,ws[1:])):
                    return None
                c, error = r.old.acoustic_cut(r.make_unit(ws, units), ps, rms, si, limit, left, right)
                if error or c['start'] < left-1e-6 or c['end'] > right+1e-6:
                    return None
                if any(w['start']+1e-6 < edge < w['end']-1e-6
                       for p in ps for w in p['words'] for edge in (c['start'],c['end'])):
                    return None
                return c

            if cs and bad:
                ws = [w for w in ps[si]['words'] if w['start'] >= cs[-1]['speech_start']-1e-7
                      and w['end'] <= b and 0 < w['end']-w['start'] <= 1.5]
                left = cs[-2]['end'] if len(cs)>1 else a
                options = []
                full = cut(ws,left,b)
                if full:
                    options.append([full])
                for k in range(1,len(ws)):
                    c = cut(ws[:k],left,b)
                    c2 = cut(ws[k:],c['end'],b) if c else None
                    if c2:
                        options.append([c,c2])
                for alt in options:
                    added = [w for w in ws if not covered(w,cs) and covered(w,alt)]
                    if not added or not all(covered(w,alt) for w in ws if covered(w,[cs[-1]])):
                        continue
                    results.append(dict(kind='greedy_tail',pair=d['pair'],domain=[a,b],speaker_index=si,
                                        current=cs,alternative=cs[:-1]+alt,added_words=added,failures=bad))
                    break

            # Only test extensions into a source gap, never into the next role,
            # known quality blocker, or an existing planned clip.
            if abs(b-reg['end']) > 1e-6:
                continue
            for feedback in ps[1-si]['segments']:
                if not feedback['feedback'] or not (feedback['start'] < b < feedback['end'] <= b+1.5):
                    continue
                next_start = min([x['start'] for x in d['regions'] if x['start'] >= b-1e-7 and x is not reg] + [limit])
                next_block = min([x for x,y in blocks if x >= b-1e-7] + [limit])
                next_clip = min([x['start'] for x in d['clips'] if x['start'] >= b-1e-7] + [limit])
                extended = min(feedback['end']+.3,next_start,next_block,next_clip,limit)
                if extended < feedback['end']:
                    continue
                alt, alt_bad = r.recover_region(ps,rms,si,a,extended,limit)
                if not covered(feedback,alt) or covered(feedback,cs):
                    continue
                results.append(dict(kind='feedback_tail',pair=d['pair'],domain=[a,b],extended_end=extended,
                                    speaker_index=si,feedback=feedback,current=cs,alternative=alt,
                                    failures=bad,alternative_failures=alt_bad))
    return results


def replay(item):
    pair = item['pair']
    cached = OUT / (pair+'.json')
    if cached.exists():
        return json.loads(cached.read_text())
    ctx = r.SourceContext(item,shot_times=[[],[]])
    previous = ROOT/'data_corpora/seamless_listen_v8_audio_final_full/results'/(pair+'.json')
    prior = json.loads(previous.read_text()) if previous.exists() else dict(clips=[])
    clips, diagnostic = ctx.propose(prior['clips'])
    production_path = ROOT/'data_corpora/seamless_listen_local_full/results'/(pair+'.json')
    production = json.loads(production_path.read_text())
    d = dict(pair=pair,item=item,people=ctx.people,rms=[x.tolist() for x in ctx.rms],
             regions=ctx.regions,source_gaps=ctx.source_gaps,limit=ctx.limit,sources=ctx.sources,
             clips=clips,diagnostic=diagnostic,production_clips=production['clips'])
    save(cached,d)
    return d


if __name__ == '__main__':
    catalog = json.loads((ROOT/'data_corpora/seamless_listen_local_full/catalog.json').read_text())['rows']
    rng = random.Random(20261029)
    selected = rng.sample([x for x in catalog if x['label']=='naturalistic' and x['split']=='train'],24)
    prior = ['naturalistic_V02_S3610_I00000088','improvised_V01_S0122_I00000125']
    selected = [next(x for x in catalog if x['pair']==p) for p in prior] + selected
    all_results, inspected = [], []
    for item in selected:
        try:
            d = replay(item)
            results = analyze(d)
            inspected.append(dict(pair=d['pair'],seconds=d['limit'],clips=len(d['clips'])))
            all_results.extend(results)
            print(json.dumps(dict(pair=d['pair'],seconds=d['limit'],findings=[dict(kind=x['kind'],domain=x['domain'],
                words=[w['text'] for w in x.get('added_words',[])],feedback=x.get('feedback',{}).get('text'))
                for x in results]),ensure_ascii=False),flush=True)
        except Exception as e:
            inspected.append(dict(pair=item['pair'],error=repr(e)))
            print(item['pair'],repr(e),flush=True)
        save(OUT/'findings.json',dict(inspected=inspected,findings=all_results))
        if any(x['kind']=='feedback_tail' for x in all_results):
            break
