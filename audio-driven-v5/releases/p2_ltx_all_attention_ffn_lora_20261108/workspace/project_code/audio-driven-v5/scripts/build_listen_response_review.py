"""Prepare a CPU-only review of risky response merges; keep all 50 samples available."""
import argparse,hashlib,json,shutil,sys,wave
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.response_review import VERSION,alignment_track,deduplicate_tracks,review_group

def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def build(source,alignment,output,preview,font=None):
    for p in (output,preview):
        if p.exists() and any(p.iterdir()):raise ValueError('Output must be empty: '+str(p))
    raw=(source/'observations.jsonl').read_bytes();original={d['sample_id']:d for d in map(json.loads,raw.splitlines())}
    old=json.loads((source/'review.json').read_text());samples=[];hashes={};totals=dict(samples=0,focus_samples=0,review_groups=0,reference_tokens=0,invalid_time_tokens=0,candidate_duplicate_clusters=0)
    for s in old['samples']:
        episodes=[e for e in s['events'] if len(e.get('member_event_ids',[]))>1]
        tracks=[]
        if episodes:
            for p in sorted(alignment.glob(s['id']+'*.json')):
                b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();tracks.append(alignment_track(json.loads(b),s['duration']))
        groups=[review_group(e,original[s['id']]['events'],tracks,s['decisions']) for e in episodes]
        clusters=deduplicate_tracks(tracks)
        tokens=[t for tr in tracks for t in tr['tokens']]
        assert sorted(t['id'] for t in tokens)==sorted(t['id'] for c in clusters for t in c['members'])
        assert all(len({t['pass_id'] for t in c['members']})==len(c['members']) for c in clusters)
        totals['samples']+=1;totals['focus_samples']+=bool(groups);totals['review_groups']+=len(groups)
        totals['reference_tokens']+=len(tokens);totals['invalid_time_tokens']+=sum(not t['time_valid'] for t in tokens)
        totals['candidate_duplicate_clusters']+=sum(len(c['members'])>1 for c in clusters)
        samples.append(dict(id=s['id'],duration=s['duration'],video=s['video'],audio=s['audio'],
            review_groups=groups,alignment_tracks=tracks,token_clusters=clusters,
            original_events=original[s['id']]['events'],
            visual_candidates=[e for e in original[s['id']]['events'] if e['modality']=='visual' and e['mapping_status']=='mapped_candidate']))
    assert totals['review_groups']==5
    data=dict(schema=VERSION,run_id=output.name,summary=totals,samples=samples,human_ground_truth=False,
              policy=dict(raw_preserved=True,full_clip_asr_is_reference_not_truth=True,boundary_match_tolerance_s=.2,
                          same_pass_tokens_never_merged=True,transitive_interval_union=False,media_inference_rerun=False,
                          manual_listening_completed=False,training_admitted=False))
    # CPU waveform is for listening/positioning only, not a VAD or repetition detector.
    import numpy as np
    import soundfile as sf
    media_root=ROOT/'data_previews/listen_benchmark_review'
    for s in samples:
        if not s['review_groups']:continue
        path=(media_root/'response_review'/s['audio']).resolve()
        audio,rate=sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
        step=max(1,rate//100);s['waveform']=[float(np.max(np.abs(audio[i:i+step]))) for i in range(0,len(audio),step)]
        s['waveform_step_s']=step/rate;s['media_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    html=(ROOT/'scripts/listen_response_review.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    font_bytes=None
    if font:
        import io
        from fontTools.ttLib import TTFont
        from fontTools.subset import Subsetter,Options
        face=TTFont(str(font),fontNumber=2);options=Options();options.flavor='woff'
        subset=Subsetter(options=options);subset.populate(text=html);subset.subset(face)
        face.flavor='woff';buffer=io.BytesIO();face.save(buffer);font_bytes=buffer.getvalue()
    else:
        # Do not reuse an old subset font: it may lack the new review text.
        html=html.replace('@font-face{font-family:Review;src:url(review.woff)}','')
    for dest in (output,preview):
        dest.mkdir(parents=True,exist_ok=True);(dest/'observations.jsonl').write_bytes(raw)
        (dest/'index.html').write_text(html);dump(dest/'review.json',data);dump(dest/'summary.json',totals)
        (dest/'review_candidates.jsonl').write_text(''.join(json.dumps(dict(sample_id=s['id'],review_groups=s['review_groups'],visual_candidates=s['visual_candidates']),ensure_ascii=False)+'\n' for s in samples))
        if font_bytes:(dest/'review.woff').write_bytes(font_bytes)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),source=str(source),alignment_source=str(alignment),
                  source_observation_sha256=hashlib.sha256(raw).hexdigest(),alignment_sha256=hashes,code_sha256={})
    for name in ['src/listen_benchmark/response_review.py','scripts/build_listen_response_review.py','scripts/listen_response_review.html','tests/test_listen_response_review.py']:
        p=ROOT/name;target=output/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);manifest['code_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(output/'manifest.json',manifest);print(json.dumps(totals))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);p.add_argument('--font',type=Path);a=p.parse_args();build(a.source.resolve(),a.alignment.resolve(),a.output.resolve(),a.preview.resolve(),a.font)
