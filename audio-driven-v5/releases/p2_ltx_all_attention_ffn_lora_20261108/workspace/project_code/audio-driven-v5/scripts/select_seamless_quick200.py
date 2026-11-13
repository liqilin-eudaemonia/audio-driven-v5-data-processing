"""Frozen, source-balanced diagnostic subset of the approved Seamless test release."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import orjson

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / 'experiments/listen_j_labels_manifest_20261105'
PARENT = ROOT.parent / 'OmniResponse/evaluation/seamless/prepared_20261102'
SEED = 'seamless-quick200-latest-20261105-v1'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT.parent / 'OmniResponse')]
HINTS = {
    'head_nod': r'\bnod(?:s|ded|ding)?\b',
    'head_shake': r'\bhead[- ]shak\w*|\bshak\w*\b.{0,35}\bhead\b|\bhead\b.{0,25}\bside[- ]to[- ]side\b',
    'smile': r'\bsmil\w*|\bgrin\w*',
    'laugh': r'\blaugh\w*|\bchuckl\w*|\bgiggl\w*',
    'brow_lower_frown': r'\bfrown\w*|\bfurrow\w*.{0,20}\bbrow\w*|\bbrow\w*.{0,20}\blower\w*',
    'brow_raise': r'\brais\w*.{0,22}\b(?:eyebrows?|brows?)\b|\b(?:eyebrows?|brows?)\b.{0,22}\brais\w*',
    'blink': r'\bblink\w*',
    'eye_close': r'\bclos\w*.{0,18}\beyes?\b|\beyes?\b.{0,18}\bclos\w*',
    'gaze_change': r'\b(?:shift\w*|turn\w*|mov\w*).{0,20}\b(?:gaze|eyes)\b|\blook\w*.{0,12}\b(?:down|up|away|left|right)\b',
    'head_tilt': r'\btilt\w*.{0,18}\bhead\b|\bhead\b.{0,18}\btilt\w*',
    'head_turn': r'\bturn\w*.{0,18}\bhead\b|\bhead\b.{0,18}\bturn\w*',
    'hand_motion': r'\b(?:rais\w*|lift\w*|lower\w*|mov\w*|gestur\w*).{0,25}\b(?:hands?|arms?)\b|\b(?:hands?|arms?)\b.{0,25}\b(?:rais\w*|lift\w*|lower\w*|mov\w*|gestur\w*)',
    'mouth_open': r'\bopen\w*.{0,18}\bmouth\b|\bmouth\b.{0,18}\bopen\w*',
    'mouth_close': r'\bclos\w*.{0,18}\bmouth\b|\bmouth\b.{0,18}\bclos\w*',
    'lip_purse': r'\bpurs\w*.{0,18}\blips?\b|\blips?\b.{0,18}\bpurs\w*',
    'posture_change': r'\blean\w*|\bshrug\w*|\b(?:shift\w*|mov\w*).{0,18}\b(?:body|shoulder|posture)\b',
}


def text_hints(text):
    """Selection hints with exact quotes, never a replacement taxonomy or GT count."""
    result = {}
    for key, pattern in HINTS.items():
        evidence = []
        for match in re.finditer(pattern, text, re.I):
            before = re.split(r'\b(?:but|then|however)\b|[.;!?]', text[max(0, match.start()-65):match.start()], flags=re.I)[-1]
            if re.search(r'\b(?:not|never|without|no)(?:\s+\w+){0,3}\s*$', before, re.I): continue
            evidence.append(dict(start=match.start(), end=match.end(), quote=match.group()))
        if evidence: result[key] = evidence
    return result


def merge_intervals(spans):
    result = []
    for a, b in sorted(spans):
        if b <= a: continue
        if result and a <= result[-1][1] + 1e-8: result[-1][1] = max(result[-1][1], b)
        else: result.append([a, b])
    return result


def official_spans(row):
    d = row['duration_s']
    return merge_intervals((max(0., x['start']-.1), min(d, x['end']+.1))
        for values in row['audio_recipe']['official_listener'].values() for x in values
        if x['end'] > 0 and x['start'] < d)


def audio_measure(row, out):
    import numpy as np
    from listen_benchmark.j_omni import read_audio
    path = out / 'audio_checks' / (row['sample_id'] + '.json')
    if path.exists(): return orjson.loads(path.read_bytes())
    cap_raw = Path(row['caption_file']).read_bytes()
    if hashlib.sha256(cap_raw).hexdigest() != row['caption_file_sha256']:
        raise ValueError('Latest raw caption changed')
    cap = orjson.loads(cap_raw)
    if cap['caption'] != row['caption'] or cap['source'] != row['source']:
        raise ValueError('Latest caption/source differs from manifest')
    wave, sr, audit = read_audio(row['audio_recipe'])
    if audit['processed_waveform_sha256'] != cap['input_evidence']['audio_identity']['processed_waveform_sha256']:
        raise ValueError('Purified audio differs from latest caption input')
    if audit['full_silence'] != row['listener_audio_is_silent']:
        raise ValueError('Silent flag differs from actual waveform')
    mono = wave.reshape(-1).astype(np.float64)
    window = round(.02 * sr); hop = round(.01 * sr)
    if len(mono) < window: raise ValueError('Clip too short')
    starts = np.arange(0, len(mono)-window+1, hop)
    sums = np.r_[0., np.cumsum(mono * mono)]
    rms = np.sqrt(np.maximum(0., (sums[starts+window]-sums[starts])/window))
    active = rms >= 10 ** (-50/20)
    spans = merge_intervals((float(a/sr), float((a+window)/sr)) for a in starts[active])
    longest = max((b-a for a, b in spans), default=0.)
    group = 'silent' if audit['full_silence'] else 'long' if longest >= 1.44 else 'short'
    ev = cap['input_evidence']
    result = dict(sample_id=row['sample_id'], group=group, sample_rate=sr, samples=len(mono),
        full_silence=audit['full_silence'], processed_waveform_sha256=audit['processed_waveform_sha256'],
        active_intervals=spans, active_seconds=sum(b-a for a, b in spans), longest_active_seconds=longest,
        rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(mono*mono))), 1e-12))),
        measure='20ms RMS/10ms hop >= -50dBFS; acoustic proxy, not semantic speech or word count',
        nod_cues=ev.get('supplemental_nods', ev.get('nod_intervals', [])),
        vocal_cues=(ev.get('vocal_evidence') or {}).get('cues', []),
        reaction_cues=(ev.get('reaction_supplement') or {}).get('cues', []))
    write(path, result)
    return result


def candidate_features(row):
    hints = text_hints(row['caption'])
    spans = official_spans(row)
    longest = max((b-a for a,b in spans), default=0.)
    group = 'silent' if row['listener_audio_is_silent'] else 'long' if longest >= 1.44 else 'short'
    caption = row['caption']
    return dict(id=row['sample_id'], row=row, hints=hints, tags=set(hints), audio_group=group,
        source=row['sample_id'].split('_',1)[0], group_id=row['group_id'], pair=row['pair'],
        person=row['listener_id'].rsplit('_',1)[-1], duration=row['duration_s'], face=row['face_p10'],
        duration_bin='short' if row['duration_s']<4 else 'mid' if row['duration_s']<8 else 'long',
        repeated=bool(re.search(r'\b(?:again|repeatedly|several|twice|multiple times)\b',caption,re.I)),
        simultaneous=bool(re.search(r'\b(?:while|simultaneously|at the same time)\b',caption,re.I)),
        sustained=bool(re.search(r'\b(?:throughout|maintains|remains|sustained|steady)\b',caption,re.I)))


def choose(features, quotas, out, actual=True):
    picked=[]; pairs=set(); audio=Counter(); tags=Counter(); sources=Counter(); groups=Counter(); people=Counter(); durations=Counter()
    populations=Counter(k for f in features for k in f['tags'])
    total=sum(quotas.values()); attempts=0
    while len(picked)<total:
        options=[]
        # Reserve the scarce long-response pool before abundant silent/short
        # clips consume its source-pair slots.
        focus=next(k for k in ('long','short','silent') if audio[k]<quotas[k])
        moderate=sum(x['face']<224 for x in picked)
        for f in features:
            if f['pair'] in pairs or groups[f['group_id']]>=6 or people[f['person']]>=6: continue
            if f['audio_group']!=focus: continue
            if audio[f['audio_group']]>=quotas[f['audio_group']]: continue
            # Primarily large faces; cap moderate faces at 10%, with at most 2 in core48.
            if f['face']<224 and moderate >= (20 if total==200 else 2): continue
            target_source = .2*total if f['source']=='improvised' else .8*total
            diversity=sum((2.8 if populations[k]<50 else 1.7) / (1+tags[k]) for k in f['tags'])
            diversity+=1.2/(1+groups[f['group_id']])+.8/(1+people[f['person']])
            diversity+=1.4*max(0.,1-sources[f['source']]/max(1.,target_source))
            diversity+=.9/(1+durations[f['duration_bin']]/(total/3))
            diversity+=.7*min(f['face'],384)/384 + .3*f['repeated']+.3*f['simultaneous']
            diversity-=.08*max(0.,f['duration']-8)
            tie=hashlib.sha256((SEED+'\0'+f['id']).encode()).hexdigest()
            options.append((diversity,tie,f))
        if not options:
            raise ValueError('Unable to fill independent pair/audio quotas: '+repr(dict(audio)))
        options.sort(key=lambda x:(-x[0],x[1])); f=options[0][2]
        if actual and 'audio' not in f:
            measured=audio_measure(f['row'],out);f['audio']=measured
            f['audio_group']=measured['group']
            if not measured['full_silence'] and measured['active_seconds']<.04:
                # Extremely weak nonzero residue is not admitted as a voiced positive.
                features.remove(f);continue
            # The group may have changed, so rerank using the actual quota cell.
            continue
        picked.append(f);pairs.add(f['pair']);audio[f['audio_group']]+=1
        tags.update(f['tags']);sources[f['source']]+=1;groups[f['group_id']]+=1;people[f['person']]+=1;durations[f['duration_bin']]+=1
        attempts+=1
        if len(picked)%25==0:print(json.dumps(dict(selected=len(picked),total=total,audio=dict(audio))),flush=True)
    return picked


def summarize(selected):
    return dict(clips=len(selected),seconds=sum(f['duration'] for f in selected),
        minutes=sum(f['duration'] for f in selected)/60,audio_groups=dict(Counter(f['audio_group'] for f in selected)),
        source_types=dict(Counter(f['source'] for f in selected)),
        duration_bins=dict(Counter(f['duration_bin'] for f in selected)),
        face_ge224=sum(f['face']>=224 for f in selected),face_160_to224=sum(f['face']<224 for f in selected),
        face_p10_median=sorted(f['face'] for f in selected)[len(selected)//2],
        unique_pairs=len({f['pair'] for f in selected}),unique_groups=len({f['group_id'] for f in selected}),
        unique_listener_codes=len({f['person'] for f in selected}),
        max_per_group=max(Counter(f['group_id'] for f in selected).values()),
        textual_hint_clip_counts=dict(Counter(k for f in selected for k in f['tags'])),
        repeated_description=sum(f['repeated'] for f in selected),simultaneous_description=sum(f['simultaneous'] for f in selected),
        acoustic_windows_ge3s=sum(f['audio']['longest_active_seconds']>=3.01 for f in selected),
        hint_semantics='Quoted positive text hints for sampling only; not validated behavior classes or exact event counts')


def select(out):
    audit=orjson.loads((out/'CANDIDATES_AUDIT.json').read_bytes())
    candidates=rows(out/'candidates.latest.jsonl');other=set(audit['other_split_group_ids'])
    eligible=[candidate_features(r) for r in candidates if r['generation_checks_passed'] and not r['review_flags']
        and r['finish_reason']=='stop' and r['face_p10'] is not None and r['face_p10']>=160
        and r['group_id'] not in other and r['quality']['decision']=='retain']
    if any(r['split']!='test' for r in candidates): raise ValueError('Non-test candidate')
    picked=choose(eligible,dict(silent=64,short=116,long=20),out)
    core=choose(list(picked),dict(silent=16,short=24,long=8),out,actual=False)
    core_ids={f['id'] for f in core};ordered=core+[f for f in picked if f['id'] not in core_ids]
    records=[]
    for i,f in enumerate(ordered,1):
        r=f['row']
        records.append(dict(id=f['id'],rank=i,core48=f['id'] in core_ids,split='test',pair=f['pair'],
            group_id=f['group_id'],source_type=f['source'],duration_s=f['duration'],face_p10_px=f['face'],
            audio_group=f['audio_group'],audio_check=str(out/'audio_checks'/(f['id']+'.json')),
            caption=r['caption'],caption_file=r['caption_file'],caption_sha256=r['caption_file_sha256'],
            latest_manifest_reference=r['_latest_manifest_reference'],
            selection_hints=f['hints'],sampling_notes=dict(repeated=f['repeated'],simultaneous=f['simultaneous'],sustained=f['sustained']),
            annotation_status='automatic_candidate_not_human_ground_truth'))
    (out/'selection.jsonl').write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in records))
    (out/'latest_labels.test200.jsonl').write_bytes(b''.join(orjson.dumps(f['row'])+b'\n' for f in ordered))
    with (out/'selection.csv').open('w',newline='') as stream:
        names=['rank','id','core48','split','source_type','pair','group_id','duration_s','face_p10_px','audio_group','caption','caption_file']
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    write(out/'summary.json',dict(schema='seamless-quick200-latest-v1',seed=SEED,
        candidates=len(candidates),eligible=len(eligible),latest_caption_identity=audit['latest_caption_identity'],
        full=summarize(picked),core48=summarize(core),split_group_overlap=0,train_dev_members=0,
        original_labels_modified=False,human_review_complete=False,selection_sha256=sha(out/'selection.jsonl'),
        population_representative=False,selection='Stratified diagnostic subset; one per source pair; positive text hints only; actual purified audio verified',
        audio_group_definitions=dict(silent='Waveform entirely zero',short='Nonzero, RMS activity >=40ms, longest continuous activity <1.44s',long='Longest continuous RMS activity >=1.44s; does not imply semantic speech')))


def prepare(out):
    """Preserve the approved media reader; attach new labels as separate GT metadata."""
    from evaluation.seamless.reader import resolve, reader_contract
    from evaluation.seamless.bind import load_references
    selections=rows(out/'selection.jsonl'); latest={r['sample_id']:r for r in rows(out/'latest_labels.test200.jsonl')}
    parent_refs={r['id']:r for r in load_references(PARENT)}
    if len(selections)!=200 or len(latest)!=200: raise ValueError('Expected frozen 200 selection')
    checks=[]
    for selected in selections:
        sid=selected['id']; row=latest[sid]; ref=parent_refs[sid]
        old_row,resolved=resolve(ref['gt_reference']); src=resolved['source']; new=row['source']
        for key in ('sample_id','split','listener_id','speaker_id','source_start_s','source_end_s','duration_s','source','video_transform'):
            if src[key]!=new[key]: raise ValueError('Latest/approved media mismatch: '+sid+' '+key)
        raw=Path(row['caption_file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['caption_file_sha256']: raise ValueError('Latest raw caption changed')
        cap=orjson.loads(raw)
        oldcap=orjson.loads(Path(old_row['target']['source_file']).read_bytes())
        if (oldcap['input_evidence']['audio_identity']['processed_waveform_sha256'] !=
                cap['input_evidence']['audio_identity']['processed_waveform_sha256']):
            raise ValueError('Approved GT audio differs from latest annotation input')
        if oldcap['audio_recipe']!=cap['audio_recipe']: raise ValueError('Approved/latest audio recipes differ')
        check=orjson.loads((out/'audio_checks'/(sid+'.json')).read_bytes())
        checks.append(dict(id=sid,latest_caption_sha256=row['caption_file_sha256'],
            same_source_time_participants_final_crop=True,same_purified_audio_recipe_and_hash=True,
            actual_waveform_sha256=check['processed_waveform_sha256']))
    for name,subset in [('prepared_200',selections),('prepared_core48',[r for r in selections if r['core48']])]:
        folder=out/name; folder.mkdir(exist_ok=False)
        shutil.copyfile(PARENT/'crop_config.json',folder/'crop_config.json')
        references=[]; offset=0; annotations=[]; inputs=[]
        with (folder/'source.test.jsonl').open('wb') as target, (PARENT/'source.test.jsonl').open('rb') as source:
            for selected in subset:
                sid=selected['id']; previous=parent_refs[sid]; row=latest[sid]
                parent_ref=previous['gt_reference']; source.seek(parent_ref['offset']); raw=source.read(parent_ref['bytes'])
                if hashlib.sha256(raw).hexdigest()!=parent_ref['row_sha256']: raise ValueError('Parent row changed')
                target.write(raw)
                reference=dict(parent_ref,manifest=str(folder/'source.test.jsonl'),offset=offset,bytes=len(raw),
                    crop_config=str(folder/'crop_config.json'),reader_contract=reader_contract())
                offset+=len(raw)
                annotation=dict(caption=row['caption'],caption_file=row['caption_file'],caption_sha256=row['caption_file_sha256'],
                    caption_identity=row['caption_identity_version'],latest_manifest_reference=row['_latest_manifest_reference'],
                    human_ground_truth=False,original_language_preserved=True)
                references.append(dict(previous,gt_reference=reference,latest_annotation=annotation,
                    latest_annotation_crop_differs=False,selection_rank=selected['rank'],core48=selected['core48'],
                    audio_group=selected['audio_group']))
                annotations.append(dict(id=sid,**annotation))
                media=orjson.loads(raw)
                inputs.append(dict(id=sid,dataset='seamless',split='test',duration_s=row['duration_s'],
                    speaker_audio=media['inputs']['speaker_audio'],listener_first_frame=media['inputs']['listener_first_frame'],
                    gt_reference=reference,annotation_file=str(folder/'annotations.jsonl'),
                    conditioning='speaker_audio_and_listener_first_frame_only',target_text_is_input=False))
        for filename,records in [('references.jsonl',references),('annotations.jsonl',annotations),('inference_inputs.jsonl',inputs)]:
            (folder/filename).write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in records))
        (folder/'generation_map.template.jsonl').write_bytes(b''.join(orjson.dumps(dict(id=r['id'],dataset='seamless',
            generated=None,generated_audio=None,target_role='listener',timeline='same_pair_window',expected_fps=25,
            expected_width=512,expected_height=512,expected_frames=r['frames'],pair_duration_s=r['duration_s']))+b'\n' for r in references))
        ids=sorted(r['id'] for r in references)
        write(folder/'seal.json',dict(schema='seamless-approved-subset-v1',dataset='seamless',expected_clips=len(references),
            parent_prepared=str(PARENT),parent_seal_sha256=sha(PARENT/'seal.json'),
            parent_references_sha256=sha(PARENT/'references.jsonl'),
            selected_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            references_sha256=sha(folder/'references.jsonl'),source_sha256=sha(folder/'source.test.jsonl'),
            crop_config_sha256=sha(folder/'crop_config.json'),reader_contract=reader_contract(),
            latest_annotations_sha256=sha(folder/'annotations.jsonl'),selection_sha256=sha(out/'selection.jsonl')))
        assert len(load_references(folder))==len(references)
    write(out/'PREPARED_AUDIT.json',dict(success=True,clips=200,core48=48,checks=checks,
        latest_annotations_preserved=True,media_and_audio_identical_to_approved_parent=True,
        prior_caption_only_in_unchanged_media_provenance_rows=True,
        new_caption_used_for_selection_and_review_not_as_generation_input=True,
        full_media_decoding='performed by actual inference/metrics; metadata and all selected purified waveforms checked here',
        prepared_200=str(out/'prepared_200'),prepared_core48=str(out/'prepared_core48')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rows(path):
    with Path(path).open('rb') as stream:
        return [orjson.loads(line) for line in stream if line.strip()]


def extract(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'CANDIDATES_AUDIT.json').exists():
        audit = orjson.loads((out / 'CANDIDATES_AUDIT.json').read_bytes())
        if sha(out / 'candidates.latest.jsonl') != audit['candidate_manifest_sha256']:
            raise ValueError('Frozen candidates changed')
        return
    summary = orjson.loads((LATEST / 'summary.json').read_bytes())
    assert orjson.loads((LATEST / 'COVERAGE_AUDIT.json').read_bytes())['passed']
    ids = {r['id'] for r in rows(PARENT / 'references.jsonl')}
    assert len(ids) == 6624
    h = hashlib.sha256(); groups = set(); overlap = set(); found = set(); offset = 0; count = 0
    begin = time.time(); source = LATEST / 'labels.jsonl'
    with source.open('rb', buffering=4 * 1024 * 1024) as stream, (out / 'candidates.latest.jsonl').open('wb') as target:
        for raw in stream:
            h.update(raw)
            # Only parse the compact prefix for train/dev; one sequential scan,
            # no parallel stat/open storm over one million caption files.
            i = raw.find(b',"caption":')
            if i < 0: i = raw.find(b', "caption":')
            if i < 0: raise ValueError('Unexpected manifest header layout')
            header = orjson.loads(raw[:i] + b'}'); sid = header['sample_id']
            if header['split'] in ('train', 'dev'):
                groups.add(header['group_id'])
                if sid in ids: overlap.add(sid)
            elif header['split'] == 'test' and sid in ids:
                if sid in found: raise ValueError('Duplicate latest test ID')
                r = orjson.loads(raw)
                r['_latest_manifest_reference'] = dict(path=str(source), offset=offset, bytes=len(raw),
                    row_sha256=hashlib.sha256(raw).hexdigest())
                target.write(orjson.dumps(r) + b'\n'); found.add(sid)
            offset += len(raw); count += 1
            if count % 200000 == 0:
                print(json.dumps(dict(scanned=count, matched=len(found), elapsed_s=round(time.time()-begin, 1))), flush=True)
    if h.hexdigest() != summary['files']['labels']['sha256'] or found != ids or overlap:
        raise ValueError('Latest source hash, test coverage or split isolation differs')
    write(out / 'CANDIDATES_AUDIT.json', dict(source=str(source), sha256=h.hexdigest(), bytes=offset,
        rows=count, matched=len(found), parent_prepared=str(PARENT),
        latest_caption_identity=summary['caption_identity_version'], other_split_group_ids=sorted(groups),
        candidate_manifest_sha256=sha(out / 'candidates.latest.jsonl'), elapsed_s=time.time()-begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('extract','select','prepare'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); {'extract':extract,'select':select,'prepare':prepare}[a.command](a.output.resolve())
