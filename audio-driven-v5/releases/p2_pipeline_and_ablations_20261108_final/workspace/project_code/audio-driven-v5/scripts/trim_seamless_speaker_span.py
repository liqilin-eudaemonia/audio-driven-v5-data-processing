"""V4: trim V2 previews to speaker word bounds, preserving overlapping feedback."""
import argparse
import collections
import concurrent.futures
import copy
import json
import math
from pathlib import Path

import build_seamless_listening_preview as media
from build_seamless_rule_turns import prepare, save

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'data_previews/seamless_rule_turns_v2'
V3 = ROOT / 'data_previews/seamless_speaker_end_v3'
OUT = ROOT / 'data_previews/seamless_speaker_span_v4'
FILES = ['faces.mp4', 'speaker_512.mp4', 'listener_512.mp4', 'context.mp4',
         'speaker.m4a', 'listener.m4a', 'poster.jpg']
POLICY = 'speaker_first_word_start_to_last_word_end_clamped_to_v2_interval'


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(source.resolve())


def revise(clip, people):
    c = copy.deepcopy(clip)
    old_a, old_b = c['start'], c['end']
    speaker = people[c['speaker']]
    words = [w for w in speaker['words'] if w['start'] < old_b-.001 and w['end'] > old_a+.001]
    assert words, c['path']
    first, last = min(w['start'] for w in words), max(w['end'] for w in words)
    a, b = round(max(old_a, first), 3), round(min(old_b, last), 3)
    assert old_a <= a < b <= old_b
    c.update(start=a, end=b, duration=round(b-a, 3), previous_start=old_a, previous_end=old_b,
             previous_start_reason=c['start_reason'], previous_end_reason=c['end_reason'],
             head_removed_seconds=round(a-old_a, 3), tail_removed_seconds=round(old_b-b, 3),
             speaker_first_word_start=first, speaker_last_word_end=last,
             boundary_policy=POLICY, segmentation_revision=4, media_revision=5,
             shorter_than_previous_minimum=b-a < 4,
             entire_floor_turn=abs(a-c['parent_turn_start']) < .001 and abs(b-c['parent_turn_end']) < .001)
    if a > old_a+.001: c['start_reason'] = 'speaker_first_word_start'
    if b < old_b-.001: c['end_reason'] = 'speaker_last_word_end'
    # Preserve the existing sampled emotion values and their true offsets; do not interpolate labels.
    sample_indices = [i for i in range(max(len(v) for role in c['emotion'].values() for v in role.values()))
                      if a <= old_a+i*.2 < b]
    c['emotion_sample_times'] = [round(old_a+i*.2-a, 6) for i in sample_indices]
    for role in ['speaker', 'listener']:
        person = people[c[role]]
        c[role+'_vad'] = [[round(max(a,x)-a, 6), round(min(b,y)-a, 6)]
                         for x,y in person['vad'] if min(b,y)-max(a,x) > .001]
        c[role+'_vad_fraction'] = round(media.overlap(person['vad'], a, b)/(b-a), 3)
        c[role+'_text'] = ' '.join(w['text'] for w in person['words']
                                 if min(b,w['end'])-max(a,w['start']) > .001)
        for key, values in c['emotion'][role].items():
            c['emotion'][role][key] = [values[i] if i < len(values) else None for i in sample_indices]
    c['listener_events'] = [dict(e, start=max(a,e['start']), end=min(b,e['end']))
                            for e in clip['listener_events'] if min(b,e['end'])-max(a,e['start']) > .001]
    c['kind'] = ('feedback_candidate' if any(e['label']=='feedback_candidate' for e in c['listener_events'])
                 else 'vocal_candidate' if c['listener_events'] else 'quiet_candidate')
    for event in c['parent_events']:
        event['in_export'] = not event['exclude'] and min(b,event['end']) > max(a,event['start'])
    return c


def render(c):
    dest = OUT / c['path']; dest.mkdir(parents=True, exist_ok=True)
    cached = dest/'metadata.json'
    if cached.exists():
        previous = json.loads(cached.read_text())
        assert (previous['start'],previous['end'],previous['boundary_policy']) == (c['start'],c['end'],POLICY)
        assert all((dest/name).exists() for name in FILES)
        return
    # Reuse an existing export only when both boundaries agree exactly.
    for folder in [V3, OLD]:
        source = folder/c['path']; metadata = source/'metadata.json'
        if metadata.exists():
            old = json.loads(metadata.read_text())
            if (old['start'],old['end']) == (c['start'],c['end']):
                for name in FILES: link(source/name, dest/name)
                c['media_source'] = str(folder.relative_to(ROOT))
                save(cached,c)
                return
    source = OLD/c['path']
    inputs=[]
    for name in FILES[:6]:
        inputs += ['-threads','1','-ss',str(c['head_removed_seconds']),'-i',str(source/name)]
    args = ['ffmpeg','-v','error','-nostdin','-y',*inputs]
    for i,name in enumerate(FILES[:6]):
        args += ['-map',str(i),'-t',str(c['duration'])]
        if i < 4:
            args += ['-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p']
        if i != 3: args += ['-c:a','aac','-b:a','192k']
        else: args += ['-an']
        args += ['-movflags','+faststart',str(dest/name.replace('.','.pending.',1))]
    media.run(args)
    for name in FILES[:6]: (dest/name.replace('.','.pending.',1)).replace(dest/name)
    media.run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(min(1,c['duration']/2)),
               '-i',str(dest/'faces.mp4'),'-frames:v','1',str(dest/'poster.jpg')])
    c['media_source'] = str(OLD.relative_to(ROOT))
    save(cached,c)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--plan-only',action='store_true'); args=ap.parse_args()
    manifest=json.loads((OLD/'manifest.json').read_text())
    selection=json.loads((ROOT/'data_previews/seamless_face_listening_v1/selection.json').read_text())
    records={r['id']:r for p in selection['attempts'] for r in p['records']}
    people={key:prepare(record,0) for key,record in records.items()}
    audit=[]; clips=[]
    for pair in manifest['pairs']:
        revised=[]
        for old in pair['clips']:
            c=revise(old,people); revised.append(c); clips.append(c)
            for e in old['listener_events']:
                overlap=max(0,min(c['end'],e['end'])-max(c['start'],e['start']))
                if overlap < e['end']-e['start']-.001:
                    audit.append(dict(path=c['path'],event=e,removed=overlap<=.001,
                                      retained_duration=round(overlap,6)))
        pair['clips']=revised; pair['segmentation_revision']=4; pair['floor_turns_retained_from_revision']=2
    stats=dict(pairs=len(manifest['pairs']),clips=len(clips),
               head_changed=sum(c['head_removed_seconds']>.001 for c in clips),
               tail_changed=sum(c['tail_removed_seconds']>.001 for c in clips),
               head_removed_seconds=round(sum(c['head_removed_seconds'] for c in clips),3),
               tail_removed_seconds=round(sum(c['tail_removed_seconds'] for c in clips),3),
               duration_seconds=round(sum(c['duration'] for c in clips),3),
               shorter_than_four_seconds=sum(c['duration']<4 for c in clips),
               kinds=dict(collections.Counter(c['kind'] for c in clips)))
    manifest.update(segmentation_revision=4,parent_segmentation_revision=2,
                    method='rules_only_speaker_word_span',boundary_policy=POLICY,trim_summary=stats)
    manifest['rules']=dict(manifest['rules'],export_boundary_policy=POLICY,
                           overlapping_listener_vad='retain inside speaker span')
    save(OUT/'planned_manifest.json',manifest); save(OUT/'trim_summary.json',stats)
    save(OUT/'boundary_event_audit.json',audit)
    print('PLANNED',json.dumps(stats),flush=True)
    if args.plan_only: return
    for pair in manifest['pairs']: link(OLD/pair['conversation_path'],OUT/pair['conversation_path'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(render,clips),1):
            if i%20==0: print('DONE',i,'/',len(clips),flush=True)
    for pair in manifest['pairs']: save(OUT/'media'/pair['pair']/'pair.json',pair)
    save(OUT/'manifest.json',manifest)
    print('FINISHED',flush=True)


if __name__=='__main__': main()
