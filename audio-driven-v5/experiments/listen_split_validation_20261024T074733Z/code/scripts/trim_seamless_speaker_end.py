"""Trim the existing 50-pair review at the speaker's last aligned word end."""
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
OUT = ROOT / 'data_previews/seamless_speaker_end_v3'
FILES = ['faces.mp4', 'speaker_512.mp4', 'listener_512.mp4', 'context.mp4',
         'speaker.m4a', 'listener.m4a', 'poster.jpg']


def link(source, destination):
    if not destination.exists(): destination.symlink_to(source)


def revise(clip, records):
    c = copy.deepcopy(clip)
    people = {role: prepare(records[c[role]], i) for i, role in enumerate(['speaker', 'listener'])}
    a, old_end = c['start'], c['end']
    words = [w for w in people['speaker']['words'] if w['start'] < old_end - .001 and w['end'] > a + .001]
    assert words, c['path']
    end = round(min(old_end, max(w['end'] for w in words)), 3)
    assert a < end <= old_end + .001
    c.update(end=end, duration=round(end-a, 3), previous_end=old_end,
             previous_end_reason=c['end_reason'], tail_removed_seconds=round(old_end-end, 3),
             segmentation_revision=3, media_revision=4,
             end_policy='speaker_last_aligned_word_end_capped_by_previous_boundary',
             speaker_word_end=max(w['end'] for w in words),
             shorter_than_previous_minimum=end-a < 4,
             entire_floor_turn=abs(a-c['parent_turn_start']) < .001 and abs(end-c['parent_turn_end']) < .001)
    if old_end-end > .001: c['end_reason'] = 'speaker_last_word_end'
    for role, person in people.items():
        c[role+'_vad'] = [[max(a,x)-a, min(end,y)-a] for x,y in person['vad'] if min(end,y)-max(a,x) > .001]
        c[role+'_vad_fraction'] = round(media.overlap(person['vad'], a, end)/(end-a), 3)
        c[role+'_text'] = ' '.join(w['text'] for w in person['words'] if min(end,w['end'])-max(a,w['start']) > .001)
        for feature, values in c['emotion'][role].items():
            c['emotion'][role][feature] = values[:math.ceil((end-a)/.2-1e-8)]
    c['listener_events'] = [dict(e, start=max(a,e['start']), end=min(end,e['end']))
                            for e in c['listener_events'] if min(end,e['end'])-max(a,e['start']) > .001]
    c['kind'] = ('feedback_candidate' if any(e['label']=='feedback_candidate' for e in c['listener_events'])
                 else 'vocal_candidate' if c['listener_events'] else 'quiet_candidate')
    return c


def render(c):
    source, dest = OLD / c['path'], OUT / c['path']
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / 'metadata.json'
    if cached.exists():
        previous = json.loads(cached.read_text())
        assert previous['end'] == c['end'], 'Changed endpoint; use a new output directory'
        return previous
    if c['tail_removed_seconds'] <= .001:
        for name in FILES: link(source/name, dest/name)
    else:
        inputs = []
        for name in FILES[:6]: inputs += ['-threads', '1', '-i', str(source/name)]
        args = ['ffmpeg', '-v', 'error', '-nostdin', '-y', *inputs]
        for i, name in enumerate(FILES[:6]):
            args += ['-map', str(i), '-t', str(c['duration'])]
            if i < 4:
                args += ['-c:v', 'libx264', '-threads', '1', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p']
            if i != 3: args += ['-c:a', 'aac', '-b:a', '192k']
            else: args += ['-an']
            args += ['-movflags', '+faststart', str(dest/name.replace('.', '.pending.', 1))]
        media.run(args)
        for name in FILES[:6]: (dest/name.replace('.', '.pending.', 1)).replace(dest/name)
        link(source/'poster.jpg', dest/'poster.jpg')
    save(cached, c)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OLD/'manifest.json').read_text())
    selection = json.loads((ROOT/'data_previews/seamless_face_listening_v1/selection.json').read_text())
    records = {r['id']: r for p in selection['attempts'] for r in p['records']}
    clips = []
    for p in manifest['pairs']:
        p['clips'] = [revise(c, records) for c in p['clips']]
        p['segmentation_revision'] = 3
        p['floor_turns_retained_from_revision'] = 2
        # Floor ownership is unchanged; exports now end earlier than those parent floors.
        dest = OUT / p['conversation_path']; dest.parent.mkdir(parents=True, exist_ok=True)
        link(OLD/p['conversation_path'], dest)
        clips.extend(p['clips'])
    manifest.update(segmentation_revision=3, method='rules_only_speaker_word_end',
                    parent_segmentation_revision=2,
                    export_end_policy='last speaker word end within previous segment; no added tail padding',
                    emotion_sampling_seconds=.2)
    manifest['rules'] = dict(manifest['rules'], export_end_policy=manifest['export_end_policy'])
    stats = dict(pairs=len(manifest['pairs']), clips=len(clips),
                 changed_clips=sum(c['tail_removed_seconds']>.001 for c in clips),
                 duration_seconds=round(sum(c['duration'] for c in clips), 3),
                 removed_seconds=round(sum(c['tail_removed_seconds'] for c in clips), 3),
                 shorter_than_four_seconds=sum(c['shorter_than_previous_minimum'] for c in clips),
                 kinds=dict(collections.Counter(c['kind'] for c in clips)))
    manifest['trim_summary'] = stats
    save(OUT/'trim_summary.json', stats)
    save(OUT/'planned_manifest.json', manifest)
    print('PLANNED', json.dumps(stats), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(render,c): c for c in clips}
        for i, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            future.result()
            if i % 20 == 0: print('DONE', i, '/', len(clips), flush=True)
    for p in manifest['pairs']: save(OUT/'media'/p['pair']/'pair.json', p)
    save(OUT/'manifest.json', manifest)
    print('FINISHED', flush=True)


if __name__ == '__main__': main()
