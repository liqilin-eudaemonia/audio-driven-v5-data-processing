"""Audit paired native video experiment; lexical metrics are not accuracy."""
import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from run import ARMS, RUN, ROOT, observation_prompt, views, write, validate_timeline

EXPLICIT = re.compile(r'\b(indicat\w*|suggest\w*|impl[yi]\w*|signall?ing|convey\w*|as if|perhaps|possibly)\b', re.I)
ACTIONS = {
    'nod': r'\bnod\w*', 'smile': r'\bsmil\w*', 'blink': r'\b(blink\w*|eyes? (?:briefly )?clos\w*)',
    'laugh': r'\b(laugh\w*|chuckl\w*|giggl\w*)', 'hand': r'\b(hand\w*|arm\w*)',
    'brow': r'\b(eyebrow\w*|brow\w*|frown\w*)', 'gaze': r'\b(gaze|look\w*)',
    'head': r'\b(head|nod\w*|tilt\w*|shak\w*)', 'mouth': r'\b(mouth|lip\w*)',
}
NEGATION = re.compile(r'\b(no|not|without|doesn.t|isn.t|never|neither)\b', re.I)


def main():
    samples = json.load(open(RUN / 'samples.json'))
    expected = {s['sample_id'] for s in samples}
    files = sorted((RUN / 'results').glob('*.json'))
    stats = {a: collections.Counter() for a in ARMS}
    full = {a: collections.Counter() for a in ARMS}
    durations = {a: [] for a in ARMS}
    rows, errors, mentions, descriptions = [], [], [], {a: [] for a in ARMS}
    seen = set()
    fingerprints = {}
    max_error = 0.
    for file in files:
        d = json.load(open(file))
        sid = d['sample_id']
        assert sid in expected and sid not in seen
        seen.add(sid)
        by_view = collections.defaultdict(dict)
        fingerprints[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        for r in d['records']:
            a, view = r['arm'], r['view']
            assert a in ARMS and a not in by_view[view]
            by_view[view][a] = r
            assert r['prompt'] == observation_prompt(r['end']-r['start'])
            c = collections.Counter(requests=1, valid_requests=int(r['status']=='valid'),
                truncated=int(r['finish_reason']!='stop'), completion_tokens=r['completion_tokens'], prompt_tokens=r['prompt_tokens'])
            if r['status'] != 'valid':
                errors.append(dict(sample_id=sid, arm=a, view=view, error=r.get('error')))
            else:
                validate_timeline(r['parsed'], r['end']-r['start'])
                assert json.loads(r['raw']) == r['parsed']
                events = r['parsed']['events']
                texts = [e['behavior'] for e in events]
                c.update(valid_events=len(events), nonempty_requests=int(bool(events)),
                    requests_with_explicit_markers=int(any(EXPLICIT.search(t) for t in texts)),
                    exact_repeated_descriptions_within_request=len(texts)-len({t.casefold().strip() for t in texts}),
                    description_chars=sum(map(len, texts)), single_event_requests=int(len(events)==1))
                for e in events:
                    text = e['behavior']
                    length = e['end']-e['start']
                    durations[a].append(length)
                    c['shorter_than_0_2s'] += length < .2-1e-7
                    c['interval_over_80pct_window'] += length >= .8*(r['end']-r['start'])
                    c['explicit_marker_descriptions'] += bool(EXPLICIT.search(text))
                    c['descriptions_with_negation'] += bool(NEGATION.search(text))
                    for key, regex in ACTIONS.items():
                        if re.search(regex, text, re.I):
                            c['action_'+key] += 1
                            if key == 'nod':
                                mentions.append(dict(sample_id=sid, arm=a, view=view, start=e['start']+r['start'], end=e['end']+r['start'], behavior=text, has_negation_word=bool(NEGATION.search(text))))
                    descriptions[a].append(dict(sample_id=sid, view=view, start=e['start']+r['start'], end=e['end']+r['start'], behavior=text))
            stats[a].update(c)
            if view == 'full':
                full[a].update(c)
        for view, start, end in views(d['duration_s']):
            rr = by_view[view]
            assert set(rr) == set(ARMS)
            assert rr['image8']['media'] == rr['video8']['media']
            for a, r in rr.items():
                assert r['start'] == start and r['end'] == end
                if a != 'image8':
                    nat = r['native']
                    assert nat['do_resize'] is False and nat['do_sample_frames'] is False
                    frames = r['media']['video']['frames']
                    indices = nat['metadata']['frames_indices']
                    expected_indices = [round(f['time_s'] * nat['metadata']['fps']) for f in frames]
                    # HF's processor pads the shared metadata index list in-place
                    # for an odd number of video frames, matching temporal_patch_size=2.
                    assert indices == expected_indices or (len(expected_indices) % 2 == 1 and indices == expected_indices + expected_indices[-1:])
                    assert nat['final_frame_padding_count'] == len(frames) % 2
                    max_error = max(max_error, nat['max_abs_timestamp_rounding_error_s'])
            sets = {a: {x['jpeg_sha256'] for x in r['media']['video']['frames']} for a,r in rr.items()}
            assert sets['image8'] == sets['video8'] and sets['video8'] <= sets['video16']
            if view == 'full':
                rows.append(dict(sample_id=sid, duration=d['duration_s'], arms={a:dict(status=r['status'],events=r.get('parsed',{}).get('events',[])) for a,r in rr.items()}))
        assert set(by_view) == {v[0] for v in views(d['duration_s'])}
    report = dict(completed_samples=len(files), planned_samples=50, all_samples_complete=seen==expected,
        stats={a:dict(c,median_event_duration_s=statistics.median(durations[a]) if durations[a] else None) for a,c in stats.items()},
        full_view_stats={a:dict(c) for a,c in full.items()}, errors=errors,
        max_native_timestamp_rounding_error_s=max_error, full_comparisons=rows,
        human_ground_truth=False, lexical_metrics_are_not_accuracy=True,
        native_input_frames_verified=True, negative_mentions_not_positive_detections=True)
    write(RUN / 'analysis.json', report)
    write(RUN / 'nod_mentions.json', mentions)
    for a, es in descriptions.items():
        write(RUN / f'descriptions_{a}.json', es)
    if seen == expected:
        assert all(s['requests'] == 293 for s in stats.values())
        write(RUN / 'COMPLETE.json', dict(completed_samples=50, completed_requests=879, source_sha256=fingerprints,
            human_ground_truth=False, production_modified=False, status='completed_with_recorded_invalid_responses'))
    print(json.dumps({k:v for k,v in report.items() if k not in ('full_comparisons','errors')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
