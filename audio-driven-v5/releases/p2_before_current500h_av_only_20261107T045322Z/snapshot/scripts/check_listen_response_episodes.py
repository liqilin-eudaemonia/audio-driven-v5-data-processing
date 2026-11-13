"""Verify provenance accounting, independent gestures and the response review page."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return {s['sample_id']: s for s in (json.loads(line) for line in path.read_text().splitlines())}


def check(run, url):
    manifest = json.loads((run / 'manifest.json').read_text())
    source = Path(manifest['source_run'])
    for name, sha in manifest['input_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == sha
    assert (run / 'observations.jsonl').read_bytes() == (source / 'labels.jsonl').read_bytes()
    original, labels = rows(run / 'observations.jsonl'), rows(run / 'labels.jsonl')
    traces = json.loads((run / 'provenance.json').read_text())['samples']
    assert original.keys() == labels.keys() == traces.keys()
    words = 0
    for sid, sample in labels.items():
        before = {e['event_id']: e for e in original[sid]['events']}
        accounted = []
        for e in sample['events']:
            if e.get('event_unit') != 'vocal_response_episode':
                assert e == before[e['event_id']]
                accounted.append(e['event_id'])
                continue
            trace = traces[sid][e['event_id']]
            assert trace['member_event_ids'] == e['member_event_ids']
            assert trace['observations'] == [before[i] for i in e['member_event_ids']]
            assert all(x['categories'] == e['categories'] for x in trace['observations'])
            support = [before[i] for i in trace['boundary_support_event_ids']]
            assert e['start'] == min(x['start'] for x in support)
            assert e['end'] == max(x['end'] for x in support)
            assert e['behavior'] in [x['behavior'] for x in support]
            assert e['repetition_count'] is None
            assert e['word_observations'] == [{k: x[k] for k in ('event_id', 'start', 'end', 'text')}
                for x in trace['observations'] if x.get('event_unit') == 'word']
            accounted.extend(e['member_event_ids'])
        for unresolved in sample['unresolved_observations']:
            e = unresolved['observation']
            assert e == before[e['event_id']]
            accounted.append(e['event_id'])
        assert Counter(accounted) == Counter(before.keys())
        words += sum(e.get('event_unit') == 'word' for e in before.values())
    example_id = 'naturalistic_V00_S0805_I00000126_local_f7c1fb6d13ca'
    example = labels[example_id]
    episode = next(e for e in example['events'] if 'event-0028' in e.get('member_event_ids', []))
    assert set(['event-0028', 'event-0029', 'event-0030', 'event-0031']) <= set(episode['member_event_ids'])
    assert len(episode['member_event_ids']) == 14 and (episode['start'], episode['end']) == (3.32, 4.96)
    okay_sample = next(s for sid, s in labels.items() if 'S0699' in sid)
    assert [(e['start'], e['end']) for e in okay_sample['events'] if e.get('text') == 'Okay'] == [(1.78, 2.1), (4.08, 4.4)]
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1450, 'height': 1100}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        assert page.goto(url + '?sample=' + example_id).status == 200
        page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        assert page.locator('#sid').inner_text() == example_id
        assert page.locator('#samples option').count() == 50
        page.select_option('#category', 'listener_vocalization')
        assert page.locator('#mapped .event').count() == 6
        page.locator(f'#mapped [data-id="{episode["source_open_event_id"]}"]').click()
        page.wait_for_timeout(700)
        state = page.evaluate('({loop:sevenClassState().loop,v:video.currentTime,a:audio.currentTime,paused:video.paused})')
        assert state['loop'] == [3.32, 4.96] and not state['paused'], state
        assert abs(state['v'] - state['a']) < .2
        assert page.locator('#original .selected').count() == 14
        assert page.locator('#mapped .selected').count() == 1
        # Clicking any member still highlights the full response group.
        page.locator('#original [data-id="event-0029"]').click()
        assert page.locator('#original .selected').count() == 14
        assert page.locator('#mapped .selected').count() == 1
        page.select_option('#status', 'needs_review')
        assert page.locator('#mapped .event').count() == 0
        page.select_option('#status', 'all')
        for name in ['labels.jsonl', 'observations.jsonl']:
            with page.expect_download() as result:
                page.locator(f'a[download][href="{name}"]').click()
            assert Path(result.value.path()).read_bytes() == (run / name).read_bytes()
        assert context.request.get(page.evaluate('video.src'), headers={'Range': 'bytes=0-1023'}).status == 206
        for name in ['provenance.json', 'source_provenance.json', 'mapping_policy.json', 'summary.json']:
            assert context.request.get(url + name).status == 200
        page.select_option('#category', 'all')
        page.locator(f'#mapped [data-id="{episode["source_open_event_id"]}"]').click()
        page.evaluate('video.pause();window.scrollTo(0,0)')
        page.screenshot(path=str(run / 'review_desktop.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        browser.close()
    report = dict(samples=len(labels), original_words_preserved=words,
        example=dict(sample_id=example_id, member_observations=14, response_episodes=1, start=3.32, end=4.96),
        checks=['Source bytes unchanged; every observation accounted for exactly once',
                'Visual events and semantic categories unchanged; original descriptions and word evidence retained',
                'Separated okay responses stay separate; reported four yeah records belong to one continuous group',
                'Browser group highlighting, synchronized playback, downloads, media seeking and mobile layout pass'],
        errors=[])
    (run / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8876/listen_benchmark_review/response_episodes/')
    args = parser.parse_args()
    check(Path(args.run), args.url.rstrip('/') + '/')
