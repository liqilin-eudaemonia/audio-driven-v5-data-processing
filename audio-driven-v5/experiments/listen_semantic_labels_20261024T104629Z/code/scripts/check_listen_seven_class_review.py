"""Check frozen source preservation, seven-class accounting, and browser playback."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8876/listen_benchmark_review/seven_class/')
    a = p.parse_args();run = Path(a.run);url = a.url.rstrip('/') + '/'
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, sha in manifest['input_sha256'].items():
        assert hashlib.sha256((run / 'input' / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((Path(manifest['source_run']) / name).read_bytes()).hexdigest() == sha
    data = json.loads((run / 'review.json').read_text())
    full_labels = {s['sample_id']:s for s in (json.loads(line) for line in (run/'labels.jsonl').read_text().splitlines())}
    words = 0
    for s in data['samples']:
        decisions = {d['open_event_id']: d for d in s['decisions']}
        assert len(decisions) == len(s['decisions'])
        assert len(s['events']) == sum(d['status'] == 'mapped_candidate' for d in decisions.values())
        for e in s['events']:
            original = decisions[e['source_open_event_id']]['open_event']
            assert (e['start'], e['end'], e.get('text')) == (original['start'], original['end'], original.get('text'))
            types=e.get('categories',[e.get('type')])
            assert ('listener_vocalization' in types) == (e['modality'] == 'audio')
        if data['policy'].get('semantic'):
            full=full_labels[s['id']]['events']
            assert len(full)==len(decisions)
            for e in full:
                original=decisions[e['event_id']]['open_event']
                assert all(e[k]==v for k,v in original.items())
                words+=e['event_unit']=='word'
        else:
            words+=sum(e['event_unit']=='word' for e in s['events'])
    checks = ['All input snapshots and original source hashes unchanged',
              'Every open event accounted for; mapped times, text and modalities preserved']
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1450, 'height': 1100}, accept_downloads=True)
        page = context.new_page();errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        assert page.goto(url).status == 200
        page.wait_for_function('window.sevenClassReview && window.sevenClassState')
        assert page.locator('#samples option').count() == len(data['samples'])
        assert page.locator('#classCounts tr').count() == 7
        sample = next(s for s in data['samples'] if 'S0699' in s['id'])
        page.select_option('#samples', sample['id'])
        page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        okay = next(e for e in sample['events'] if e.get('text') == 'Okay' and e['start'] > 4)
        page.locator(f'#mapped [data-id="{okay["source_open_event_id"]}"]').click()
        page.wait_for_timeout(800)
        state = page.evaluate('({loop:sevenClassState().loop,v:video.currentTime,a:audio.currentTime,paused:video.paused})')
        assert state['loop'] == [okay['start'], okay['end']] and not state['paused'], state
        assert abs(state['v'] - state['a']) < .2
        page.select_option('#category', 'listener_vocalization')
        expected = sum('listener_vocalization' in e.get('categories',[e.get('type')]) for e in sample['events'])
        assert page.locator('#mapped .event').count() == expected
        page.select_option('#status', 'needs_review')
        assert page.locator('#mapped .event').count() == 0
        page.select_option('#category', 'all');page.select_option('#status', 'all')
        with page.expect_download() as result:
            page.locator('a[download][href="labels.jsonl"]').click()
        downloaded = Path(result.value.path()).read_bytes()
        assert downloaded == (run / 'labels.jsonl').read_bytes()
        assert context.request.get(page.evaluate('video.src'), headers={'Range': 'bytes=0-1023'}).status == 206
        for name in ('provenance.json', 'summary.json', 'mapping_policy.json'):
            assert context.request.get(url + name).status == 200
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(run / 'review_desktop.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        browser.close()
    checks += ['Seven-class counts and class/status filters pass',
               'Repeated okay keeps own interval; synchronized playback and media ranges pass',
               'JSONL download matches exported bytes; mobile layout and JavaScript checks pass']
    report = dict(samples=len(data['samples']), preserved_asr_words=words, checks=checks, errors=[])
    (run / 'browser_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
