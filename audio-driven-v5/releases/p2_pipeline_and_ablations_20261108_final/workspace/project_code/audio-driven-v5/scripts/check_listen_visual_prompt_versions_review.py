#!/usr/bin/env python3
"""Browser checks against the existing local review server; synthetic feedback only."""
import json
import os
from pathlib import Path

# Reuse already installed browser libraries in this workspace, without changing the host.
lib = Path('/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu')
if lib.exists():
    os.environ['LD_LIBRARY_PATH'] = str(lib) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
if Path('/tmp/p2_review_fonts.conf').exists():
    os.environ['FONTCONFIG_FILE'] = '/tmp/p2_review_fonts.conf'
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_previews/listen_benchmark_review/prompt_versions_20261029'
URL = 'http://127.0.0.1:8876/listen_benchmark_review/prompt_versions_20261029/'


def main():
    data = json.loads((OUT / 'data.json').read_text())
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1560, 'height': 1050}, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(URL)
        page.wait_for_function('window.visualPromptReview && !document.querySelector("#app").hidden')
        assert page.locator('#sampleSelect option').count() == 50
        assert page.locator('.card').count() == 4
        assert page.locator('#stats tr').count() == len(data['series'])
        page.wait_for_function('video.readyState>=2')
        checks.append('50 samples, four default versions, six independently identified series')
        for mode, keys in {'r1': ['r1:baseline', 'r1:observable'], 'r2': ['r2:baseline', 'r2:observation_v1'], 'r3': ['r3:observation_v1', 'r3:observed_actions_v2']}.items():
            page.locator('#mode').select_option(mode)
            assert page.locator('.card').evaluate_all('(es)=>es.map(e=>e.dataset.series)') == keys
        # Every rendered description for all 50 full clips must equal the source text.
        page.locator('#mode').select_option('four')
        default = ['r1:baseline', 'r1:observable', 'r2:observation_v1', 'r3:observed_actions_v2']
        for s in data['samples']:
            page.locator('#sampleSelect').select_option(s['id'])
            for key in default:
                r = next(x for x in s['series'][key] if x['view'] == 'full')
                expected = [e['behavior'] for e in r['parsed']['events']] if r['status'] == 'valid' else []
                actual = page.locator(f'[data-series="{key}"] .behavior').all_text_contents()
                assert actual == expected, (s['id'], key)
        checks.append('All 50 full-clip outputs preserve original descriptions and valid/invalid state')
        if data.get('input_comparison'):
            page.locator('#mode').select_option('inputs')
            assert page.locator('.card').evaluate_all('(es)=>es.map(e=>e.dataset.series)') == ['r4:image8','r4:video8','r4:video16']
            for s in data['samples']:
                page.locator('#sampleSelect').select_option(s['id'])
                for key in ('r4:image8','r4:video8','r4:video16'):
                    r = next(r for r in s['series'][key] if r['view'] == 'full')
                    expected = [e['behavior'] for e in r['parsed']['events']] if r['status']=='valid' else []
                    assert page.locator(f'[data-series="{key}"] .behavior').all_text_contents() == expected
            assert '相同 C prompt' in page.locator('#focusNote').inner_text()
            checks.append('All 50 paired image8/video8/video16 full outputs preserve original descriptions and input-mode provenance')
        s = next(s for s in data['samples'] if s['short'] == 'S1103')
        page.locator('#sampleSelect').select_option(s['id'])
        page.locator('#mode').select_option('r3')
        r = next(r for r in s['series']['r3:observed_actions_v2'] if r['start'] > 0 and r['status'] == 'valid' and r['parsed']['events'])
        page.locator('#view').select_option(r['view'])
        event = r['parsed']['events'][0]
        btn = page.locator('[data-series="r3:observed_actions_v2"] .seek').first
        assert abs(float(btn.get_attribute('data-a')) - (r['start'] + event['start'])) < 1e-8
        page.wait_for_function('video.readyState>=2')
        btn.click()
        page.wait_for_timeout(1000)
        state = page.evaluate('({t:video.currentTime,paused:video.paused,muted:video.muted,range:visualPromptState().chosen})')
        assert not state['paused'] and state['muted'] and state['range']
        assert state['range'][0] - .1 <= state['t'] <= state['range'][1] + .15, state
        page.locator('#speed').select_option('0.5')
        assert page.evaluate('video.playbackRate') == .5
        page.locator('#fullPlay').click()
        assert page.evaluate('visualPromptState().chosen') is None
        page.evaluate('video.pause()')
        checks.append('Short-window timestamps offset to clip time; seek, repeat and half-speed playback work')
        # Invalid requests must remain inspectable, without clickable candidate events.
        invalid = next((s, key, r) for s in data['samples'] for key, rs in s['series'].items() for r in rs if r['status'] != 'valid')
        s, key, r = invalid
        page.locator('#sampleSelect').select_option(s['id'])
        page.locator('#mode').select_option(key.split(':')[0])
        page.locator('#view').select_option(r['view'])
        card = page.locator(f'[data-series="{key}"]')
        assert card.locator('.bad').count() == 1
        assert card.locator('.seek').count() == 0
        assert card.locator('pre').nth(1).text_content() == r['raw']
        checks.append('Invalid/truncated requests expose exact raw responses and are not playable labels')
        page.locator('#note').fill('SYNTHETIC CHECK — not human feedback')
        page.locator('#preference').select_option('各有优缺点')
        page.locator('#save').click()
        with page.expect_download() as dl:
            page.locator('#export').click()
        exported = json.loads(Path(dl.value.path()).read_text())
        assert exported['human_ground_truth'] is False and exported['training_admitted'] is False
        assert exported['records'][0]['sample_id'] == s['id']
        saved_mode, saved_view = key.split(':')[0], r['view']
        page.reload()
        page.wait_for_function('window.visualPromptReview')
        page.locator('#mode').select_option(saved_mode)
        page.locator('#view').select_option(saved_view)
        assert page.locator('#note').input_value().startswith('SYNTHETIC CHECK')
        checks.append('Review persistence keyed by sample, comparison round and view; JSON exports non-GT status')
        for s in data['samples']:
            response = context.request.get(URL + s['video'], headers={'Range': 'bytes=0-127'})
            assert response.status == 206, (s['id'], response.status)
            assert len(response.body()) == 128
        for path in ['', 'existing_fusion_verbatim/', 'semantic_prompt_pilot_20261027/']:
            response = context.request.get('http://127.0.0.1:8876/listen_benchmark_review/' + path)
            assert response.status == 200 and 'visual-prompt-versions-20261029' in response.text()
        checks.append('All 50 videos support HTTP 206; three historical pages retain working navigation')
        # Both participant recordings must be reachable; processed audio is generated only in memory.
        for s in data['samples']:
            for role in ('original', 'speaker'):
                response = context.request.get(URL + s['audio'][role], headers={'Range': 'bytes=0-127'})
                assert response.status == 206 and len(response.body()) == 128, (s['id'], role)
        hashes = page.evaluate('''async () => {
          const result=[];
          for (const s of visualPromptReview.samples) {
            const pcm=await reviewProcessedPCM(s);
            const digest=await crypto.subtle.digest('SHA-256',pcm.buffer);
            result.push({id:s.id,hash:Array.from(new Uint8Array(digest)).map(x=>x.toString(16).padStart(2,'0')).join('')});
          }
          return result;
        }''')
        for result, s in zip(hashes, data['samples']):
            assert result['id'] == s['id'] and result['hash'] == s['audio']['processed_pcm_sha256'], result
        checks.append('100 participant WAV endpoints support ranges; all 50 browser-processed PCM hashes exactly match training reader')
        voiced = next(s for s in data['samples'] if s['audio']['processed_rms'] > .001)
        page.locator('#sampleSelect').select_option(voiced['id'])
        page.wait_for_function('reviewAudioState().ready && video.readyState>=2')
        page.locator('#audioMode').select_option('processed')
        page.wait_for_function('reviewAudioState().ready && listenerAudio.readyState>=2')
        page.locator('#fullPlay').click()
        page.wait_for_timeout(700)
        state = page.evaluate('({a:reviewAudioState(),v:video.currentTime,muted:listenerAudio.muted,volume:listenerAudio.volume})')
        assert not state['a']['listenerPaused'] and not state['muted'] and state['volume'] > 0, state
        assert abs(state['a']['listenerTime'] - state['v']) < .15, state
        page.locator('#audioMode').select_option('dialogue')
        page.wait_for_function('reviewAudioState().ready && speakerAudio.readyState>=2')
        page.wait_for_timeout(350)
        state = page.evaluate('({a:reviewAudioState(),v:video.currentTime})')
        assert not state['a']['listenerPaused'] and not state['a']['speakerPaused'], state
        assert max(abs(state['a']['listenerTime']-state['v']), abs(state['a']['speakerTime']-state['v'])) < .15, state
        page.evaluate('video.currentTime=2')
        page.wait_for_timeout(350)
        page.locator('#speed').select_option('0.5')
        assert page.evaluate('listenerAudio.playbackRate===.5 && speakerAudio.playbackRate===.5')
        page.evaluate('video.pause()')
        assert page.evaluate('listenerAudio.paused && speakerAudio.paused')
        page.locator('#audioMode').select_option('silent')
        assert page.evaluate('listenerAudio.paused && speakerAudio.paused')
        silent = next(s for s in data['samples'] if s['audio']['processed_full_silence'])
        page.locator('#sampleSelect').select_option(silent['id'])
        page.locator('#audioMode').select_option('processed')
        page.wait_for_function('reviewAudioState().ready')
        assert '全静音' in page.locator('#audioStatus').inner_text()
        page.locator('#audioMode').select_option('original')
        page.wait_for_function('reviewAudioState().ready')
        checks.append('Audible listener and dual-speaker modes, processing toggle, seek/rate/pause synchronization, and explicit full-silence notice')
        page.locator('#search').fill('S0404')
        assert page.locator('#sampleSelect option').count() == 1
        page.locator('#search').fill('')
        s = next(s for s in data['samples'] if s['short'] == 'S0404')
        page.locator('#sampleSelect').select_option(s['id'])
        page.locator('#mode').select_option('four')
        page.locator('#view').select_option('full')
        page.wait_for_function('video.readyState>=2')
        page.evaluate('video.currentTime=6.4')
        page.wait_for_timeout(350)
        page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)
        if data.get('input_comparison'):
            page.locator('#mode').select_option('inputs')
            page.screenshot(path=str(OUT / 'input_comparison_desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors, errors
        checks.append('Sample search/deep link, desktop and 390px mobile layout; no page JavaScript errors')
        browser.close()
    report = {'passed': True, 'checks': checks, 'page_errors': errors, 'synthetic_review_only': True}
    (OUT / 'browser_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
