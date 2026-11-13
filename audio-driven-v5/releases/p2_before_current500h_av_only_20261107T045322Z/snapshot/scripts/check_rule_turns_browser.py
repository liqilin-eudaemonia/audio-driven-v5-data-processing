"""Exercise the published review page in a fresh browser context."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / 'data_previews/seamless_rule_turns_v2'


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', type=Path, default=OUT)
    ap.add_argument('--url', default='http://127.0.0.1:8875/rule_turns/')
    ap.add_argument('--revision', type=int, default=2)
    args = ap.parse_args()
    OUT = args.directory.resolve()
    title = {2:'Seamless · 规则话轮 V2',3:'Seamless · Speaker 结束 V3',4:'Seamless · Speaker 起止 V4',5:'Seamless · 子话轮与统一音量 V5'}[args.revision]
    storage = {2:'seamless_rule_turns_v2',3:'seamless_speaker_end_v3',4:'seamless_speaker_span_v4',5:'seamless_subturns_v5'}[args.revision]
    errors = []
    manifest = json.loads((OUT / 'manifest.json').read_text())
    expected = sum(len(p['clips']) for p in manifest['pairs'])
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until='domcontentloaded')
        page.wait_for_function('current !== null && video.readyState >= 2')
        assert page.locator('#list .item').count() == expected
        assert page.title() == title
        if args.revision == 5:
            assert page.evaluate('clips.every(c=>c.duration<=10.000001) && new Set(clips.map(c=>c.key)).size === clips.length')
        if args.revision >= 3:
            assert page.evaluate("document.querySelectorAll('#timeline rect').length === 3+current.speaker_vad.length+current.listener_vad.length+current.listener_events.filter(e=>e.label!=='untranscribed_vocalization').length")
        page.evaluate('video.play()')
        page.wait_for_function('video.currentTime > .3')
        page.locator('[data-track="listener"]').click()
        page.wait_for_function('audio.readyState >= 2 && !audio.paused')
        assert page.evaluate('video.muted && audio.src.endsWith("listener.m4a")')
        page.locator('#conversationDetails summary').click()
        page.locator('#contextEnd').click()
        page.wait_for_function('conversation.readyState >= 2 && !conversation.paused && conversation.currentTime >= current.end - 5')
        assert page.evaluate('video.paused && audio.paused')
        page.locator('#contextStart').click()
        page.wait_for_function('Math.abs(conversation.currentTime - Math.max(0,current.start-5)) < 2')
        page.locator('#contextFloor').click()
        page.wait_for_function('Math.abs(conversation.currentTime - current.parent_turn_start) < 2')
        page.locator('#parentEvents [data-event]').first.click()
        page.wait_for_function('Math.abs(conversation.currentTime - Math.max(0,current.parent_events[0].start-5)) < 2')
        page.locator('#next').click()
        assert page.evaluate('conversation.paused')
        first_pair = manifest['pairs'][0]['pair']
        page.locator('#search').fill(first_pair)
        assert page.locator('#list .item').count() == len(manifest['pairs'][0]['clips'])
        page.locator('#search').fill('')
        page.locator('#kind').select_option('quiet_candidate')
        assert page.evaluate('filtered.length > 0 && filtered.every(c=>c.kind === "quiet_candidate")')
        page.locator('#kind').select_option('')
        page.locator('#note').fill('browser validation only')
        assert page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))[current.key].note === "browser validation only"', storage)
        with page.expect_download() as download:
            page.locator('#export').click()
        payload = json.loads(Path(download.value.path()).read_text())
        assert payload['reviews'][0]['segmentation_revision'] == args.revision
        assert payload['reviews'][0]['turn_id']
        for selector in ['#download', '#downloadSpeaker', '#downloadListener']:
            url = page.locator(selector).get_attribute('href')
            response = page.request.get(args.url + url, headers={'Range': 'bytes=0-127'})
            assert response.status == 206 and len(response.body()) == 128
        page.evaluate('select(clips[0]); window.scrollTo(0,0)')
        page.locator('#note').fill('')
        page.screenshot(path=str(OUT / 'browser_desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(OUT / 'browser_mobile.png'), full_page=True)
        assert not errors, errors
        browser.close()
    result = dict(pairs=len(manifest['pairs']), clips=expected, page_errors=errors,
                  main_playback=True, listener_audio=True, context_navigation=True,
                  filters=True, review_export=True, media_byte_ranges=True, mobile_overflow=False)
    (OUT / 'browser_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
