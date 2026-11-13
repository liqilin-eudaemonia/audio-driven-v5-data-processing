"""Validate the live sample page, browser playback, seeks, and review export."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / 'data_previews/seamless_listen_v7_sample'
errors = []
checked = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8875/listen_qc3/', wait_until='domcontentloaded')
    page.wait_for_url('**/listen_v7/')
    page.wait_for_function('current !== null && samples.length === 24')
    assert page.locator('#list button').count() == 24
    # Every published clip must load, play and seek in the actual browser.
    for i in range(24):
        page.evaluate('(i)=>choose(samples[i])', i)
        page.wait_for_function('v.readyState >= 2 && !v.error')
        page.evaluate('v.play()')
        page.wait_for_function('v.currentTime > .15')
        page.evaluate('v.currentTime = Math.max(0, v.duration - .6)')
        page.wait_for_function('!v.seeking && v.currentTime >= v.duration - .8')
        checked.append(page.evaluate('current.sample_id'))
    for mode in ('listener', 'speaker', 'both'):
        page.locator('#channel').select_option(mode)
        page.locator('#restart').click()
        page.wait_for_function('v.readyState >= 2 && v.currentTime > .15 && !v.paused && !v.error')
    page.locator('#filter').select_option('voice')
    assert page.locator('#list button').count() == 12
    page.locator('#filter').select_option('quiet')
    assert page.locator('#list button').count() == 12
    page.locator('#filter').select_option('all')
    page.locator('#decision').select_option('keep')
    page.locator('#note').fill('Browser verification only')
    with page.expect_download() as download:
        page.locator('#export').click()
    payload = json.loads(Path(download.value.path()).read_text())
    assert len(payload['notes']) == 1
    assert next(iter(payload['notes'].values()))['note'] == 'Browser verification only'
    page.locator('#note').fill('')
    page.locator('#decision').select_option('')
    media_url = page.evaluate('v.src')
    response = page.request.get(media_url, headers={'Range': 'bytes=0-127'})
    assert response.status == 206 and len(response.body()) == 128
    page.screenshot(path=str(OUT / 'browser_desktop.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(OUT / 'browser_mobile.png'), full_page=True)
    assert not errors, errors
    browser.close()
result = dict(samples_played_and_seeked=checked, page_errors=errors, channels_played=3,
              old_url_redirect=True, range_206=True, review_export=True, mobile_overflow=False)
(OUT / 'browser_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
