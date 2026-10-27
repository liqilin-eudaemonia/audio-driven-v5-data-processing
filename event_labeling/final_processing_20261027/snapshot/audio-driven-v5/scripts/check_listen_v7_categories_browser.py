"""Check every category sample and its evidence controls on the live page."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / 'data_previews/seamless_listen_v7_sample'
data = json.loads((OUT / 'samples_categories.json').read_text())
errors = []
checked = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8875/listen_qc3/', wait_until='domcontentloaded')
    page.wait_for_url('**/listen_v7/')
    page.wait_for_function('current !== null && samples.length === 80')
    assert page.locator('#filter').input_value() == 'review_primary'
    for category, info in data['categories'].items():
        page.locator('#filter').select_option(category)
        assert page.locator('#list button').count() == info['sample_count']
        assert page.evaluate('current.review_category') == category
        assert page.locator('#reason').inner_text()
    page.locator('#filter').select_option('all')
    for i in range(80):
        page.evaluate('(i)=>choose(samples[i])', i)
        page.wait_for_function('v.readyState >= 2 && !v.error')
        page.evaluate('v.play()')
        page.wait_for_function('v.currentTime > .12')
        page.evaluate('v.currentTime = Math.max(0, v.duration - .5)')
        page.wait_for_function('!v.seeking && v.currentTime >= v.duration - .7')
        checked.append(page.evaluate('current.sample_id'))
    for mode in ('listener', 'speaker', 'both'):
        page.locator('#channel').select_option(mode)
        page.locator('#restart').click()
        page.wait_for_function('v.readyState >= 2 && v.currentTime > .12 && !v.paused && !v.error')
    for option, has_voice in [('voice', True), ('quiet', False)]:
        page.locator('#vocalFilter').select_option(option)
        assert page.locator('#list button').count() == sum(bool(r['listener_human_evidence']) == has_voice for r in data['samples'])
    page.locator('#vocalFilter').select_option('all')
    page.locator('#filter').select_option('review_primary')
    page.evaluate("choose(samples.find(r=>r.review_evidence.tracks.listener.flagged_intervals.length))")
    page.locator('#flags button').first.click()
    page.wait_for_function('v.readyState >= 2 && !v.paused && v.currentTime > .05')
    page.locator('#decision').select_option('keep')
    page.locator('#note').fill('Browser verification only')
    with page.expect_download() as download:
        page.locator('#export').click()
    payload = json.loads(Path(download.value.path()).read_text())
    assert payload['review'] == 'v7-six-categories'
    assert next(iter(payload['notes'].values()))['original_category']
    page.locator('#note').fill(''); page.locator('#decision').select_option('')
    response = page.request.get(page.evaluate('v.src'), headers={'Range': 'bytes=0-127'})
    assert response.status == 206 and len(response.body()) == 128
    page.evaluate('v.pause()');page.evaluate('document.fonts.ready')
    page.screenshot(path=str(OUT / 'categories_desktop.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(OUT / 'categories_mobile.png'), full_page=True)
    assert not errors, errors
    browser.close()
result = dict(samples_played_and_seeked=checked, page_errors=errors,
              categories_checked=list(data['categories']), evidence_seek=True,
              old_url_redirect=True, range_206=True, review_export=True, mobile_overflow=False)
(OUT / 'categories_browser_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(samples=len(checked), page_errors=errors, categories=6, evidence_seek=True)))
