"""Verify published RGB metric scopes against their exact data values."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'data.json').read_text())
errors = []
checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',
        headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1080})
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:8876/p2_av_review/')
    page.wait_for_function('window.reviewReady === true')
    for dataset in data['datasets']:
        assert dataset['detector_version'] == 'rgb_v2'
        page.locator(f'[data-dataset={dataset["id"]}]').click()
        assert page.locator('#metric-scope').is_enabled()
        for scope in ['paired', 'within']:
            page.locator('#metric-scope').select_option(scope)
            rows = page.locator('#metric-rows tr')
            assert rows.count() == 8
            assert 'RGB修正版' in page.locator('#metric-notes').inner_text()
            assert '进行中' not in rows.all_inner_texts().__str__()
            for i, row in enumerate(dataset['metrics']):
                cells = rows.nth(i).locator('td').all_text_contents()
                assert cells[1] == '48', cells
                values = row['values' if scope == 'paired' else 'within_values']
                for j, key in enumerate(['FVD-32', 'UTMOSv2', 'LSE-D', 'LSE-C', 'R-F1', 'R-tIoU', 'R-ATD', 'R-FID']):
                    value = values[key]
                    if value is None:
                        assert cells[j + 2] == '—'
                    else:
                        precision = 4 if key == 'R-F1' else 3
                        assert abs(float(cells[j + 2]) - value) <= 0.5001 * 10 ** -precision
            checks.append(dict(dataset=dataset['id'], scope=scope, rows=8, outputs_per_row=48))
        assert all(o['metric_status'] != 'not_evaluated' for s in dataset['samples'] for o in s['outputs'])
    page.locator('#metric-scope').select_option('paired')
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(OUT / 'rgb_metrics_preview.png'))
    browser.close()
assert not errors, errors
result = dict(status='passed', checks=checks, page_errors=errors)
(OUT / 'browser_rgb_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
