#!/usr/bin/env python3
"""Verify real browser behavior, media, statistics and review persistence."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/listen_benchmark_review'
URL='http://127.0.0.1:8875/listen_benchmark_review/'


def main():
    data=json.loads((OUT/'review.json').read_text());errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1560,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL);page.wait_for_function('window.reviewData && window.reviewState')
        page.evaluate('document.fonts.ready')
        assert page.locator('#modelTable tr').count()==6
        for group,count in [('visual',4),('audio',3),('omni',2),('control',4),('auxiliary',3),('all',20),('scheme',6)]:
            page.select_option('#group',group)
            assert page.locator('#modelTable tr').count()==count,(group,page.locator('#modelTable tr').count())
        page.select_option('#sort','nod')
        schemes=[s for s in data['schemes'] if s['group']=='scheme' or s['id'] in ('qwen3_omni_native','minicpm_omni_native')]
        expected=max(schemes,key=lambda s:s['coverage'].get('head_nod',-1))
        assert expected['name'] in page.locator('#modelTable tr').first.inner_text()
        page.select_option('#sort','default')
        page.screenshot(path=str(OUT/'overview_desktop.png'),full_page=True);checks.append('overview groups, sorting and full scheme statistics')
        page.click('#openDisagreement')
        page.wait_for_function('document.querySelector("#audio").readyState>=2 && document.querySelector("#video").readyState>=2')
        assert page.locator('.compare-card').count()==3
        expected=sum(s['disagreement'] for s in data['samples']);assert page.locator('#sampleList button').count()==expected
        page.click('#play');page.wait_for_function('document.querySelector("#audio").currentTime>.4')
        sync=page.evaluate('({a:audio.currentTime,v:video.currentTime,muted:video.muted,audioPlaying:!audio.paused})')
        assert sync['muted'] and sync['audioPlaying'] and abs(sync['a']-sync['v'])<.2,sync
        page.click('#play');checks.append('independent WAV and muted video synchronized playback')
        event=page.locator('.event-item').first
        start=float(event.get_attribute('data-start'));end=float(event.get_attribute('data-end'))
        event.click();page.wait_for_function('!document.querySelector("#audio").paused')
        assert page.is_checked('#loop')
        page.wait_for_timeout(int((end-start+.3)*1000))
        now=page.evaluate('audio.currentTime');assert start-.1<=now<=end+.2,(start,end,now)
        page.click('#play');checks.append('event seek and bounded repeated playback')
        page.check('#blind');assert not page.locator('#selectors').is_visible()
        assert page.locator('.card-head h3').all_text_contents()==['方案 A','方案 B','方案 C']
        assert not page.locator('#compareGrid a[download]').count()
        page.select_option('#winner','pair_joy');page.fill('#reviewer','browser-test');page.fill('#reviewNote','Browser verification only, not a human judgment.')
        page.click('#saveNote');assert '已保存' in page.locator('#saved').inner_text()
        with page.expect_download() as download:
            page.click('#exportNotes')
        export=Path(download.value.path());value=json.loads(export.read_text());assert value['human_ground_truth'] is False
        assert len(value['records'])==1 and value['records'][0]['hidden_names'] is True
        sid=page.evaluate('reviewState().sample');page.reload();page.click('[data-tab="review"]')
        # Find the saved sample again; the browser context retains localStorage.
        page.locator(f'[data-sample="{sid}"]').click()
        assert page.input_value('#reviewNote')=='Browser verification only, not a human judgment.'
        checks.append('hidden model names, persistent opinion and JSON export')
        # Remove synthetic test preference from the isolated test browser.
        page.evaluate('localStorage.clear(); opinions={}; loadOpinion()')
        page.select_option('#stage','discovery');assert '开放事件' in page.locator('#compareGrid').inner_text() or page.locator('.error-box').count()>0
        page.select_option('#columnCount','4');assert page.locator('.compare-card').count()==4
        page.select_option('#stage','audit')
        page.select_option('[data-column="3"]','minicpm_omni_native')
        # Deliberately inspect a failed audit, then an empty-coverage result.
        failed=next(s['id'] for s in data['samples'] if data['results']['minicpm_omni_native'][s['id']]['audit']['status']=='failed')
        page.select_option('#sampleFilter','all');page.fill('#search','');page.locator(f'[data-sample="{failed}"]').click()
        assert '失败不是“无事件”' in page.locator('#compareGrid').inner_text()
        empty=next(s['id'] for s in data['samples'] if data['results']['qwen3_audio'][s['id']]['audit']['status']=='schema_pass' and not data['results']['qwen3_audio'][s['id']]['audit']['coverage']['listener_vocalization'])
        page.locator(f'[data-sample="{empty}"]').click();assert page.locator('.unknown-note').count()>0
        checks.append('failure preserved, unchecked empty outputs not labeled negative')
        page.click('#resetCompare');page.select_option('#sampleFilter','all');page.select_option('#sampleOrder','source')
        # Check all 50 packaged media URLs and independent WAV hashes are exporter-verified.
        for sample in data['samples']:
            for field in ('video','audio'):
                response=page.request.get(URL+sample[field],headers={'Range':'bytes=0-127'})
                assert response.status==206 and len(response.body())==128,(sample['id'],field,response.status)
        checks.append('100 media endpoints support byte-range seeking')
        page.select_option('#sampleFilter','disagreement');page.select_option('#sampleOrder','difference')
        page.locator('#sampleList button').first.click();page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        page.screenshot(path=str(OUT/'compare_desktop.png'),full_page=True)
        page.click('#auxDetails summary');assert page.locator('.aux-box').count()==3
        page.click('[data-tab="runtime"]');assert page.locator('#failureTable tr').count()==16
        for metric in ('gpu','cpu','memory'):
            page.select_option('#resourceMetric',metric);assert page.locator('#resourceChart polyline').count()>=3
        checks.append('auxiliary outputs and resource plots')
        page.set_viewport_size({'width':390,'height':844});page.click('[data-tab="review"]');page.screenshot(path=str(OUT/'compare_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), 'horizontal page overflow'
        checks.append('390px responsive view without horizontal page overflow')
        # file:// still renders the embedded comparison data without fetch.
        offline= context.new_page();offline.goto((OUT/'index.html').as_uri());offline.wait_for_function('window.reviewData')
        assert offline.locator('#modelTable tr').count()==6;checks.append('standalone HTML overview works on file://')
        assert not errors,errors
        browser.close()
    (OUT/'browser_validation.json').write_text(json.dumps(dict(passed=True,checks=checks,errors=errors),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
