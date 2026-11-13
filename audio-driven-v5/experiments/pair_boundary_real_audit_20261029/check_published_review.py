"""Browser checks against the existing 8875 review service."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=ROOT/'data_previews/seamless_listen_full_source_recovery'
BASE='http://127.0.0.1:8875/listen_full_source/'
index=json.loads((OUT/'index.json').read_text())
old=json.loads((EXP/'before_review_publish/index.json').read_text())
assert all(x in index['pairs'] for x in old['pairs'])
assert len(index['pairs'])==4
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    for entry in index['pairs']:
        pair=entry['pair'];d=json.loads((OUT/pair/'review.json').read_text())
        issues=d.get('issues',[]);target=issues[0]['focus'] if issues else 2
        page.goto(BASE+f'?pair={pair}&t={target}')
        page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.querySelector("#video").readyState>=2',arg=pair)
        page.wait_for_function('(t)=>Math.abs(document.querySelector("#video").currentTime-t)<.1',arg=target)
        assert page.locator('#source').input_value()==pair
        timeline=d['timeline']
        assert timeline[0]['start']==0 and timeline[-1]['end']==d['summary']['source_seconds']
        assert all(abs(a['end']-b['start'])<1e-6 for a,b in zip(timeline,timeline[1:]))
        for mode in ['timeline','attempts','old']:
            page.locator('#listMode').select_option(mode)
            assert page.locator('#rows tr').count()==len(d[mode])
        page.locator('#listMode').select_option('timeline')
        page.locator('#play').click()
        page.wait_for_function('(t)=>!document.querySelector("#video").paused&&document.querySelector("#video").currentTime>t+.2',arg=target)
        page.locator('#play').click()
        assert page.request.get(BASE+d['media'],headers={'Range':'bytes=0-127'}).status==206
        if issues:
            issue=issues[0]
            assert page.locator('#issues .issue').count()==1
            assert issue['title'] in page.locator('#issues').inner_text()
            page.locator('#issues [data-jump]').first.click()
            page.wait_for_function('(t)=>!document.querySelector("#video").paused&&document.querySelector("#video").currentTime>=t',arg=target)
            page.locator('#play').click()
            omitted=issue['intervals'][-1]
            page.evaluate('(t)=>seek(t)',(omitted['start']+omitted['end'])/2)
            page.wait_for_function('document.querySelector("#issueNow").textContent.includes("未进入生产片段")')
            for channel in ['both','speaker','listener','a','b']:
                page.locator('#channel').select_option(channel)
            page.locator('#issues summary').click()
            videos=page.locator('#issues .comparisons video')
            assert videos.count()==len(issue['comparisons'])
            for i,c in enumerate(issue['comparisons']):
                videos.nth(i).evaluate('(v)=>{v.load();return v.play()}')
                page.wait_for_function('(i)=>document.querySelectorAll("#issues .comparisons video")[i].currentTime>.15',arg=i)
                videos.nth(i).evaluate('(v)=>v.pause()')
                assert page.locator('#video').evaluate('(v)=>v.paused')
            assert '未重新做音质准入' in page.locator('#issues').inner_text()
            for width in [1440,390]:
                page.set_viewport_size({'width':width,'height':1000})
                page.evaluate('document.fonts.ready')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(pair,width)
                page.screenshot(path=str(EXP/f'{issue["id"]}_published_{width}.png'),full_page=True)
            page.set_viewport_size({'width':1440,'height':1100})
        else:
            assert page.locator('#issues').is_hidden()
        page.evaluate('document.querySelector("#play").click();document.querySelector("#play").click()')
        page.wait_for_timeout(150)
        assert not page.locator('#error').inner_text(), (pair,page.locator('#error').inner_text())
        checks.append(dict(pair=pair,complete_timeline=True,deep_link_seconds=target,issues=len(issues),playback=True,range_206=True))
    browser.close()
assert not errors,errors
(EXP/'published_browser_validation.json').write_text(json.dumps(dict(checks=checks,page_errors=errors,old_entries_preserved=True),indent=2)+'\n')
print(json.dumps(dict(checked_pairs=len(checks),page_errors=errors)))
