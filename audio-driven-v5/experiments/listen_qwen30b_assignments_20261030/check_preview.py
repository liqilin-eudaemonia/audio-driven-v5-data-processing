"""Browser and HTTP checks for this frozen review page (no training changes)."""
import json
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright

W=Path(__file__).resolve().parent
URL='http://127.0.0.1:8876/listen_benchmark_review/semantic_qwen30b_20261030/'


def main():
    data=json.load(urllib.request.urlopen(URL+'data.json',timeout=10))
    assert len(data['samples'])==73
    for s in data['samples']:
        req=urllib.request.Request(URL+s['video'],method='HEAD')
        with urllib.request.urlopen(req,timeout=10) as response:
            assert response.status==200 and int(response.headers['Content-Length'])>0
    errors=[];states=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',
            headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1550,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL);page.wait_for_function('window.reviewData?.samples.length===73')
        page.wait_for_function('document.getElementById("video").readyState>=2')
        assert page.locator('#before .event').count()==21
        assert page.locator('#after .event').count()==14
        page.locator('#play').click();page.wait_for_timeout(1200)
        state=page.locator('#video').evaluate('(v)=>({time:v.currentTime,muted:v.muted,volume:v.volume,audioBytes:v.webkitAudioDecodedByteCount,width:v.videoWidth,height:v.videoHeight,error:v.error?.message})')
        assert state['time']>.3 and not state['muted'] and state['audioBytes']>0,state
        states.append({'sample':'S5193','playback':state});page.locator('#play').click()
        page.locator('#after [data-link]').first.click();assert page.locator('#before .linked').count()>0
        page.screenshot(path=str(W/'preview_desktop.png'),full_page=False)
        for sid,expected in [('improvised_V01_S1400_I00000135_local_ee550232f257','未通过'),('naturalistic_V03_S0220_I00000135_local_6339c7c6a045','预检查')]:
            page.locator('#samples').select_option(sid);assert expected in page.locator('#banner').inner_text()
            if 'S0220' in sid:assert page.locator('#after .event').count()==0
            else:assert page.locator('#after .event').count()>0
        s=next(x for x in data['samples'] if x['role']=='j50_visual_nod_pilot')
        page.locator('#samples').select_option(s['id']);page.wait_for_function('document.getElementById("video").readyState>=2')
        page.locator('#play').click();page.wait_for_timeout(800)
        state=page.locator('#video').evaluate('(v)=>({time:v.currentTime,muted:v.muted,audioBytes:v.webkitAudioDecodedByteCount,width:v.videoWidth,height:v.videoHeight,error:v.error?.message})')
        assert state['time']>.2 and state['audioBytes']>0 and state['width']==state['height']==512,state
        states.append({'sample':s['id'],'playback':state})
        page.locator('#status').select_option('accepted');assert page.locator('#samples option').count()==35
        page.locator('#status').select_option('rejected');assert page.locator('#samples option').count()==38
        page.locator('#status').select_option('all');page.locator('#cohort').select_option('all65');assert page.locator('#samples option').count()==65
        page.locator('#search').fill('no-such-sample');assert page.locator('#samples option').count()==0
        page.locator('#search').fill('');page.locator('#cohort').select_option('all')
        page.locator('#samples').select_option('naturalistic_V03_S0127_I00000135_local_142d65a9d1ad')
        nod=page.locator('#after .event').filter(has_text='Head nodding.');assert nod.count()>0
        nod.first.locator('[data-link]').click();assert page.locator('#before .linked').count()>0
        page.set_viewport_size({'width':720,'height':950})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not errors,errors
        browser.close()
    result={'samples':73,'media_head_200':73,'browser_errors':errors,'playback_checks':states,
        'status_filter_counts':{'accepted':35,'rejected':38},'proposal_and_preflight_distinct':True,
        'source_links_work':True,'mobile_no_horizontal_overflow':True}
    (W/'preview_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
