"""Browser acceptance for the two-source repaired whole-pair review."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/fixed_pairs/manifest.json').read_text())
errors=[];checked=[];playbacks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    for number,d in enumerate(manifest['pairs'],1):
        pair=d['pair'];withheld=[r for r in d['rows'] if r['initial'] and r['pool']!='keep_candidate']
        page.goto(BASE+f'?pair={pair}&fixed=1#fixedPairReview')
        page.wait_for_function('(p)=>window.fixedPairCurrent?.pair===p && document.querySelector("#fixedPairReview").open',arg=pair)
        assert page.locator('#fixedList button').count()==len(withheld)
        assert page.locator('#manualReview > summary').inner_text().startswith('历史')
        assert page.locator('#historicalTimeline').is_visible()
        for width in [1440,390]:
            page.set_viewport_size({'width':width,'height':1100})
            page.evaluate('document.fonts.ready')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#fixedPairReview').scroll_into_view_if_needed()
            page.screenshot(path=str(EXP/f'fixed_source{number}_{width}.png'))
        page.set_viewport_size({'width':1440,'height':1100})
        for mode in ['all','terminal']:
            page.locator('#fixedFilter').select_option(mode)
            expected=[r for r in d['rows'] if r['initial']] if mode=='all' else [r for r in d['rows'] if r['id'] in d['terminal_ids']]
            assert page.locator('#fixedList button').count()==len(expected)
            for r in expected:
                if r['id'] in checked:continue
                page.locator(f'[data-fixed-id="{r["id"]}"]').click()
                page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id && document.querySelector("#fixedPlayer video").readyState>=1',arg=r['id'])
                assert abs(page.locator('#fixedPlayer video').evaluate('(v)=>v.duration')-r['duration'])<.09
                assert r['added_context_seconds']==0
                assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
                assert '没有额外上下文' in page.locator('#fixedPlayer .fixed-range').inner_text()
                checked.append(r['id'])
        page.locator('#fixedFilter').select_option('withheld')
        for r in withheld:
            page.locator(f'[data-fixed-id="{r["id"]}"]').click()
            page.locator('#fixedReplay').click()
            page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.1 && !document.querySelector("#fixedPlayer video").paused')
            for mode in ['a','b','both']:page.locator('#fixedChannel').select_option(mode)
            for wi,w in enumerate(r['evidence_windows']):
                button=page.locator('[data-evidence-time]').nth(wi)
                button.click()
                page.wait_for_function('(t)=>Math.abs(document.querySelector("#fixedPlayer video").currentTime-t)<.8',arg=max(0,w['start']-r['start']-.2))
            page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
            playbacks.append(r['id'])
        page.locator('#fixedFilter').select_option('withheld')
        selected=page.evaluate('window.fixedPairCurrent')
        page.locator('#fixedDecision').select_option('keep')
        page.locator('#fixedStart').fill(str(selected['start']+.2))
        page.locator('#fixedNote').fill('自动验收记录，不是真实人工结论')
        page.locator('#fixedSave').click()
        page.reload()
        page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id',arg=selected['id'])
        assert page.locator('#fixedDecision').input_value()=='keep'
        assert float(page.locator('#fixedStart').input_value())==selected['start']
        if number==1:
            assert abs(selected['duration']-11.273)<1e-6
            assert selected['start']==19.437 and selected['end']==30.71
        for b in page.locator('[data-repaired]').all():
            b.click()
            assert page.evaluate('window.fixedPairCurrent.pool')=='keep_candidate'
            page.locator('#fixedReplay').click()
            page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.15')
            page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
        page.locator('#fixedFilter').select_option('withheld')
    with page.expect_download() as event:page.locator('#fixedExport').click()
    feedback=json.loads(Path(event.value.path()).read_text())
    assert len(feedback['rows'])==2
    assert all(r['reviewed_whole_pair'] and r['start']==r['suggested_start'] and r['end']==r['suggested_end'] for r in feedback['rows'])
    assert feedback['production_modified'] is False
    page.evaluate('localStorage.removeItem("listen-repaired-whole-pairs-review-v2")')
    browser.close()
assert not errors,errors
report=dict(unique_clips_checked=len(set(checked)),withheld_whole_pairs_played=playbacks,
    page_errors=errors,no_added_context=True,first_music_pair_seconds=11.273,
    saved_and_exported_full_pair_range=True,mobile_no_overflow=True,repair_examples_pass_rules=True)
(EXP/'fixed_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
