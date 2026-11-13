"""Browser acceptance for the two-source repaired whole-pair review."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/localbc_pairs/manifest.json').read_text())
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
        assert page.locator('#fixedVersion').input_value()==d['revision']
        page.locator('#fixedFilter').select_option('semantic')
        assert page.locator('#fixedList button').count()==d['semantic_review_count']
        page.locator('#fixedVersion').select_option('local-listening-boundary-repair-2')
        assert page.evaluate('window.fixedPairCurrent.id').startswith('F')
        page.locator('#fixedVersion').select_option(d['revision'])
        assert page.evaluate('window.fixedPairCurrent.id').startswith('L')
        page.locator('#fixedVersion').select_option('local-listening-boundary-repair-gap5-3')
        assert page.evaluate('window.fixedPairCurrent.id').startswith('G')
        page.locator('#fixedVersion').select_option(d['revision'])
        for c in d['comparison_cases']:
            for r in c['before']+c['after']:
                button=page.locator(f'[data-comparison-replay="{r["id"]}"]')
                button.click()
                video=button.locator('..').locator('video')
                page.wait_for_function('(id)=>{const v=document.querySelector(`[data-comparison-replay="${id}"]`).parentElement.querySelector("video");return !v.paused && v.currentTime>.1}',arg=r['id'])
                assert abs(video.evaluate('(v)=>v.duration')-r['duration'])<.09
                assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
                video.evaluate('(v)=>v.pause()')
        if number==1:
            assert '句尾额外留白为 0' in page.locator('#compare-sentence_priority').inner_text()
        for width in [1440,390]:
            page.set_viewport_size({'width':width,'height':1100})
            page.evaluate('document.fonts.ready')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#fixedPairReview').scroll_into_view_if_needed()
            page.screenshot(path=str(EXP/f'localbc_source{number}_{width}.png'))
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
            r=page.evaluate('window.fixedPairCurrent')
            if r.get('boundary_repair') or (r['start']<=335.787 and r['end']>=336.355 and number==2):
                assert r['pool']=='keep_candidate'
            page.locator('#fixedReplay').click()
            page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.15')
            page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
        page.locator('#fixedFilter').select_option('withheld')
    with page.expect_download() as event:page.locator('#fixedExport').click()
    feedback=json.loads(Path(event.value.path()).read_text())
    assert len(feedback['rows'])==2
    assert all(r['reviewed_whole_pair'] and r['start']==r['suggested_start'] and r['end']==r['suggested_end'] for r in feedback['rows'])
    assert feedback['production_modified'] is False
    first=manifest['pairs'][0]
    page.goto(BASE+f'?pair={first["pair"]}&fixed=1&clip=G1-14#fixedPairReview')
    page.wait_for_function('window.fixedPairCurrent?.id==="G1-14"')
    assert 'It was crazy' in page.evaluate('window.fixedPairCurrent.text')
    assert page.evaluate('window.fixedPairCurrent.pool')=='keep_candidate'
    assert '尚未确认' in page.locator('.fixed-semantic').inner_text()
    assert '候选时长' in page.locator('#fixedPlayer > strong').inner_text()
    assert page.locator('#fixedVersion').input_value()=='local-listening-boundary-repair-gap5-3'
    page.goto(BASE+f'?pair={first["pair"]}&fixed=1&clip=F1-13#fixedPairReview')
    page.wait_for_function('window.fixedPairCurrent?.id==="F1-13"')
    assert page.locator('#fixedVersion').input_value()=='local-listening-boundary-repair-2'
    for case in first['comparison_cases']:
        page.goto(BASE+f'?pair={first["pair"]}&fixed=1&compare={case["id"]}#fixedPairReview')
        page.wait_for_function('(id)=>{const r=document.getElementById("compare-"+id)?.getBoundingClientRect();return r && r.top>=-1 && r.top<100}',arg=case['id'])
        for r in case['after']:
            page.locator(f'[data-new-candidate="{r["id"]}"]').click()
            assert page.evaluate('window.fixedPairCurrent.id')==r['id']
            assert page.evaluate('window.fixedPairCurrent.pool')=='keep_candidate'
            page.locator('#fixedReplay').click()
            page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.15 && !document.querySelector("#fixedPlayer video").paused')
            assert page.evaluate('[...document.querySelectorAll("#localBcComparisons video")].every(v=>v.paused)')
            page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
    page.evaluate('localStorage.removeItem("listen-repaired-whole-pairs-review-v3")')
    browser.close()
assert not errors,errors
report=dict(unique_clips_checked=len(set(checked)),withheld_whole_pairs_played=playbacks,
    page_errors=errors,no_added_context=True,first_music_pair_seconds=11.273,
    saved_and_exported_full_pair_range=True,mobile_no_overflow=True,repair_examples_pass_rules=True,
    local_feedback_variant_default=True,five_and_three_second_history_available=True,all_before_after_playbacks=True,main_player_independent=True,semantic_review_separate=True,it_was_crazy_same_pair=True)
(EXP/'localbc_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
