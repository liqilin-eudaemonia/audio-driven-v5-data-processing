"""Verify provenance, independent motion readout, and actual review pages."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/common_method_comparison_20261106'
PAGE=ROOT/'data_previews/listen_benchmark_review/common_method_comparison_20261106'
REVIEW=ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106'
L2L=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
URL='http://127.0.0.1:8876/listen_benchmark_review/'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data=read(PAGE/'data.json')
    for path,expected in data['source_sha256'].items():assert sha(Path(path))==expected
    assert not data['claims']['published_tables_comparable_to_local']
    assert not data['claims']['controlled_training_comparison']
    assert len(data['published_reference']['rows'])==8
    assert all(r['origin'].startswith('reported by DIM') for r in data['published_reference']['rows'])
    csv_rows=list(csv.DictReader((PAGE/'video_metrics.csv').open()))
    for row in csv_rows:
        source=read(Path(row['source']))
        ref=next(r for r in data['local_video_results'] if r['dataset']==row['dataset'] and r['arm']==row['arm'])
        if 'source_key' in ref:source=source[ref['source_key']]
        assert float(row['value'])==source['metrics'][row['metric']]
    common=read(WORK/'vico_deca/RESULTS.json')
    if data.get('l2l_retrained'):
        common['models']['learning2listen/retrained']=data['l2l_retrained']
    independent_error=0.
    slices={'exp':slice(0,50),'head_rotation':slice(50,53),'jaw_rotation':slice(53,56)}
    for name,model in common['models'].items():
        for row in model['per_clip']:
            uid=row['id'];gt=np.load(L2L/'features/gt'/(uid+'.npz'))['motion'].astype(float)
            if name=='learning2listen/retrained':
                active=read(WORK/'ACTIVE_L2L_RUN.json') if (WORK/'ACTIVE_L2L_RUN.json').exists() else {'path':str(WORK/'l2l_retrained')}
                path=Path(active['path'])/'test_generated'/(uid+'.npz')
            elif name.startswith('learning2listen/'):path=L2L/'generated'/name.split('/')[1]/(uid+'.npz')
            else:path=WORK/'vico_deca/features'/name/(uid+'.npz')
            assert sha(path)==row['prediction_sha256']
            pred=np.load(path)['motion'].astype(float)
            for group,sl in slices.items():
                value=float(np.abs(gt[:,sl]-pred[:,sl]).sum()/gt[:,sl].size)
                independent_error=max(independent_error,abs(value-row['metrics']['L1'][group]))
    assert independent_error<1e-12
    os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport={'width':1600,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL+PAGE.name+'/').status==200
        assert page.locator('h2').count()==6
        assert '本表不加入我们的分数' in page.inner_text('body')
        assert page.locator('table').count()==5
        page.screenshot(path=str(WORK/'overview.png'),full_page=True)
        assert page.goto(URL+REVIEW.name+'/').status==200
        page.wait_for_function('window.reviewData&&reviewDebug().loaded',timeout=60000)
        assert page.locator('#commonDecaOverview').get_attribute('open') is not None
        assert page.locator('#commonDecaSummary tbody tr').count()==27
        for i,(name,model) in enumerate(common['models'].items()):
            value=page.locator('#commonDecaSummary tbody tr').nth(0).locator('td').nth(i+1).inner_text()
            assert value==f"{model['metrics']['L1']['exp']:.4f}"
        cases=read(REVIEW/'data.json')['cases']
        picks=[cases[0],min(cases,key=lambda c:c['frames']),next(c for c in cases if c['frames']==90)]
        for case in picks:
            page.evaluate('id=>show(reviewData.cases.findIndex(c=>c.id===id)).catch(fail)',case['id'])
            page.wait_for_function('reviewDebug().loaded',timeout=60000)
            assert page.locator('#commonDecaCase tbody tr').count()==27
            for i,(name,model) in enumerate(common['models'].items()):
                row=next(r for r in model['per_clip'] if r['id']==case['id'])
                value=page.locator('#commonDecaCase tbody tr').nth(0).locator('td').nth(i+1).inner_text()
                assert value==f"{row['metrics']['L1']['exp']:.4f}"
            if case['frames']==30:assert page.locator('#commonDecaCase').inner_text().count('N/A')>=36
        page.click('#play');page.wait_for_timeout(700)
        playback=page.evaluate('reviewDebug()')
        assert min(playback['times'].values())>.2 and max(playback['times'].values())-min(playback['times'].values())<.17
        page.click('#pause');page.screenshot(path=str(WORK/'vico_common_deca.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.goto(URL+PAGE.name+'/')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.set_viewport_size({'width':1400,'height':1000})
        pair_data=read(PAGE/'video_pairs.json')
        assert sum(len(d['cases']) for d in pair_data['datasets'].values())==400
        for dataset in ['responsenet','seamless']:
            assert page.goto(URL+PAGE.name+'/video_pairs.html?dataset='+dataset).status==200
            page.wait_for_function('window.pairDebug&&pairDebug().ready',timeout=60000)
            assert page.locator('#clip option').count()==200
            assert page.evaluate('pairDebug().sizes')==[[512,512],[512,512],[640,640] if dataset=='responsenet' else [512,512]]
            page.click('#play');page.wait_for_timeout(700);state=page.evaluate('pairDebug()')
            assert min(state['times'])>.2 and max(state['times'])-min(state['times'])<.17
            page.click('#pause')
            page.select_option('#sound','listener_audio');page.wait_for_timeout(200)
            page.click('#next');page.wait_for_function('pairDebug().ready',timeout=60000)
            assert page.evaluate('pairDebug().id')==pair_data['datasets'][dataset]['cases'][1]['id']
            page.locator('#seek').evaluate("el=>{el.value=.8;el.dispatchEvent(new Event('input'))}")
            page.wait_for_timeout(100);assert min(page.evaluate('pairDebug().times'))>.75
            page.set_viewport_size({'width':390,'height':844})
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
            page.set_viewport_size({'width':1400,'height':1000})
        page.screenshot(path=str(WORK/'fixed200_video_pairs.png'),full_page=True)
        assert not errors,errors
        browser.close()
    report=dict(passed=True, source_hashes_verified=len(data['source_sha256']), video_csv_values_verified=len(csv_rows),
        independent_l1_streams=43*len(common['models']), independent_l1_max_abs=independent_error, common_deca_models=len(common['models']),
        common_summary_rows=27, tested_media_cases=[c['id'] for c in picks], synchronous_playback=playback,
        published_and_local_separate=True, fixed200_native_video_pairs=400,two_dataset_video_playback=True,
        mobile_no_page_overflow=True, browser_errors=errors,
        data_sha256=sha(PAGE/'data.json'), review_data_sha256=sha(REVIEW/'data.json'))
    (WORK/'FINAL_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: comparison provenance, all258 motion outputs, shared DECA tables and synchronized video')


if __name__=='__main__':main()
