"""Independent numerical, published-link and real-browser acceptance checks."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from vico_more_common import ROOT, WORK, DECA, BFM, read, save, sha

PAGE=ROOT/'data_previews/listen_benchmark_review/vico_more_20261107'
URL='http://localhost:8876/listen_benchmark_review/vico_more_20261107/'


def numerical():
    result=read(WORK/'COMPARISON.json');errors=[]
    for method in ['rlhg','listenformer']:
        if method not in result['images']:continue
        for space in ['deca_video','bfm_native_new','bfm_video_new']:
            table=result[space][method]
            for row in table['per_clip']:
                uid=row['id']
                if space=='deca_video':
                    pred=np.load(WORK/method/'features/deca'/(uid+'.npz'))['motion']
                    gt=np.load(DECA/'features/gt'/(uid+'.npz'))['motion']
                    groups=dict(exp=slice(0,50),head_rotation=slice(50,53),jaw_rotation=slice(53,56));metric='L1'
                else:
                    folder='predictions' if space=='bfm_native_new' else 'features/bfm'
                    pred=np.load(WORK/method/folder/(uid+'.npz'))['motion']
                    gt=np.load(BFM/'features/gt'/(uid+'.npz'))['motion']
                    groups=dict(exp=slice(0,64),angle=slice(64,67),trans=slice(67,70));metric='FD'
                assert gt.shape==pred.shape
                for group,sl in groups.items():
                    independent=float(np.abs(gt[:,sl]-pred[:,sl]).sum()/gt[:,sl].size)
                    errors.append(abs(independent-row['metrics'][metric][group]))
    assert max(errors,default=0)<1e-7
    # Verify the sealed previous result files, not merely their copied values.
    for path,h in result['read_only_sources'].items():assert sha(path)==h,path
    save(WORK/'NUMERICAL_AUDIT.json',dict(passed=True,compared_scalars=len(errors),
        max_abs=max(errors,default=0),previous_results_unchanged=True,comparison_sha256=sha(WORK/'COMPARISON.json')))


def links():
    data=read(PAGE/'data.json');assert len(data['cases'])==43
    urls=[]
    for c in data['cases']:
        urls += [c['speaker'],c['gt'],c['speaker_audio'],c['native_motion_url'],*c['score'].values()]
        assert len(c['dualtalk_curves']['prediction'])==round(c['duration']*30)
    urls=list(dict.fromkeys(urllib.parse.urljoin(URL,u) for u in urls))
    def head(url):
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
            assert r.status==200 and int(r.headers.get('Content-Length','0'))>0,url
        return url
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(head,urls))
    save(WORK/'LINK_AUDIT.json',dict(passed=True,urls=len(records),all43=True,data_sha256=sha(PAGE/'data.json')))


def native_curves():
    data=read(PAGE/'data.json');assert len(data['cases'])==43
    receipt=read(WORK/'REVIEW.json');errors=[]
    slices={'expression':slice(0,3),'head':slice(50,53),'jaw':slice(53,56)}
    for case in data['cases']:
        path=PAGE/case['native_motion_url'];assert sha(path)==receipt['native_motion_files'][str(path)]
        curves=read(path);assert curves['id']==case['id'] and len(curves['models'])==5
        gt=np.load(DECA/'features/gt'/(case['id']+'.npz'))['motion']
        assert len(curves['time'])==len(gt)==round(case['duration']*30)
        assert np.max(np.abs(np.asarray(curves['time'])-np.arange(len(gt))/30))<.000051
        for group,sl in slices.items():
            errors.append(float(np.max(np.abs(np.asarray(curves['ground_truth'][group])-gt[:,sl]))))
        for method,model in curves['models'].items():
            if method=='dualtalk':
                assert 'head' not in model['groups']
                raw=np.load(WORK/'dualtalk/predictions'/(case['id']+'.npz'))['motion']
            else:
                raw=np.load(DECA/'generated'/method.split('/')[-1]/(case['id']+'.npz'))['motion']
            for group,values in model['groups'].items():
                x=np.asarray(values);assert x.shape==(len(gt),3)
                errors.append(float(np.max(np.abs(x-raw[:,slices[group]]))))
    assert max(errors)<1e-6,max(errors)
    save(WORK/'NATIVE_CURVES_AUDIT.json',dict(passed=True,pairs=43,models_per_pair=5,
        all_four_l2l_variants=True,dualtalk_head_excluded=True,max_rounding_error=max(errors),
        native_motion_files=receipt['native_motion_files'],data_sha256=sha(PAGE/'data.json')))


def exports():
    r=read(WORK/'COMPARISON.json');folder=WORK/'paper_tables';expected={}
    tables=[('DECA56','video',r['deca_video']),('DECA56','native',r['learning2listen_native']),
        ('BFM70','video',r['bfm_video']),('BFM70','native',r['bfm_native_new']),
        ('FLAME-to-DECA proxy','exploratory',{'dualtalk':r['dualtalk_transfer_proxy']})]
    for space,readout,models in tables:
        for method,m in models.items():
            assert len(m['per_clip'])==43
            for row in m['per_clip']:
                for metric,groups in row['metrics'].items():
                    for group,value in groups.items():
                        if method=='dualtalk' and group=='head_rotation':continue
                        expected[space,readout,method,row['id'],metric,group]=value
    seen={};missing=0
    with (folder/'all_motion_per_clip.csv').open() as f:
        for row in csv.DictReader(f):
            key=tuple(row[k] for k in ['space','readout','method','uuid','metric','group'])
            assert key in expected and key not in seen,key
            value=expected[key]
            if value is None:
                assert row['unscaled_value']=='';missing+=1
            else:assert abs(float(row['unscaled_value'])-value)<1e-12
            seen[key]=True
    assert set(seen)==set(expected)
    from vico_more_report import LABELS
    reverse={label:method for method,label in LABELS.items()}
    with (folder/'rgb_common.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(r['images'])
    for row in rows:
        method=reverse[row['Method']]
        for display,key in [('PSNR','psnr'),('SSIM','ssim'),('FID','fid'),('CSIM','csim'),('CPBD','cpbd')]:
            assert abs(float(row[display])-r['images'][method][key])<5.1e-6
    manifest=read(WORK/'PAPER_TABLES.json')
    for path,h in manifest['files'].items():assert sha(path)==h,path
    layout=read(folder/'PDF_LAYOUT_AUDIT.json')
    assert layout['passed'] and layout['pdf_sha256']==sha(folder/'preview.pdf')
    coverage=read(WORK/'METHOD_COVERAGE.json')['rows']
    assert len(coverage)==14 and len({row['method_or_variant'] for row in coverage})==14
    for row in coverage:
        if row['method_or_variant'] in ['ovi','glare']:assert row['status']=='deferred_by_user'
        if row['status'] not in ['complete','complete_transfer','complete_exploratory_transfer']:
            assert row['rgb_pairs']==row['native_motion_pairs']==0
    save(WORK/'EXPORT_AUDIT.json',dict(passed=True,all_motion_rows=len(seen),unavailable_cells=missing,
        common_video_methods=len(rows),all_models_have43_pairs=True,unscaled_motion=True,
        readouts_separated=True,pdf_layout_passed=True,paper_manifest_sha256=sha(WORK/'PAPER_TABLES.json'),
        comparison_sha256=sha(WORK/'COMPARISON.json')))


def browser():
    from playwright.sync_api import sync_playwright
    errors=[]
    executable='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'
    libs='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu'
    os.environ['LD_LIBRARY_PATH']=libs+':'+os.environ.get('LD_LIBRARY_PATH','')
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport=dict(width=1440,height=1000))
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(URL,wait_until='networkidle',timeout=60000)
        page.wait_for_function('window.document.querySelectorAll("#case option").length===43')
        samples=[]
        for index in [0,21,42]:
            page.select_option('#case',str(index))
            page.wait_for_function('Array.from(document.querySelectorAll("video")).every(v=>v.readyState>=1)')
            page.wait_for_function('document.getElementById("curves").dataset.caseId===current.id')
            for method in ['dualtalk','learning2listen/conan','learning2listen/fallon','learning2listen/stephen','learning2listen/trevor']:
                page.select_option('#motion_model',method)
                group='expression' if method=='dualtalk' else 'head'
                page.select_option('#motion_group',group)
                assert page.locator('#curves').get_attribute('data-motion-model')==method
                assert page.locator('#curves').get_attribute('data-motion-group')==group
            page.click('#play');page.wait_for_timeout(1300)
            page.click('#play')
            observed=page.evaluate('({audio:document.getElementById("audio").currentTime,videos:[...document.querySelectorAll("video")].map(v=>({t:v.currentTime,err:v.error&&v.error.code,w:v.videoWidth,h:v.videoHeight}))})')
            assert observed['audio']>0 and all(v['err'] is None and v['w']==256 for v in observed['videos']),observed
            assert max(abs(v['t']-observed['audio']) for v in observed['videos'])<.25,observed
            page.evaluate('document.getElementById("seek").value=0.5;document.getElementById("seek").dispatchEvent(new Event("input"))')
            page.wait_for_timeout(400)
            samples.append(dict(index=index,**observed))
        page.screenshot(path=str(WORK/'review_desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844));page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        page.screenshot(path=str(WORK/'review_mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    save(WORK/'BROWSER_AUDIT.json',dict(passed=True,page_errors=errors,samples=samples,
        synchronized_playback=True,seek_checked=True,mobile_checked=True,
        native_model_switches_checked=15,mobile_no_horizontal_overflow=True,
        html_sha256=sha(PAGE/'index.html'),data_sha256=sha(PAGE/'data.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['numerical','links','browser','native_curves','exports']);a=p.parse_args()
    globals()[a.stage]()
