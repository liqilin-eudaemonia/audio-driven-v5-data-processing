"""Validate raw source values, every delta and browser delivery on both pages."""
import csv,hashlib,json,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_responsenet_random200_twelveway_20261105'
PAGE=ROOT/'data_previews/listen_benchmark_review/p2_responsenet_random200_20261105'
URL='http://127.0.0.1:8876/listen_benchmark_review/p2_responsenet_random200_20261105/'


def main():
 d=json.loads((PAGE/'dashboard.json').read_text());checks=0
 for key,ds in d['datasets'].items():
  original={}
  for filename,digest in ds['sources'].items():
   p=Path(filename);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
   original.update(json.loads(p.read_text()))
  def value(arm,metric):
   s=original[arm+'/'+key];a=s['audio']['activity']['acoustic']
   if metric=='emo_div_gap':return abs(s['metrics']['emo_div_generated']-s['metrics']['emo_div_gt'])
   if metric=='strict_audio_f1':return a['strict']['f1']
   if metric in ['missed_response_rate','false_sound_clip_rate']:return a[metric]
   return s['metrics'][metric]
  for row in ds['rows']:
   for metric in d['metrics']:assert row[metric['key']]==value(row['arm'],metric['key'])
  for pair in ds['comparisons']:
   for metric in d['metrics']:
    x=next(x for x in pair['metrics'] if x['key']==metric['key']);a=value(pair['baseline'],metric['key']);b=value(pair['comparison'],metric['key'])
    assert x['baseline']==a and x['comparison']==b
    expected=None if a is None or b is None else b-a
    assert x['delta']==expected
    if expected is not None:
     assert x['display_delta']==expected*(100 if metric.get('percent') else 1)
     expected_better=None if abs(expected)<1e-12 or metric['direction']==0 else expected*metric['direction']>0
     assert x['better']==expected_better
    checks+=1
  csvrows=list(csv.DictReader((PAGE/(key+'_ablation_deltas.csv')).open(encoding='utf-8-sig')))
  assert len(csvrows)==21*14
  with (PAGE/(key+'_all14_summary.csv')).open(encoding='utf-8-sig') as f:
   c=csv.DictReader(f);assert len(c.fieldnames)==len(set(c.fieldnames));assert len(list(c))==14
 # Independent known comparison: 6000 Event on new ResponseNet, positive PSNR
 # and negative LPIPS are BOTH improvements. Expression deltas are points.
 rn=d['datasets']['responsenet'];p=next(x for x in rn['comparisons'] if x['id']=='event6000')
 vals={x['key']:x for x in p['metrics']}
 assert abs(vals['psnr']['delta']-.270774220)<1e-5
 assert vals['psnr']['better'] and vals['lpips']['better']
 assert vals['emo_acc']['delta_unit']=='百分点'
 libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
 os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
 fonts=Path('/tmp/native-fps-browser-libs/extracted/usr/share/fonts/opentype/noto')
 if fonts.exists():
  fc=WORK/'dashboard_fontconfig.xml';fc.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>/usr/share/fonts</dir><dir>{fonts}</dir><cachedir>/tmp/p2-dashboard-fontcache</cachedir></fontconfig>');os.environ['FONTCONFIG_FILE']=str(fc)
 from playwright.sync_api import sync_playwright
 errors=[];browserchecks=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox','--disable-gpu'])
  page=browser.new_page(viewport={'width':1510,'height':1150},accept_downloads=True);page.on('pageerror',lambda e:errors.append(str(e)))
  assert page.goto(URL).status==200;page.wait_for_function('window.dashboardData && document.querySelector("#ablationTables").dataset.loaded==="true"')
  for key in ['responsenet','seamless']:
   page.click('[data-dataset="'+key+'"]');page.wait_for_function('(key)=>window.dashboardDataset===key',arg=key)
   assert page.locator('#overall tbody tr').count()==14 and page.locator('.p2-ablation').count()==7
   if key=='responsenet':assert '23.98 / 24 / 25 / 29.97 / 30' in page.locator('#protocol').inner_text()
   assert page.locator('.p2-delta-table tbody tr').count()==21
   valid=page.evaluate('''()=>{let n=0;const ds=dashboardData.datasets[dashboardDataset];for(const cell of document.querySelectorAll('.p2-delta-table td[data-raw-delta]')){const pair=ds.comparisons.find(x=>x.id===cell.dataset.comparison);const metric=pair.metrics.find(x=>x.key===cell.dataset.metric);if(Number(cell.dataset.rawDelta)!==metric.delta)throw Error('DOM raw delta mismatch');const cls=metric.better===true?'good':metric.better===false?'bad':'flat';if(!cell.classList.contains(cls))throw Error('DOM direction mismatch');n++;}return n;}''')
   assert valid==168
   page.locator('.p2-original').first.evaluate('(e)=>e.open=true');assert page.locator('.p2-original').first.locator('tbody tr').count()==14
   assert 'pp' in page.locator('.p2-original').first.inner_text()
   assert page.locator('#overall .rank1').count()>=8 and page.locator('#overall .rank2').count()>=8
   page.evaluate('scrollTo(0,0)');page.screenshot(path=str(WORK/('dashboard_'+key+'.png')),full_page=True)
   page.set_viewport_size({'width':390,'height':844});page.locator('details.card').last.evaluate('(e)=>e.open=true')
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),'mobile page overflow'
   page.screenshot(path=str(WORK/('dashboard_'+key+'_mobile.png')),full_page=True)
   page.set_viewport_size({'width':1510,'height':1150})
   browserchecks.append(dict(dataset=key,rows=14,ablation_groups=7,visible_delta_cells=valid,mobile_no_overflow=True))
  page.goto(URL+'?dataset=responsenet&ablation=joint_whisper');page.wait_for_function('window.dashboardDataset==="responsenet"')
  with page.expect_download() as event:page.get_by_text('下载全部消融增减CSV',exact=True).click()
  assert event.value.suggested_filename=='responsenet_ablation_deltas.csv'
  assert len(list(csv.DictReader(Path(event.value.path()).open(encoding='utf-8-sig'))))==294
  seam='http://127.0.0.1:8876/listen_benchmark_review/p2_seamless200_20261105/?preset=joint_av&metrics=1'
  assert page.goto(seam).status==200
  page.wait_for_function('window.reviewData && reviewDebug().loaded && document.querySelector("#ablationTables").dataset.loaded==="true"',timeout=60000)
  assert page.locator('#ablationTables .p2-delta-table tbody tr').count()==21
  assert page.locator('#modelB').input_value()=='base_event_feature_whisper_6000'
  assert page.locator('#summaryTable tbody tr').count()==14
  assert page.locator('[data-preset="joint_event"]').is_visible()
  page.click('[data-preset="joint_event"]');page.wait_for_function('reviewDebug().loaded && document.querySelector("#modelA").value==="av_event_6000"')
  assert page.locator('#modelB').input_value()=='base_event_feature_whisper_6000'
  page.click('[data-focus="b"]');page.click('#play');page.wait_for_timeout(600)
  assert page.evaluate('reviewDebug().context==="running" && reviewDebug().sources===1')
  page.click('#pause');assert not errors,errors
  browser.close()
 report=dict(passed=True,source_numeric_and_delta_checks=checks,browser_checks=browserchecks,
  original_seamless_video_audio_controls=True,download_csv_checked=True,page_errors=errors)
 (WORK/'DASHBOARD_BROWSER_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
