import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review'
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page(viewport={'width':1536,'height':1100});p.goto('http://127.0.0.1:8797/');p.wait_for_function('selected!==undefined');assert p.locator('#changes button').count()==3
 checks=[]
 for i in range(3):
  p.locator('#changes button').nth(i).click();p.wait_for_function('(i)=>v.readyState>=2 && !v.seeking && !v.paused && Math.abs(v.currentTime-(data.sentence_changes[i].preview_time-2))<1',arg=i);checks.append({'change':i,'before_boundary_playback':True})
 p.evaluate('v.pause()');p.locator('#changes').screenshot(path=str(OUT/'sentence_changes.png'));b.close()
(OUT/'sentence_cards_validation.json').write_text(json.dumps(checks,indent=2));print('PASS',len(checks),'sentence boundary cards')
