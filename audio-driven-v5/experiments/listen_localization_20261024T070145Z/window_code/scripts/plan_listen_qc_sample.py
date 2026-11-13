"""Seeded, bounded planning for the next quality-review sample, without training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,random,sys,time
import preview_listen_v6 as preview
from p3_listen.common import read,write,digest
ROOT=preview.ROOT
OUT=ROOT/'data_previews/seamless_listen_qc_review'
SEED=2026102203

def main():
 preview.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
 catalog=read(preview.OLD/'catalog.json');lookup={x['pair']:x for x in catalog['train']+catalog['dev']}
 previous=read(ROOT/'data_previews/seamless_listen_v6_review/samples.json')['samples']
 used_groups={x['group_id'] for x in previous}
 # Previously examined pairs are excluded even if not published.
 examined={f.stem for f in (ROOT/'data_previews/seamless_listen_v6_review/audit').glob('*.json')}
 pool=list(read(preview.OLD/'selected.json')['pairs']);random.Random(SEED).shuffle(pool)
 chosen=[];checked=0;group_seen=set(used_groups)
 for item in pool:
  record=lookup[item['pair']]
  if record['pair'] in examined or record['group_id'] in group_seen:continue
  checked+=1
  norm=preview.OLD/'normalized_pairs'/(item['pair']+'.json')
  if not norm.exists() or not read(norm)['summary']['ok']:continue
  plan=read(item['path'])
  if not any(c['crop_boxes']['listener']['size']>=512 and not c['crop_boxes']['listener']['wide_crop'] for c in plan.get('clips',[])):continue
  chosen.append(record);group_seen.add(record['group_id'])
  if len(chosen)>=48:break
 write(OUT/'random_selection.json',dict(seed=SEED,scope='already normalized official train/dev; original listener crop >=512; new groups',
  candidates_checked=checked,candidates=chosen,excluded_previous_pairs=sorted(examined),excluded_previous_groups=sorted(used_groups)))
 print('bounded pairs',len(chosen),'checked',checked,flush=True)
 with ThreadPoolExecutor(max_workers=6) as executor:
  results=list(executor.map(preview.inspect_pair,chosen))
 rows=[]
 for result in results:
  candidates=result['clips'];random.Random(SEED+int(digest(result['pair'])[:8],16)).shuffle(candidates)
  # At most four randomly selected semantic clips per pair for audio review.
  rows+=candidates[:4]
 write(OUT/'planned_candidates.json',dict(seed=SEED,pairs=len(chosen),rows=rows,
   note='Random candidates before acoustic/model quality screening; not qualified data.'))
 print('planned audio candidates',len(rows),flush=True)
if __name__=='__main__':main()
