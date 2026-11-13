"""Native event comparison on common known class/sample support across stages."""
from pathlib import Path
import copy,sys,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import eval28 as E
from p2_event_presence.scoring import score

def compare(steps):
 records=[];expected=[]
 for step in steps:
  c=E.read(E.R.RUN/'config.json');E.E.verify_report(c,step)
  out=E.folder(step);name=f'step{step}'
  for row in E.values(out/'event_pairs_fixed_reference.jsonl'):records.append(dict(row,method=name))
  for row in E.values(out/'events/frozen_pairs.jsonl'):expected.append(dict(row,method=name))
 native=score(records,expected);dataset=native['datasets']['stage2_dev']
 summary={}
 for step in steps:
  method=dataset['methods'][f'step{step}'];item={}
  for mod in ('visual','audio'):
   metrics=method['common_subset_draw_aggregate'][mod]
   item[mod]=dict(macro_f1=metrics['macro_f1']['mean'],per_class_known_pairs={k:len(v)for k,v in dataset['common_evaluable_ids_by_draw_class']['0'][mod].items()},
    per_class_metrics={k:{key:value for key,value in v.items()if key in('tp','fp','fn','tn','f1','precision','recall','evaluated_pairs','reference_positive_evaluated')}for k,v in method['common_subset_draws'][0][mod]['per_class'].items()})
  summary[str(step)]=item
 result=dict(at=time.time(),steps=steps,summary=summary,native=native,code_sha256=E.sha(__file__),
  reports={str(s):E.sha(E.folder(s)/'REPORT.json')for s in steps},frozen_reference_sha256=E.sha(E.folder(0)/'REFERENCE_EVENTS.json'),
  semantics='Native available-subset metrics on exactly the same known sample/class intersections; unknown/error excluded, never negatives; not full64-complete F1 or human-validated appropriateness')
 E.atomic_json(W/'COMMON_SUPPORT_EVENTS.json',result)
 print({s:{mod:v['macro_f1']for mod,v in item.items()}for s,item in summary.items()})
 return result

if __name__=='__main__':
 result=E.read(W/'FINAL_COMPARISON.json');compare(sorted(map(int,result['summaries'])))
