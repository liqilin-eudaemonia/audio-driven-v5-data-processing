from pathlib import Path
import sys,random,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_ltx_all_lora_components.config import load
from p2_ltx_all_lora_components.data import Dataset
from p2_whisper_event_feature.execution import CPUInputStream
from p2_j_cache.data import caption_prompt
from p2_listen.contracts import read,atomic_json
W=Path(__file__).parent;torch.set_num_threads(2);results={};targets={}
for arm in ['no_feature','no_whisper']:
 c=load(W/arm/'config.json');d=Dataset(c);exclude={r['sample_id']for r in read(W/arm/'preflight.json')['canary_reads']};r=random.Random(20261109)
 ids=[]
 for split,n in [('train',12),('dev',4)]:ids+=r.sample([i for i,v in enumerate(d.rows)if v['split']==split and v['sample_id']not in exclude],n)
 rows=[];requests=[(j+1,[(i,10000+j)])for j,i in enumerate(ids)]
 with CPUInputStream(d,requests,2)as stream:
  for j,item in enumerate(stream):
   assert item['step']==j+1 and item['chosen']==requests[j][1];v=item['samples'][0];sid=d.rows[ids[j]]['sample_id'];assert v['sample_id']==sid
   row=d.reader.source.row(sid);assert v['text']['positive']['prompt']==caption_prompt(row['caption'])
   assert (v.get('whisper')is not None)==(arm=='no_feature')and(v.get('teacher')is not None)==(arm=='no_whisper')
   for k,label in v['events'].items():
    assert label['known'].dtype==torch.bool and ((label['y']==0)|(label['y']==1)).all() and torch.isfinite(label['weight']).all()
    key=(sid,k)
    if arm=='no_feature':targets[key]={n:t.clone()for n,t in label.items()}
    else:assert all(torch.equal(t,targets[key][n])for n,t in label.items())
   rows.append(dict(sample_id=sid,split=row['split'],event_classes=len(v['events'])))
 results[arm]=rows
assert results['no_feature']==results['no_whisper']
atomic_json(W/'REVIEW_PREFETCH_ACCEPTED.json',dict(success=True,each_arm_reads=16,real_background_prefetch=True,disjoint_from_canaries=True,event_targets_equal_across_arms=True,rows=results))
print('Both arms: 16 extra samples each, real prefetch, exact caption and equal Event tensors')
