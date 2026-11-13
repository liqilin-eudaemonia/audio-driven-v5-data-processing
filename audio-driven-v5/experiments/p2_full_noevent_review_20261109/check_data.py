from pathlib import Path
import sys,random,torch,copy
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,atomic_json
from p2_ltx_all_lora.data import Dataset as Full
from p2_ltx_all_lora_noevent.data import Dataset as NoEvent
from p2_whisper_event_feature.execution import CPUInputStream
from p2_j_cache.data import caption_prompt
W=Path(__file__).parent;torch.set_num_threads(2);results={}
for arm,name,cls in [('full','p2_ltx_all_lora_full28_batch28_20261109',Full),('noevent','p2_ltx_all_lora_noevent24_after6000_20261109',NoEvent)]:
 c=read(ROOT/'experiments'/name/'config.json')
 if arm=='full':p=c.pop('batch28_continuation');c['training'].update(global_batch=31,steps=p['original_steps'])
 d=cls(c);r=random.Random(1109);ids=[]
 for split,n in [('train',12),('dev',4)]:ids+=r.sample([i for i,row in enumerate(d.rows)if row['split']==split],n)
 rows=[]
 with CPUInputStream(d,[(j+1,[(i,j)])for j,i in enumerate(ids)],2)as stream:
  for j,item in enumerate(stream):
   v=item['samples'][0];sid=d.rows[ids[j]]['sample_id'];assert v['sample_id']==sid
   # Both current readers bind the complete caption to the same final selection.
   reader=d.reader if hasattr(d,'reader')else d.current_reader
   row=reader.source.row(sid)
   assert v['text']['positive']['prompt']==caption_prompt(row['caption'])
   assert v.get('teacher')is not None and v.get('whisper')is not None
   assert ('events'in v)==(arm=='full')and('hidden_event_layout'in v)==(arm=='full')
   if arm=='full':
    assert len(v['events'])==19
    for label in v['events'].values():assert label['known'].dtype==torch.bool and ((label['y']==0)|(label['y']==1)).all()
   rows.append(dict(sample_id=sid,split=row['split'],caption_sha256=row['caption_file_sha256']))
 results[arm]=rows
assert results['full']==results['noevent']
atomic_json(W/'DATA_ACCEPTED.json',dict(success=True,each_arm_reads=16,background_prefetch=True,matching_members_and_full_latest_captions=True,full_event_present_noevent_absent=True,rows=results))
print('Data review passed')
