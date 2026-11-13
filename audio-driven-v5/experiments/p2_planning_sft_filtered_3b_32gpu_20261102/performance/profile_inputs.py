import cProfile,pstats,time,json,sys
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(root/'src'))
from p2_planning_sft.reference_data import ReferencePlanningDataset
from p2_planning_sft.data import PlanningCollator
import torch
torch.set_num_threads(2)
c=json.load(open(root/'configs/p2_planning_sft_filtered_3b_32gpu_20261102.json'))
d=ReferencePlanningDataset(c['data']['seal'],'train',limit=8,seed=20261102)
coll=PlanningCollator(c['model'],c['max_length']);coll.get_processor()
p=cProfile.Profile();p.enable();start=time.monotonic();rows=[]
for i in range(len(d)):
 t=time.monotonic();row=d[i];rows.append(row);print(json.dumps(dict(i=i,seconds=time.monotonic()-t,sample=row['sample_id'])),flush=True)
t=time.monotonic();coll(rows);print('COLLATE_SECONDS',time.monotonic()-t,flush=True);p.disable();print('TOTAL',time.monotonic()-start);pstats.Stats(p).sort_stats('cumulative').print_stats(35)
