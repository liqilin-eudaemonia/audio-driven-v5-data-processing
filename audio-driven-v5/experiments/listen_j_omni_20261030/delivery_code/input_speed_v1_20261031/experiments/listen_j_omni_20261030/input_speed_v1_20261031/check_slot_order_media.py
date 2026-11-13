import os,sys,json,hashlib,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    J_OMNI_BOUNDED_CROP_V1='1',J_OMNI_INPUT_OPTIMIZATIONS=json.dumps(dict(fast_crop=True,shared_decode=True)))
from listen_benchmark.j_omni import prepare_input
from listen_benchmark.j_annotation import read,write
root=D/'slot_order_media';root.mkdir(exist_ok=True)
cfg=read(W/'config.json');cfg['bounded_crop_input']['nod_cache']=str(root/'nod_cache');write(root/'config.json',cfg)
if not (root/'official_metadata').exists():(root/'official_metadata').symlink_to(W/'official_metadata',target_is_directory=True)
tasks=read(D/'gpu_canary/tasks.json');records=[]
for t in [tasks[0],tasks[3]]:
    start=time.monotonic();p=prepare_input(t['row'],t['old_work'],root,cfg['nodes']['node12']['resource_work'])
    actual=dict(prompt=p['prompt'],video=hashlib.sha256(p['frames']).hexdigest(),audio=hashlib.sha256(p['audio']).hexdigest())
    assert actual==t['expected'],t['row']['sample_id']
    records.append(dict(sample_id=t['row']['sample_id'],seconds=time.monotonic()-start,passed=True,execution=p['_input_execution']))
write(D/'slot_order_media_check.json',dict(passed=True,records=records));print(json.dumps(records))
