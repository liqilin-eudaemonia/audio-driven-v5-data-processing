"""Audited single->two-GPU checkpoint migration, then strict native resume."""
from pathlib import Path
import sys,time,os,json,copy,runpy,subprocess,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,sha,atomic_json,digest
from p2_listen.cache import atomic_torch
from p2_ltx_events.config import source_identity,validate_baseline
from p2_ltx_events.training import preflight
from p2_ltx_events.checkpoint import checked
from p2_whisper_event_feature.execution import settings
from listen_cluster import launch
OUT=Path(__file__).resolve().parent;OLD=ROOT/'experiments/p2_ltx_clip_node11_gpu6_20261106';torch.set_num_threads(2)
def status(state,**extra):
 result=dict(state=state,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),**extra)
 atomic_json(OUT/'CONTROLLER_STATE.json',result);print(json.dumps(result),flush=True)
status('waiting_single_gpu_checkpoint')
old_logs=sorted(OLD.glob('launch_gpu7_auto_*'))[-1]
while not (old_logs/'completed.json').exists():time.sleep(5)
assert read(old_logs/'completed.json')['returncodes']==[0]
step=read(OLD/'status.json')['step'];origin=OLD/f'step_{step:06d}.pt'
assert origin.with_suffix('.complete.json').exists()
status('preflight',saved_step=step)
c=read(OUT/'config.json');old_c=read(OLD/'config.json');validate_baseline(c,read(c['hidden_event_parent_config']))
assert {k:v for k,v in c.items() if k!='output'}=={k:v for k,v in old_c.items() if k!='output'}
pre=preflight(c,32)
status('migrating_checkpoint',saved_step=step)
saved=checked(origin);before=saved['identity'];current=source_identity();script=str(ROOT/'scripts/p2_ltx_events.py')
assert before['world']==1 and before['experiment']==old_c and len(saved['rank_rng'])==1
assert set(before['source'])==set(current)
differences={k for k in current if current[k]!=before['source'][k]}
assert differences=={script},differences
assert before['source'][script]==sha(OUT/'launcher_before.py')
assert before['data_identity']==pre['data_identity'] and before['statistics']==pre['statistics']
assert before['execution']==settings(c)
# Same 32 samples and global ordinals at the next step, split over two ranks.
from p2_ltx_events.schedule import Schedule
plan=Schedule(c);next_step=step+1
one=plan.indices(plan.size,next_step,32,c['training']['seed'],0,1)
two=[plan.indices(plan.size,next_step,32,c['training']['seed'],rank,2) for rank in (0,1)]
assert [two[rank][micro] for micro in range(16) for rank in (0,1)]==one
identity=dict(before,world=2,experiment=c,source=current,data_identity=pre['data_identity'],statistics=pre['statistics'])
migrated=dict(saved,identity=identity,rank_rng=[copy.deepcopy(saved['rank_rng'][0]) for _ in range(2)])
lineage=dict(origin=str(origin),origin_sha256=read(origin.with_suffix('.complete.json'))['sha256'],step=step,
 old_world=1,new_world=2,global_batch=32,local_accumulation=16,source_changes=list(differences),
 code_change='single-node launcher supports two visible GPUs; model/loss/optimizer code unchanged',
 optimizer_preserved=True,model_preserved=True,sample_order_and_ordinals_preserved=True,
 rng_policy='source rank0 state copied to both ranks; training reseeds CPU/CUDA and explicit generator per global sample ordinal')
migrated['world_migration']=lineage
checkpoint=OUT/f'step_{step:06d}.pt';atomic_torch(checkpoint,migrated)
atomic_json(checkpoint.with_suffix('.complete.json'),dict(sha256=sha(checkpoint),step=step))
readback=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert readback['identity']==identity and len(readback['rank_rng'])==2
for prefix in ['connector.','event_heads.','decoders.','text_lora.','whisper_adapter.','whisper_branches.']:
 key=next(k for k in saved['model'] if k.startswith(prefix))
 assert torch.equal(saved['model'][key],readback['model'][key])
assert readback['optimizer']['param_groups']==saved['optimizer']['param_groups']
atomic_json(OUT/'MIGRATION.json',lineage)
del saved,migrated,readback
cluster=read(OUT/'cluster.json');utilities=runpy.run_path(str(OLD/'wait_gpu67_resume.py'),run_name='gpu_idle_utilities')
while not utilities['idle'](utilities['probe'](cluster)):
 status('waiting_gpus6_and7',saved_step=step);time.sleep(30)
cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='gpu67_launch')
commands=cli['commands'](cluster,OUT/'config.json',resume=checkpoint)
assert 'CUDA_VISIBLE_DEVICES=6,7' in commands[0] and '--nproc_per_node=2' in commands[0]
atomic_json(OUT/'COMMANDS.json',commands)
status('training',saved_step=step,next_step=step+1,gpus=[6,7],world_size=2,global_batch=32,accumulation_per_gpu=16,save_every=50)
launch(cluster,commands,OUT/'launch')
status('finished',training_status=read(OUT/'status.json'))
