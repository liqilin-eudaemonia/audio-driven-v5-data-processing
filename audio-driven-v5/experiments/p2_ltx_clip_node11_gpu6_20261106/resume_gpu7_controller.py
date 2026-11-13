"""Resume the same single-GPU run on physical GPU7 after graceful GPU6 exit."""
from pathlib import Path
import sys,json,time,runpy,os
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,atomic_json,sha,digest
from p2_ltx_events.config import source_identity
from p2_ltx_events.data import data_identity
from p2_ltx_events.training import check_preflight
from listen_cluster import launch
OUT=Path(__file__).resolve().parent;torch.set_num_threads(2)
while not (OUT/'launch/completed.json').exists():time.sleep(5)
assert read(OUT/'launch/completed.json')['returncodes']==[0]
c=read(OUT/'config.json');step=read(OUT/'status.json')['step']
checkpoint=OUT/f'step_{step:06d}.pt';marker=read(checkpoint.with_suffix('.complete.json'));assert marker['step']==step
saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert saved['step']==step and saved['identity']['experiment']==c and saved['identity']['world']==1
assert saved['optimizer']['state'] and len(saved['rank_rng'])==1
pre=read(OUT/'preflight.json');check_preflight(c,pre,source_identity(),data_identity(c))
assert saved['identity']['source']==source_identity()
cluster=read(OUT/'cluster.json');cluster['nodes'][0]['gpus']=[7]
atomic_json(OUT/'cluster_gpu7.json',cluster,immutable=True)
cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='resume_gpu7_commands')
commands=cli['commands'](cluster,OUT/'config.json',resume=checkpoint)
assert 'CUDA_VISIBLE_DEVICES=7' in commands[0] and '--resume' in commands[0]
atomic_json(OUT/'COMMANDS_GPU7.json',commands)
atomic_json(OUT/'GPU7_MIGRATION.json',dict(state='starting_resume',from_gpu=6,to_gpu=7,checkpoint=str(checkpoint),saved_step=step,
 next_step=step+1,checkpoint_sha256=marker['sha256'],optimizer_and_rng_present=True,training_config_unchanged=True,
 matching_existing_preflight_reused=True,preflight_sha256=sha(OUT/'preflight.json'),controller_pid=os.getpid()))
(OUT/'STOP').unlink()
print(json.dumps(dict(resuming_step=step,gpu=7)),flush=True)
# The worker rechecks full checkpoint hash, identity, numerics and RNG before updates.
launch(cluster,commands,OUT/f'launch_gpu7_from_{step:06d}')
print(json.dumps(read(OUT/'status.json')),flush=True)
