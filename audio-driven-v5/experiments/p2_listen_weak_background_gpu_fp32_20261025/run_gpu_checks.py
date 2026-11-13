"""Bounded GPU validation on node8 GPUs 4/5/6; never stop other workloads."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,read
from p2_listen.config import load,activate,source_identity
from p2_listen.dataset import load_observations,load_targets
from p2_listen.features import load_training_sample,load_text
import torch
os.environ['OMP_NUM_THREADS']='4';torch.set_num_threads(4)
c=load(OUT/'config.json');activate(c)
obs=load_observations(c);targets=load_targets(c,obs)
for o in obs:
    data=load_training_sample(c,o,targets[o['sample_id']],'cpu')
    assert all(torch.isfinite(v['y']).all() for v in data[-1].values())
load_text(c,'cpu')
atomic_json(OUT/'preflight.json',dict(samples=len(obs),cache_identities_verified=True,source=source_identity(c),host=os.uname().nodename,torch=torch.__version__,cuda=torch.version.cuda))
PYTHON=sys.executable
CFG=str(OUT/'config.json');SEED='20261027';RUN=OUT/'runs'/('s0_'+SEED)
base=[PYTHON,str(ROOT/'scripts/p2_listen.py'),'train','--config',CFG,'--train-seed',SEED,'--s0','--steps','7']
obs_gen=read(OUT/'generation_observation.json');t=targets[obs_gen['sample_id']]
specs={
 'E0':([4],base+['--arm','E0'],None),
 'E1_first5':([5],base+['--arm','E1','--stop-after','5'],None),
 'E1_resume':([5],base+['--arm','E1','--resume',str(RUN/'E1/step_000005.pt')],'E1_first5'),
 'E1_ddp':([4,6],[PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'scripts/train_p2_listen_distributed.py'),'--config',str(OUT/'ddp/config.json'),'--arm','E1','--train-seed',SEED,'--s0','--steps','7'],'E0'),
 'replay':([5],[PYTHON,str(ROOT/'scripts/check_p2_listen_gpu_replay.py'),'--checkpoint',str(RUN/'E1/step_000006.pt'),'--expected',str(RUN/'E1/step_000007.pt'),'--output',str(OUT/'replay.json')],'E1_resume'),
 'generate':([5],[PYTHON,str(ROOT/'scripts/check_p2_listen_source_only.py'),'--config',CFG,'--observation',str(OUT/'generation_observation.json'),'--checkpoint',str(RUN/'E1/step_000007.pt'),'--output',str(OUT/'generated_source_only'),'--deny',t['target_video'],'--deny',t['target_audio']],'replay'),
}
records={};processes={};logs={};began=time.time()
def gpu_snapshot():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(v[0]):dict(total=int(v[1]),used=int(v[2]),utilization=int(v[3])) for line in r.stdout.splitlines() if (v:=line.split(','))}
atomic_json(OUT/'gpu_before.json',gpu_snapshot())
while True:
    for name,p in list(processes.items()):
        if p.poll() is not None and records[name]['state']=='running':
            records[name].update(state='passed' if p.returncode==0 else 'failed',exit_code=p.returncode,ended=time.time())
            logs[name].close()
    snapshot=gpu_snapshot()
    for name,(gpus,cmd,dependency) in specs.items():
        if name in records:continue
        if dependency and records.get(dependency,{}).get('state') in ('failed','blocked'):
            records[name]=dict(state='blocked',reason='failed prerequisite '+dependency);continue
        if dependency and records.get(dependency,{}).get('state')!='passed':continue
        if any(snapshot[g]['total']-snapshot[g]['used']<60*1024 for g in gpus):continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        logs[name]=(OUT/(name+'.log')).open('w')
        processes[name]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
        records[name]=dict(state='running',pid=processes[name].pid,gpus=gpus,command=cmd,started=time.time())
    terminal=len(records)==len(specs) and all(r['state']!='running' for r in records.values())
    atomic_json(OUT/'job_status.json',dict(started=began,updated=time.time(),jobs=records,gpus=snapshot,finished=terminal,passed=terminal and all(r['state']=='passed' for r in records.values())))
    if terminal:break
    if time.time()-began>3600:raise TimeoutError('Validation exceeded 1h; inspect recorded owned PIDs')
    time.sleep(10)
atomic_json(OUT/'gpu_after.json',gpu_snapshot())
print('GPU jobs finished', {k:r['state'] for k,r in records.items()},flush=True)
raise SystemExit(0 if all(r['state']=='passed' for r in records.values()) else 1)
