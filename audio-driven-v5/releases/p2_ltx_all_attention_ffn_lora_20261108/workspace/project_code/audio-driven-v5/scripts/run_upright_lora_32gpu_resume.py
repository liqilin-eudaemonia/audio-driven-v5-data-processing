"""User-requested 32-GPU continuation: preflight, owned stop, canary, full run."""
import collections,copy,hashlib,importlib.util,json,os,shlex,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
PARENT=ROOT/'experiments/p2_planning_sft_upright_3b_gpu7_20261105'
WORK=ROOT/'experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106'
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json

def status(stage,**kw):
    r=dict(stage=stage,at=time.time(),pid=os.getpid(),**kw);atomic_json(WORK/'STATUS.json',r);print(json.dumps(r),flush=True)

def preflight():
    from p2_planning_sft.reference_data import datasets
    from p2_planning_sft.data import SYSTEM,USER
    from p2_planning_sft.performance_resume import validate_performance_resume
    config=read(WORK/'config.json');prep=read(WORK/'PREPARED.json')
    assert sha(WORK/'config.json')==prep['config_sha256']
    for p,h in prep['code'].items():assert sha(p)==h,p
    train,dev,full=datasets(config,False)
    identity=dict(config=config,smoke=False,train_manifest=train.manifest_sha256,dev_manifest=dev.manifest_sha256,
        train_ids=digest(train.ids),train_count=len(train),dev_count=len(dev),system=SYSTEM,user=USER,
        code={p.name:sha(p) for p in (WORK/'vendor/p2_planning_sft').glob('*.py')},world_size=32,
        reference_data=dict(train=train.identity,diagnostic_dev=dev.identity,full_dev=full.identity),full_dev_count=len(full))
    os.environ['RANK']='0'
    result=validate_performance_resume(config,identity,WORK/'training')
    os.environ.pop('RANK',None)
    atomic_json(WORK/'PREFLIGHT.json',dict(passed=True,train_count=len(train),dev_count=len(dev),migration=result,at=time.time()))
    status('preflight_passed',migration=result)

def stop_parent():
    code=r'''
import importlib.util,json,time
from pathlib import Path
p=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_planning_sft_upright_3b_gpu7_20261105')
w=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106')
spec=importlib.util.spec_from_file_location('owned_parent_watch',p/'monitor/driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.frozen_ok();before=m.snapshot();m.save(w/'PARENT_BEFORE_STOP.json',before)
(p/'monitor/STOP').write_text('User requested migration to four-node 32-GPU continuation.\n')
for path in [p/'monitor/process.json',p/'QUEUE_LAUNCH.json',p/'training_process.json']:
 record=m.read(path,{})
 if m.owned(record):m.stop_verified_group(record)
assert not m.owned(m.read(p/'training_process.json',{}))
assert not m.owned(m.read(p/'QUEUE_LAUNCH.json',{}))
assert not m.owned(m.read(p/'monitor/process.json',{}))
record=dict(stage='migrated_to_32_gpu',at=time.time(),new_work=str(w),parent_checkpoint=str(p/'training/checkpoint-1500'),last_in_memory_step=before['step'],recompute_unsaved_steps_after=1500)
m.save(p/'MIGRATED.json',record);m.save(p/'QUEUE_STATUS.json',record);m.save(w/'PARENT_STOPPED.json',record)
print(json.dumps(record),flush=True)
'''
    cmd=['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=no','root@172.29.104.237',shlex.join([str(PY),'-u','-c',code])]
    subprocess.run(cmd,check=True,timeout=55)
    status('parent_stopped_checkpoint_preserved')

def launch(cluster_name,resume=None):
    cmd=[str(PY),str(WORK/'vendor/cluster_launcher.py'),'--config',str(WORK/'config.json'),'--cluster',str(WORK/cluster_name),'--launch']
    if resume:cmd+=['--resume',str(resume)]
    phase='canary' if cluster_name=='cluster_canary.json' else 'full'
    with (WORK/(phase+'.log')).open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()
    atomic_json(WORK/(phase+'_process.json'),dict(pid=p.pid,start_ticks=stat[19],command=cmd,started_at=time.time()))
    status(phase+'_launched',launcher_pid=p.pid,resume=str(resume) if resume else 'parent-1500')
    while p.poll() is None:time.sleep(5)
    if p.returncode:raise RuntimeError(phase+' cluster exited '+str(p.returncode)+'; inspect '+phase+'.log')
    status(phase+'_exited_successfully')

def audit_canary():
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    out=WORK/'training';end=read(out/'STOPPED.json');assert end['global_step']==1503 and end['smoke'] is False
    assert read(out/'performance_load.json')['adapter_tensors_equal']
    comm=[read(out/'communication'/f'rank-{r:02d}.json') for r in range(32)]
    assert {x['rank'] for x in comm}==set(range(32)) and all(x['nccl_sum']==496.0 for x in comm)
    assert sorted(collections.Counter(x['host'] for x in comm).values())==[8,8,8,8]
    assert len({x['gpu_uuid'] for x in comm})==32
    samplers=[read(out/'sampler_audit'/f'rank-{r:02d}.json') for r in range(32)]
    assert len({x['order_sha256'] for x in samplers})==1
    assert all(x['examples']==704368 and x['batches']==2752 and not x['tail_padding'] for x in samplers)
    for r in range(32):
        g=read(out/'gradient_audits'/f'rank-{r:02d}.json');o=read(out/'optimizer_restore'/f'rank-{r:02d}.json');rng=read(out/'rng_restore'/f'rank-{r:02d}.json')
        assert g['finite'] and g['norm']>0 and not g['frozen_gradient'] and g['trainable_tensors']==504 and g['world_size']==32
        assert o['optimizer_steps']==[1500] and o['optimizer_state_entries']==504 and o['scheduler_last_epoch']==1500
        assert rng['exact_to_migration_payload'] and rng['step']==1500
    ck=out/'checkpoint-1503';state=read(ck/'trainer_state.json');assert state['global_step']==1503 and state['max_steps']==2752
    names=None;count=0
    with safe_open(str(ck/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        names=set(f.keys());assert len(names)==504
        for n in names:
            t=f.get_tensor(n);assert torch.isfinite(t).all();count+=t.numel()
    assert count==59867136
    opt=torch.load(ck/'optimizer.pt',map_location='cpu',weights_only=True);assert len(opt['state'])==504
    for v in opt['state'].values():
        assert int(v['step'])==1503 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
    assert torch.load(ck/'scheduler.pt',map_location='cpu',weights_only=True)['last_epoch']==1503
    files={}
    for p in ck.iterdir():
        if p.is_file():
            st=p.stat();h=sha(p);assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
            files[p.name]=dict(bytes=st.st_size,sha256=h)
    assert all(f'rng_state_{r}.pth' in files for r in range(32))
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,step=1503,world_size=32,four_hosts=sorted({x['host'] for x in comm}),
        all_32_gradients_finite=True,optimizer_and_scheduler_restored_from_1500=True,
        all_32_rng_restored=True,all_rank_sample_orders_equal=True,adapter_tensors=504,
        checkpoint_files=files,at=time.time())
    atomic_json(WORK/'CANARY_ACCEPTED.json',result,immutable=True)
    (WORK/'canary').mkdir(exist_ok=True);(out/'STOPPED.json').rename(WORK/'canary/STOPPED.json')
    status('canary_accepted',step=1503,world_size=32)
    return ck

def main():
    preflight();stop_parent();launch('cluster_canary.json');checkpoint=audit_canary();launch('cluster.json',checkpoint)
    marker=read(WORK/'training/COMPLETE.json');assert marker['global_step']==2752 and marker['smoke'] is False and marker['validation_enabled'] is False
    assert (WORK/'training/final_adapter/adapter_model.safetensors').is_file()
    atomic_json(WORK/'COMPLETE.json',dict(success=True,global_step=2752,world_size=32,native_complete_sha256=sha(WORK/'training/COMPLETE.json'),at=time.time()))
    status('complete',global_step=2752)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(WORK/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));status('failed',error=repr(exc));raise
