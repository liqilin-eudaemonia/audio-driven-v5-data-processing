"""One owned complete save, exact32->31 continuation, then native joint training."""
from pathlib import Path
import os,sys,time,json,copy,shutil,subprocess,runpy,traceback,signal,csv,statistics
from collections import Counter
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_ltx_events.config import source_identity,validate_baseline
from p2_ltx_events.data import data_identity
from p2_ltx_events.training import check_preflight
from p2_ltx_events.resume31 import remaining_plan,checkpoint_due
from p2_ltx_events.schedule import Schedule
from p2_listen.cache import atomic_torch
from listen_cluster import launch
import p2_ltx_resume31 as runtime
OLD=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'
OUT=Path(__file__).resolve().parent
PARENT=read(OUT/'PARENT_IDENTITY.json')


def state(stage,**fields):
    record=dict(stage=stage,at=time.time(),controller_pid=os.getpid(),**fields)
    atomic_json(OUT/'switch_state.json',record);print(json.dumps(record),flush=True)


def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def configuration(step,directory):
    parent=read(OLD/'config.json')
    if parent!=PARENT['experiment']:raise ValueError('Parent config changed')
    changed={str(ROOT/('src/p2_ltx_events/'+name)) for name in ('config.py','schedule.py','training.py')}
    for path,expected in PARENT['source'].items():
        actual=OUT/'source_before'/Path(path).relative_to(ROOT) if path in changed else Path(path)
        if sha(actual)!=expected:raise ValueError('Parent source archive changed: '+path)
    parent_pairs=read(parent['duration_sampling']['manifest'])['pairs']
    with (Path(parent['current_source'])/'sample_list.csv').open()as stream:
        durations=[float(row['duration_s']) for row in csv.DictReader(stream)]
    plan=remaining_plan(parent_pairs,step,32,31,durations,parent['training']['seed'])
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'resume.schedule.json';atomic_json(manifest,plan)
    c=copy.deepcopy(parent);c['output']=str(OUT)
    c['training'].update(global_batch=31,steps=step+len(plan['pairs'])//31,save_every=500)
    c['checkpoint_schedule']=dict(threshold_step=5500,before_every=1000,after_every=500)
    c['world31_execution']=dict(kind='resume31-preserve-remaining-v1',world=31,reserved_node='node11',reserved_gpu=7)
    c['resume_execution']=dict(kind='preserve-parent-suffix-v1',parent_config=str(OLD/'config.json'),
        parent_config_sha256=sha(OLD/'config.json'),resume_step=step,manifest=str(manifest),
        manifest_sha256=sha(manifest),world=31,parent_world=32)
    validate_baseline(c,read(c['hidden_event_parent_config']))
    did=data_identity(c);assert did==PARENT['data_identity']
    source=source_identity();pre=copy.deepcopy(read(OLD/'preflight.json'))
    pre.update(configuration_sha256=digest(c),source=source)
    check_preflight(c,pre,source,did)
    sampler=Schedule(c);reconstructed=[]
    for s in range(step+1,c['training']['steps']+1):
        for rank in range(31):reconstructed.extend(sampler.indices(sampler.size,s,31,c['training']['seed'],rank,31))
    original=[tuple(p) for p in parent_pairs[step*32:]]
    original_ordinals={p[1] for p in original}
    retained=[p for p in reconstructed if p[1]in original_ordinals]
    assert Counter(retained)==Counter(original) and len(reconstructed)==len(plan['pairs'])
    assert len({p[1] for p in reconstructed})==len(reconstructed)
    cluster=copy.deepcopy(read(OLD/'cluster.json'));cluster['master_port']=29842
    cluster.update(training_script=str(ROOT/'scripts/p2_ltx_resume31.py'),training_config=str(OUT/'config.json'))
    cluster['nodes'][-1]['gpus']=list(range(7));runtime.layout(cluster)
    identity=copy.deepcopy(PARENT);identity.update(experiment=c,source=source,world=31,data_identity=did)
    return c,pre,cluster,identity,plan


def probes(cluster):
    module=runpy.run_path(str(ROOT/'scripts/p2_train32_guard9h_20261106.py'),run_name='probe_functions_only')
    return module['probes'](cluster)


def monitor(cluster):
    text=(OLD/'monitor.py').read_text().replace('32-GPU','31-GPU')
    text=text.replace("i,m,u,p,c=fields;cards.append(", "i,m,u,p,c=fields\n   if int(i) not in node['gpus']:continue\n   cards.append(")
    text=text.replace("len(n.get('gpus',[]))!=8", "len(n.get('gpus',[]))!=len(next(x for x in cluster['nodes'] if x['name']==n['node'])['gpus'])")
    text=text.replace('32/max(mean','31/max(mean').replace("32*(step-baseline[1])","31*(step-baseline[1])")
    compile(text,str(OUT/'monitor.py'),'exec');(OUT/'monitor.py').write_text(text)
    with (OUT/'monitor.log').open('ab',buffering=0)as f:
        p=subprocess.Popen([cluster['python'],'-u',str(OUT/'monitor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(OUT/'MONITOR.json',dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],at=time.time()))


def tensorboard(cluster):
    old=read(OLD/'TENSORBOARD_SERVER.json');p=Path('/proc',str(old['pid']))
    if p.exists():
        s=p.joinpath('stat').read_text().split();args=p.joinpath('cmdline').read_bytes().split(b'\0')
        if s[2]!='Z' and s[21]==old['start_ticks'] and b'tensorboard.main' in args and str(OLD/'tensorboard').encode()in args:
            os.kill(old['pid'],signal.SIGTERM);time.sleep(1)
    command=[cluster['python'],'-m','tensorboard.main','--logdir',str(OUT/'tensorboard'),'--host','0.0.0.0',
             '--port','6006','--reload_interval','10','--load_fast','false','--samples_per_plugin','scalars=12000']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OUT/'tensorboard_server.log').open('ab',buffering=0)as f:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    atomic_json(OUT/'TENSORBOARD_SERVER.json',dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],
        at=time.time(),port=6006,logdir=str(OUT/'tensorboard'),private_url='http://172.29.252.37:6006',
        loopback_url='http://127.0.0.1:6006',command=command,gpu_devices=''))


def main():
    import torch
    torch.set_num_threads(4)
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        step=read(OLD/'status.json')['step']
        c,pre,cluster,identity,plan=configuration(step,OUT/'dryrun')
        atomic_json(OUT/'CPU_VALIDATION.json',dict(success=True,at=time.time(),preview_resume_step=step,
            world=31,global_batch=31,data_identity_unchanged=True,unconsumed_sample_noise_pairs_exact=True,
            duration_rebatched=True,
            padding=plan['padding'],new_total_steps=c['training']['steps'],source_files=len(identity['source']),
            checkpoint_steps=[s for s in range(step+1,c['training']['steps']+1) if checkpoint_due(c,s)],
            reserved_node='node11',reserved_gpu=7))
        state('prepared',preview_step=step,steps=c['training']['steps']);return
    if not read(OUT/'CPU_VALIDATION.json')['success']:raise ValueError('CPU acceptance required')
    reuse=len(sys.argv)>1 and sys.argv[1]=='from-saved'
    if (OUT/'STOP').exists():raise ValueError('Explicit child STOP preserved')
    before_step=read(OLD/'status.json')['step']
    if reuse:
        request=read(OLD/'STOP')
        if request.get('owner')!='user31-cadence-switch':raise ValueError('Unowned parent STOP')
        began=request['at']
    else:
        if (OLD/'STOP').exists():raise ValueError('Explicit parent STOP preserved')
        began=time.time()
        token=json.dumps(dict(owner='user31-cadence-switch',pid=os.getpid(),at=began))
        with (OLD/'STOP').open('x')as f:f.write(token)
    state('waiting_complete_save',observed_step=before_step)
    until=time.time()+600;old_cluster=read(OLD/'cluster.json')
    while True:
        step=read(OLD/'status.json')['step'];marker=OLD/f'step_{step:06d}.complete.json'
        if marker.exists():
            observations=probes(old_cluster)
            if all('error' not in n and not n.get('workers') for n in observations):break
        if time.time()>until:raise TimeoutError('Graceful save/worker exit timed out; no unrelated process signalled')
        time.sleep(3)
    state('migrating_complete_state',step=step)
    c,pre,cluster,identity,plan=configuration(step,OUT)
    from p2_ltx_events.checkpoint import checked
    origin=OLD/f'step_{step:06d}.pt';saved=checked(origin)
    assert saved['identity']==PARENT and saved['step']==step and len(saved['rank_rng'])==32
    target=OUT/origin.name;value=dict(saved,identity=identity,rank_rng=saved['rank_rng'][:31])
    atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(step=step,sha256=sha(target)))
    loaded=checked(target)
    assert equal(saved['model'],loaded['model']) and equal(saved['optimizer'],loaded['optimizer'])
    assert equal(saved['rank_rng'][:31],loaded['rank_rng']) and loaded['identity']==identity
    atomic_json(OUT/'MIGRATION.json',dict(step=step,origin=str(origin),origin_sha256=read(origin.with_suffix('.complete.json'))['sha256'],
        world=31,global_batch=31,model_optimizer_bitwise_equal=True,all31_rng_bitwise_equal=True,
        parent_all32_rng_retained_in_original_checkpoint=True,data_loss_unchanged=True,
        unconsumed_sample_noise_pairs_exact=True,duration_rebatched=True,
        original_remaining=plan['original_remaining'],padding=plan['padding'],
        next_checkpoint=next(s for s in range(step+1,c['training']['steps']+1)if checkpoint_due(c,s)),
        checkpoint_schedule=c['checkpoint_schedule'],old_budget=8882,new_budget=c['training']['steps'],at=time.time()))
    del saved,value,loaded
    for name,content in [('config.json',c),('preflight.json',pre),('cluster.json',cluster),('run.json',identity)]:
        atomic_json(OUT/name,content,immutable=True)
    shutil.copy2(OLD/'metrics.jsonl',OUT/'metrics.jsonl')
    shutil.copytree(OLD/'tensorboard',OUT/'tensorboard',dirs_exist_ok=True)
    atomic_json(OUT/'status.json',dict(step=step,budget=c['training']['steps'],complete=False))
    selected=probes(old_cluster);atomic_json(OUT/'RESOURCES_BEFORE_LAUNCH.json',selected)
    for node,entry in zip(selected,cluster['nodes']):
        if 'error'in node or node.get('workers') or len(node.get('free_mib',[]))!=8:
            raise ValueError('Old workers still present or GPU query failed')
        if any(node['free_mib'][g]<108*1024 for g in entry['gpus']):
            raise ValueError('Selected training GPU has insufficient headroom; reservedGPU7 is excluded')
    atomic_json(OUT/'RESERVED_GPU.json',dict(node='node11',host='172.29.189.119',gpu=7,training_visible_gpus=list(range(7))))
    atomic_json(OUT/'QUEUE_STATE.json',dict(state='training',phase='startup',controller_pid=os.getpid(),resume=str(target),at=time.time()))
    commands=runtime.commands(cluster,OUT/'config.json',target,OUT/'rank_logs')
    atomic_json(OUT/'COMMANDS.json',commands);monitor(cluster);tensorboard(cluster)
    state('starting31',step=step,budget=c['training']['steps'],resume=str(target),switch_began=began)
    atomic_json(OUT/'SWITCH_TIMING.json',dict(began=began,parent_last_step=step,startup_began=time.time(),old_step_seconds=6.5))
    launch(cluster,commands,OUT/'launch')
    atomic_json(OUT/'QUEUE_STATE.json',dict(state='finished',controller_pid=os.getpid(),at=time.time()))
    state('finished')


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'SWITCH_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
        state('failed',error=traceback.format_exc());raise
