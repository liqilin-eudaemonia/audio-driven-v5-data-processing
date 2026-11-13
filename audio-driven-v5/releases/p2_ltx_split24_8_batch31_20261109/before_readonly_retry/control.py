"""Owned state-preserving24/8 handoff, leaving foreign vLLM processes untouched."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib,importlib.util,json,os,subprocess,sys,time,traceback
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts')]
from gate import read,write,record,request_stop
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
SELF=Path(__file__).resolve();OLD=ROOT/'experiments/p2_ltx_swap6000_20261109'
spec=importlib.util.spec_from_file_location('legacy_resource_swap',OLD/'swap.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
sha=legacy.sha;equal_tree=legacy.equal_tree;alive=legacy.alive;remote=legacy.remote
ARMS={'full':8,'noevent':24}

def protocol():return read(WORK/'PROTOCOL.json')
def path(arm,target=False):return Path(protocol()[arm+('_target'if target else'_parent')])
def native(arm,name):
    return importlib.import_module(('p2_ltx_all_lora'if arm=='full'else'p2_ltx_all_lora_noevent')+'.'+name)
def activate():legacy.activate()
def emit(stage,**kw):
    value=dict(at=time.time(),stage=stage,**kw);write(WORK/'STATE.json',value)
    with(WORK/'events.jsonl').open('a')as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)
def target_identity(arm,parent,c,pre):
    import runtime
    if parent['world']!=(31 if arm=='full'else 1):raise ValueError('Wrong parent world')
    before=copy.deepcopy(c);before['output']=parent['experiment']['output']
    if before!=parent['experiment']or c['output']==before['output']or c['training']['global_batch']!=31:
        raise ValueError('Model/data/loss/sample budget changed')
    if pre['data_identity']!=parent['data_identity']or pre['statistics']!=parent['statistics']:
        raise ValueError('Data/labels/statistics changed')
    expected=runtime.source_identity(arm)
    if any(expected.get(k)!=v for k,v in parent['source'].items())or pre['source']!=expected:
        raise ValueError('Frozen source differs')
    if set(expected)-set(parent['source'])!={str(WORK/n)for n in ['runtime.py','worker.py','training_full.py','training_noevent.py']}:
        raise ValueError('Unexpected source extension')
    result=copy.deepcopy(parent);result.update(experiment=c,world=ARMS[arm],source=expected)
    return result

def prepare():
    activate();import runtime
    from p2_listen.contracts import digest
    records={};plans=[]
    for arm,world in ARMS.items():
        parent=path(arm);out=path(arm,True);out.mkdir(exist_ok=True);(out/'rank_logs').mkdir(exist_ok=True)
        c=copy.deepcopy(read(parent/'config.json'));c['output']=str(out);native(arm,'config').validate(c)
        write(out/'config.json',c)
        cl=copy.deepcopy(read(path('full')/'cluster.json'))
        cl.update(training_script=str(WORK/'worker.py'),training_config=str(out/'config.json'),master_port=29918 if arm=='full'else 29917)
        cl['nodes']=[n for n in cl['nodes']if(n['name']=='node11')==(arm=='full')]
        for i,n in enumerate(cl['nodes']):n.update(gpus=list(range(8)),role='master'if i==0 else'slave')
        write(out/'cluster.json',cl)
        pre=native(arm,'training').preflight(c);pre['source']=runtime.source_identity(arm);write(out/'preflight.json',pre)
        parent_identity=read(parent/'run.json');target_identity(arm,parent_identity,c,pre)
        for f,h in parent_identity['source'].items():
            if sha(f)!=h:raise ValueError('Parent code changed: '+f)
        schedule=native(arm,'schedule').Schedule(c);plans.append(schedule.pairs)
        for step in range(1,c['training']['steps']+1):
            old=schedule.indices(schedule.size,step,31,c['training']['seed'],0,1)
            flat=[pair for r in range(world)for pair in runtime.partition(old,r,world)]
            if flat!=old:raise ValueError('Missing/replayed sample or noise ordinal')
        records[arm]=dict(output=str(out),world=world,config_sha256=sha(out/'config.json'),cluster_sha256=sha(out/'cluster.json'),
            preflight_sha256=sha(out/'preflight.json'),parent_source_files=len(parent_identity['source']),canary_reads=len(pre['canary_reads']),
            all9169_steps_schedule_equal=True,local_counts=[len(runtime.partition(list(range(31)),r,world))for r in range(world)])
    if plans[0]!=plans[1]:raise ValueError('Arm plans differ')
    source={p.name:sha(p)for p in WORK.glob('*.py')}
    ready=dict(success=True,at=time.time(),targets=records,source=source,protocol_sha256=sha(WORK/'PROTOCOL.json'),effective_batch=31,same_final500h_latest_caption=True)
    write(WORK/'CODE_READY.json',ready);return ready

def source_ready():
    r=read(WORK/'CODE_READY.json')
    if not r['success']or any(sha(WORK/k)!=v for k,v in r['source'].items()):raise ValueError('Split code changed')
    if sha(WORK/'PROTOCOL.json')!=r['protocol_sha256']:raise ValueError('Approved split protocol changed')
    for arm,v in r['targets'].items():
        for file in ('config','cluster','preflight'):
            if sha(path(arm,True)/(file+'.json'))!=v[file+'_sha256']:raise ValueError('Reviewed target changed')

def rewrite(arm,source,destination,identity,parent_identity):
    import torch
    from p2_listen.cache import atomic_torch
    cp=native(arm,'checkpoint');v=cp.checked(source)
    if v['identity']!=parent_identity or len(v['rank_rng'])!=parent_identity['world']:raise ValueError('Wrong parent checkpoint')
    if source.resolve()==destination.resolve()or destination.parent.resolve()!=Path(identity['experiment']['output']).resolve()or source.name!=destination.name:
        raise ValueError('Must preserve source and step')
    normalized=copy.deepcopy(identity);normalized.update(world=parent_identity['world'],source=parent_identity['source'])
    normalized['experiment']['output']=parent_identity['experiment']['output']
    if normalized!=parent_identity or identity['world']!=ARMS[arm]:raise ValueError('Unexpected migration change')
    for state in v['optimizer']['state'].values():
        for t in state.values():
            if torch.is_tensor(t)and not torch.isfinite(t).all():raise ValueError('Nonfinite Adam')
    payload=dict(v,identity=identity,rank_rng=[copy.deepcopy(v['rank_rng'][0])for _ in range(identity['world'])])
    atomic_torch(destination,payload);write(destination.with_suffix('.complete.json'),dict(step=v['step'],sha256=sha(destination)))
    new=cp.checked(destination)
    if not equal_tree(v['model'],new['model'])or not equal_tree(v['optimizer'],new['optimizer']):raise ValueError('Model/Adam changed')
    if new['identity']!=identity or new['step']!=v['step']:raise ValueError('Wrong migrated state')
    return dict(success=True,at=time.time(),source=str(source),source_sha256=sha(source),destination=str(destination),destination_sha256=sha(destination),
        step=v['step'],parent_world=parent_identity['world'],world=identity['world'],model_and_adam_equal=True,
        rng_policy='Copy parent rank0 startup state; unchanged per-sample ordinal seeds control subsequent draws',original_checkpoint_preserved=True)

def migrate(arm):
    activate();source_ready();parent=path(arm);out=path(arm,True);step=read(parent/'status.json')['step']
    if arm=='full'and step!=6000:raise ValueError('Full must stop exactly6000')
    if (out/'MIGRATION.json').exists():raise ValueError('Do not migrate twice into an existing successor')
    source=parent/f'step_{step:06d}.pt';c=native(arm,'config').load(out/'config.json');old=read(parent/'run.json')
    identity=target_identity(arm,old,c,read(out/'preflight.json'))
    proof=rewrite(arm,source,out/source.name,identity,old)
    write(out/'run.json',identity);write(out/'status.json',dict(step=step,budget=c['training']['steps'],complete=False))
    with(out/'metrics.jsonl').open('w')as f:
        for line in(parent/'metrics.jsonl').read_text().splitlines():
            if json.loads(line)['step']<=step:f.write(line+'\n')
    write(out/'MIGRATION.json',proof);return proof

def node(arm,index):
    source_ready();out=path(arm,True);cl=read(out/'cluster.json');n=cl['nodes'][index];logs=out/'rank_logs'
    offset=sum(len(n['gpus'])for n in cl['nodes'][:index]);procs=[];streams=[]
    try:
        for local,gpu in enumerate(n['gpus']):
            rank=offset+local;env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,n['gpus'])),RANK=str(rank),LOCAL_RANK=str(local),
                WORLD_SIZE=str(ARMS[arm]),LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']),
                P2_TRAIN_MEMORY_GIB='108'if arm=='full'and gpu!=0 else'134')
            cmd=[PY,'-u',str(WORK/'worker.py'),'--arm',arm,'--config',str(out/'config.json'),'--resume',read(out/'MIGRATION.json')['destination']]
            stream=(logs/f'rank{rank:02d}.log').open('a');streams.append(stream);procs.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT))
        write(logs/(n['name']+'_workers.json'),dict(node=n['name'],pids=[p.pid for p in procs],ranks=list(range(offset,offset+len(procs))),gpus=n['gpus']))
        while any(p.poll()is None for p in procs):
            if any(p.poll()not in(None,0)for p in procs):raise RuntimeError('Own worker failed')
            time.sleep(2)
    finally:
        for p in procs:
            if p.poll()is None:p.terminate()
        for p in procs:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()

def launch(arm):
    from listen_cluster import launch as cluster_launch
    source_ready();out=path(arm,True);cl=read(out/'cluster.json')
    cmds=[['env',*[k+'='+str(v)for k,v in cl['network'].items()],PY,'-u',str(SELF),'node','--arm',arm,'--node-index',str(i)]for i in range(len(cl['nodes']))]
    write(out/'COMMANDS.json',cmds);write(out/'CONTROLLER.json',record())
    cluster_launch(cl,cmds,out/('nodes_'+str(int(time.time()))))

def spawn(arm):
    with(WORK/(arm+'_launcher.log')).open('a')as f:
        p=subprocess.Popen([PY,'-u',str(SELF),'launch','--arm',arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()[19],at=time.time())
    write(WORK/(arm+'_LAUNCH_CONTROLLER.json'),r);return p

def probe(arm):
    from concurrent.futures import ThreadPoolExecutor
    out=path(arm,True);cl=read(out/'cluster.json')
    def one(index):
        n=cl['nodes'][index];offset=sum(len(v['gpus'])for v in cl['nodes'][:index])
        payload=dict(out=str(out),worker=str(WORK/'worker.py'),ranks=list(range(offset,offset+len(n['gpus']))),gpus=n['gpus'],world=ARMS[arm],node=n['name'])
        code='x='+repr(payload)+'\n'+r'''
import json,subprocess
from pathlib import Path
errors=[];workers=[]
gpus=[]
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
 parts=[v.strip()for v in line.split(',')];gpus.append(dict(gpu=int(parts[0]),uuid=parts[1],used_mib=float(parts[2]),free_mib=float(parts[3]),util=float(parts[4])))
apps=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 a=[v.strip()for v in line.split(',')];apps.append(dict(pid=int(a[0]),uuid=a[1],used_mib=float(a[2])))
for rank,gpu in zip(x['ranks'],x['gpus']):
 try:
  r=json.loads((Path(x['out'])/'rank_logs'/f'worker_{rank:02d}.json').read_text());pid=r['pid'];d=Path('/proc',str(pid))
  stat=(d/'stat').read_text().rsplit(')',1)[1].split();cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode();env=dict(s.split('=',1)for s in(d/'environ').read_bytes().decode().split('\0')if'='in s)
  assert stat[0]!='Z'and stat[19]==r['start_ticks']and x['worker']in cmd and x['out']+'/config.json'in cmd
  assert int(env['RANK'])==rank and int(env['WORLD_SIZE'])==x['world']and list(map(int,env['CUDA_VISIBLE_DEVICES'].split(',')))[int(env['LOCAL_RANK'])]==gpu
  uuid=next(v['uuid']for v in gpus if v['gpu']==gpu);assert any(a['pid']==pid and a['uuid']==uuid for a in apps)
  workers.append(r)
 except Exception as e:errors.append(dict(rank=rank,error=repr(e)))
own={r['pid']for r in workers};foreign=[]
for a in apps:
 if a['pid']in own:continue
 try:cmd=Path('/proc',str(a['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except OSError:cmd=''
 allowed=x['node']=='node11'and'VLLM' in cmd and a['used_mib']<34000
 foreign.append(dict(**a,command=cmd,allowed_vllm=allowed))
 if not allowed:errors.append(dict(foreign_pid=a['pid']))
print(json.dumps(dict(node=x['node'],workers=workers,gpus=gpus,foreign=foreign,errors=errors)))
'''
        return remote(cl,n,code,timeout=45)
    with ThreadPoolExecutor(len(cl['nodes']))as pool:nodes=list(pool.map(one,range(len(cl['nodes']))))
    return dict(at=time.time(),arm=arm,healthy=all(not n['errors']for n in nodes)and sum(len(n['workers'])for n in nodes)==ARMS[arm],nodes=nodes)

def wait_for(predicate,label,timeout=3600):
    begin=time.time()
    while time.time()-begin<timeout:
        if(WORK/'STOP').exists():raise RuntimeError('User stopped split controller')
        if predicate():return
        time.sleep(3)
    raise TimeoutError(label)

def old_gone():
    p=legacy.observation_module().probe();s=legacy.single_probe(path('noevent'))
    write(WORK/'OLD_EXIT_PROBE.json',dict(full=p,noevent=s))
    return not any(n['workers']for n in p['nodes'])and not s['workers']

def run():
    activate();source_ready()
    with(WORK/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(WORK/'CONTROLLER.json',record())
        if any((path(a,True)/'MIGRATION.json').exists()for a in ARMS):raise ValueError('Partial handoff requires explicit inspection, not blind rerun')
        emit('waiting_full5960')
        wait_for(lambda:read(path('full')/'status.json')['step']>=5960,'full5960',86400)
        emit('saving_noevent_boundary')
        request_stop(path('noevent')/'STOP',protocol()['owner'],reason='Approved24-card continuation after full6000')
        wait_for(lambda:not legacy.single_probe(path('noevent'))['workers'],'noEvent native save/exit',1800)
        emit('migrating_noevent1to24');no=migrate('noevent')
        emit('waiting_full6000')
        wait_for(lambda:(path('full')/'step_006000.complete.json').exists(),'full6000 complete',86400)
        wait_for(old_gone,'old groups exit',900)
        emit('checking_foreign_tasks')
        obs=legacy.observation_module()
        wait_for(lambda:obs.ready(obs.probe())and not legacy.single_probe(path('noevent'))['apps'],'approved GPU availability',1800)
        emit('launching_noevent24');a=spawn('noevent')
        emit('migrating_full31to8');full=migrate('full')
        emit('launching_full8');b=spawn('full')
        emit('validating_real_updates')
        for _ in range(360):
            if a.poll()is not None or b.poll()is not None:raise RuntimeError('Successor launcher exited during acceptance')
            nr=legacy.rows_after(path('noevent',True),no['step']);fr=legacy.rows_after(path('full',True),full['step'])
            legacy.check_rows(nr,False);legacy.check_rows(fr,True)
            if len(nr)>=3 and len(fr)>=3:
                ns=probe('noevent');fs=probe('full')
                if ns['healthy']and fs['healthy']:
                    canaries={arm:read(path(arm,True)/'LONGEST_UNEVEN_CHECK.json')for arm in ARMS}
                    if not all(v['success']and v['global_batch']==31 for v in canaries.values()):raise ValueError('Missing longest validation')
                    result=dict(success=True,at=time.time(),global_batch=31,noevent_step=nr[-1]['step'],full_step=fr[-1]['step'],
                        noevent_cluster=ns,full_cluster=fs,noevent_migration=no,full_migration=full,canaries=canaries,
                        noevent_seconds=[v['seconds']for v in nr],full_seconds=[v['seconds']for v in fr])
                    write(WORK/'SPLIT_ACCEPTED.json',result);emit('both_running_accepted',noevent_step=nr[-1]['step'],full_step=fr[-1]['step']);break
            time.sleep(10)
        else:raise TimeoutError('Real update acceptance')
        while a.poll()is None or b.poll()is None:
            for arm,child in [('noevent',a),('full',b)]:
                if child.poll()not in(None,0):
                    write(WORK/(arm+'_TRAINING_ERROR.json'),dict(at=time.time(),returncode=child.returncode));raise RuntimeError('Owned successor failed; other group untouched')
            time.sleep(30)
        emit('both_launchers_exited')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','migrate','node','launch','run','probe'])
    p.add_argument('--arm',choices=list(ARMS));p.add_argument('--node-index',type=int);a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare()))
    elif a.command=='migrate':print(json.dumps(migrate(a.arm)))
    elif a.command=='node':node(a.arm,a.node_index)
    elif a.command=='launch':launch(a.arm)
    elif a.command=='probe':print(json.dumps(probe(a.arm)))
    else:
        try:run()
        except Exception as e:
            write(WORK/f'ERROR_{os.getpid()}.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
