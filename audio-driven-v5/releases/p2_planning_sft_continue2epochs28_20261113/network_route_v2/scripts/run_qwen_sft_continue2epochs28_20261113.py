"""Continue completed upright planning SFT by two epochs on 8+8+8+4 GPUs."""
import argparse,concurrent.futures,fcntl,json,os,shlex,signal,socket,struct,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_planning_sft_continue2epochs28_20261113'
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import atomic_json,read

def remote(c,n,cmd):
    if n['host'] in subprocess.check_output(['hostname','-I'],text=True).split():
        return ['bash','-c',cmd]
    return ['ssh','-i',n.get('ssh_key',c['ssh_key']),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
            '-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+c['known_hosts'],
            '-o','HostKeyAlias='+n['host'],n['ssh_user']+'@'+n.get('ssh_host',n['host']),cmd]

def probe(c,n):
    last=None
    for attempt in range(4):
        try:
            r=subprocess.run(remote(c,n,'nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits'),capture_output=True,text=True,timeout=25)
            if r.returncode:raise RuntimeError(r.stderr.strip())
            free={int(s.split(',')[0]):float(s.split(',')[1])/1024 for s in r.stdout.splitlines()}
            assert all(free[g]>=c['minimum_free_gib'] for g in n['gpus']), (n['name'],free)
            return dict(node=n['name'],gpus=n['gpus'],free_gib=free,at=time.time())
        except Exception as e:last=e;time.sleep(2)
    raise RuntimeError(str(last))

def node_main(c,index,resume):
    n=c['nodes'][index];logdir=WORK/'logs';logdir.mkdir(exist_ok=True)
    state=WORK/(n['name']+'_process.json');offset=sum(len(x['gpus']) for x in c['nodes'][:index]);children=[]
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for local,gpu in enumerate(n['gpus']):
            env=dict(os.environ,**{k:str(v) for k,v in c['network'].items()})
            if c.get('control_subnet'):
                sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                interfaces=[]
                for _,name in socket.if_nameindex():
                    try:address=socket.inet_ntoa(fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',name.encode()))[20:24])
                    except OSError:continue
                    if address.startswith(c['control_subnet']):interfaces.append(name)
                sock.close();assert len(interfaces)==1, interfaces
                env.update(NCCL_SOCKET_IFNAME=interfaces[0],GLOO_SOCKET_IFNAME=interfaces[0])
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,n['gpus'])),RANK=str(offset+local),LOCAL_RANK=str(local),
                       WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=c.get('master_address',c['nodes'][0]['host']),MASTER_PORT=str(c['master_port']),
                       TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
            cmd=[str(PY),'-u',str(WORK/'vendor/train_entry.py'),'--config',str(WORK/'config.json')]
            if resume:cmd+=['--resume',str(resume)]
            with (logdir/f'rank-{offset+local:02d}.log').open('a') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        while True:
            codes=[p.poll() for p in children]
            atomic_json(state,dict(pid=os.getpid(),node=n['name'],worker_pids=[p.pid for p in children],codes=codes,at=time.time()))
            if (WORK/'STOP').exists():raise RuntimeError('Own run STOP marker')
            if any(x not in (None,0) for x in codes):raise RuntimeError('Own worker failed: '+str(codes))
            if all(x==0 for x in codes):break
            time.sleep(5)
        atomic_json(WORK/(n['name']+'_COMPLETE.json'),dict(success=True,at=time.time()))
    except BaseException:
        atomic_json(WORK/(n['name']+'_ERROR.json'),dict(error=traceback.format_exc(),at=time.time()))
        (WORK/'STOP').write_text('A worker/node of this run failed. Inspect logs.\n')
        raise
    finally:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)

def launch(c,resume):
    assert not (WORK/'STOP').exists(), 'Inspect/remove own STOP before explicit restart'
    assert not any(WORK.glob('node*_ERROR.json')), 'Inspect archived errors before restart'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        preflight=list(pool.map(lambda n:probe(c,n),c['nodes']))
    atomic_json(WORK/'GPU_PREFLIGHT.json',dict(nodes=preflight,at=time.time()))
    atomic_json(WORK/'CONTROLLER.json',dict(pid=os.getpid(),command=sys.argv,at=time.time()))
    for index,n in enumerate(c['nodes']):
        args=[str(PY),'-u',str(Path(__file__).resolve()),'--node',str(index)]
        if resume:args+=['--resume',str(resume)]
        cmd='nohup '+shlex.join(args)+' > '+shlex.quote(str(WORK/(n['name']+'_supervisor.log')))+' 2>&1 < /dev/null &'
        for attempt in range(4):
            r=subprocess.run(remote(c,n,cmd),capture_output=True,text=True,timeout=25)
            if r.returncode==0:break
            if (WORK/(n['name']+'_process.json')).exists():break
            time.sleep(2)
        else:
            (WORK/'STOP').write_text('Own node launch failed.\n');raise RuntimeError(n['name']+': '+r.stderr)
    while True:
        errors=list(WORK.glob('node*_ERROR.json'))
        if errors:raise RuntimeError('Own node failure: '+str(errors))
        done=list(WORK.glob('node*_COMPLETE.json'))
        atomic_json(WORK/'STATUS.json',dict(stage='complete' if len(done)==4 else 'running',at=time.time(),completed_nodes=len(done)))
        if len(done)==4:
            marker=read(WORK/'training/COMPLETE.json');assert marker['global_step']==8256
            atomic_json(WORK/'COMPLETE.json',dict(success=True,global_step=8256,additional_epochs=2,at=time.time()));return
        time.sleep(15)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',type=int);p.add_argument('--resume');a=p.parse_args()
    c=read(WORK/'cluster.json')
    if a.node is not None:node_main(c,a.node,a.resume)
    else:
        try:launch(c,a.resume)
        except BaseException:
            atomic_json(WORK/'ERROR.json',dict(error=traceback.format_exc(),at=time.time()));raise
