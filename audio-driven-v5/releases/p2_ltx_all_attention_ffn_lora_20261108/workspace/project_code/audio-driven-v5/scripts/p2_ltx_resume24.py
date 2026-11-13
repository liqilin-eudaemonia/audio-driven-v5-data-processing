"""Explicit24-rank launcher; node11 is absent from every training command."""
import argparse,os,runpy,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json


def layout(cluster):
    if [n['name'] for n in cluster['nodes']]!=['node12','node8','node4']:
        raise ValueError('Only node12/node8/node4 may train; reserve node11')
    result=[]
    for i,node in enumerate(cluster['nodes']):
        if node['gpus']!=list(range(8)):raise ValueError('Require all8 GPUs on each selected node')
        result.append(dict(node=node['name'],gpus=node['gpus'],ranks=list(range(i*8,i*8+8))))
    return result


def commands(cluster,config,resume,attempt):
    layout(cluster)
    return [['env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','PYTHONUNBUFFERED=1',
             'TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cluster['network'].items()],
             cluster['python'],str(SELF),'node','--config',str(config),'--node-index',str(i),
             '--resume',str(resume),'--attempt',str(attempt)] for i in range(3)]


def node(c,index,resume,attempt):
    cluster=read(Path(c['output'])/'cluster.json');mapping=layout(cluster)
    if index not in range(3):raise ValueError('Invalid selected node')
    entry=mapping[index]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3,4,5,6,7':
        raise ValueError('Unexpected physical GPU visibility')
    attempt.mkdir(parents=True,exist_ok=True);processes=[];streams=[]
    try:
        for local,rank in enumerate(entry['ranks']):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='24',LOCAL_WORLD_SIZE='8',
                     MASTER_ADDR=cluster['nodes'][0]['host'],MASTER_PORT=str(cluster['master_port']))
            stream=(attempt/f'rank{rank:02d}.log').open('a');streams.append(stream)
            cmd=[cluster['python'],str(SELF),'train','--config',str(Path(c['output'])/'config.json'),
                 '--resume',str(resume)]
            processes.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT))
        atomic_json(attempt/(entry['node']+'_workers.json'),dict(node=entry['node'],gpus=entry['gpus'],
                    ranks=entry['ranks'],pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Local distributed rank failed')
            time.sleep(2)
        if any(p.returncode for p in processes):raise RuntimeError('Distributed training failed')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for s in streams:s.close()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='train':
        from p2_whisper_av.bootstrap import activate_ltx
        activate_ltx()
        return runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='native_ltx_resume24')['main']()
    p=argparse.ArgumentParser();p.add_argument('command',choices=['node'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--node-index',type=int,required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();return node(read(a.config),a.node_index,a.resume,a.attempt)


if __name__=='__main__':main()
