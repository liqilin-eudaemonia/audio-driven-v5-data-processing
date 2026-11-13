#!/usr/bin/env python3
"""Prepare only by default; every GPU stage requires an explicit subcommand."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for k in ('config','observations','jobs','work'):q.add_argument('--'+k,required=True,type=Path)
    for command in ('plan','cache','verify','worker'):
        q=sub.add_parser(command);q.add_argument('--work',required=True,type=Path)
        q.add_argument('--shard-index',type=int,default=0);q.add_argument('--shards',type=int,default=1)
        if command=='cache':q.add_argument('--part',choices=['text','input_plan'],required=True)
        if command=='worker':q.add_argument('--minimum-free-gib',type=float,default=120)
    q=sub.add_parser('generate');q.add_argument('--job',required=True,type=Path)
    q=sub.add_parser('metrics');q.add_argument('--work',required=True,type=Path);q.add_argument('--arm',required=True)
    a=p.parse_args()
    # Legacy inference must bootstrap its checkpoint-native package BEFORE any p2_listen import.
    if a.command=='generate':
        job=json.loads(a.job.read_text())
        if job['family']=='legacy':
            sys.path.insert(0,str(ROOT/'scripts'));from p2_eval_generate import bootstrap
            bootstrap(['--run',job['run']])
    sys.path.insert(0,str(ROOT/'src'))
    from p2_autonomous_eval.protocol import prepare,open_protocol,selected
    if a.command=='prepare':result=prepare(a.config,a.observations,a.jobs,a.work)
    elif a.command=='generate':
        from p2_autonomous_eval.generation import generate
        result=generate(job)
    elif a.command=='metrics':
        from p2_autonomous_eval.evaluation import evaluate
        result=evaluate(a.work,a.arm)
    elif a.command=='worker':
        from p2_listen.contracts import atomic_json,digest
        protocol=open_protocol(a.work);ids={o['sample_id'] for o in selected(protocol,a.shard_index,a.shards)}
        if not ids:raise ValueError('Empty worker shard')
        base=[sys.executable,str(Path(__file__).resolve())];common=['--work',str(a.work),'--shard-index',str(a.shard_index),'--shards',str(a.shards)]
        def run(args):
            import torch
            free,total=torch.cuda.mem_get_info(0)
            if free<a.minimum_free_gib*1024**3:raise RuntimeError('GPU headroom below threshold; do not overlap active training')
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            subprocess.run(base+args,env=env,check=True)
        run(['plan',*common]);run(['cache',*common,'--part','text']);run(['cache',*common,'--part','input_plan']);run(['verify',*common])
        result=dict(success=True,completed=[],waiting=[],failed=[])
        for job in protocol['jobs']:
            if job['sample_id'] not in ids:continue
            if not Path(job['checkpoint']).with_suffix('.complete.json').exists():
                result['waiting'].append(job);continue
            error=None
            for attempt in range(3):
                try:run(['generate','--job',str(a.work/'jobs'/(digest(job)+'.json'))]);error=None;break
                except (subprocess.CalledProcessError,RuntimeError) as e:error=str(e)
            if error:result['failed'].append(dict(job=job,error=error))
            else:result['completed'].append(job)
            atomic_json(a.work/'reports'/f'worker.{a.shard_index}-of-{a.shards}.json',result)
        result['success']=not result['failed'] and not result['waiting']
        atomic_json(a.work/'reports'/f'worker.{a.shard_index}-of-{a.shards}.json',result)
    else:
        from p2_autonomous_eval import runtime
        kwargs=dict(shard_index=a.shard_index,shards=a.shards)
        if a.command=='cache':kwargs['part']=a.part
        result=getattr(runtime,a.command)(a.work,**kwargs)
    print(json.dumps(result,ensure_ascii=False))
    if result.get('success') is False:raise SystemExit(2)
if __name__=='__main__':main()
