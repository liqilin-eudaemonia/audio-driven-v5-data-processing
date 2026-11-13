"""32-GPU AV-only correctness, memory and recomputation benchmark. No formal training."""
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p2_listen.contracts import read, sha, atomic_json
from p2_j_av_only import DEFAULT, CLUSTER, REFERENCE, install_backend, commands
from p2_j_duration_benchmark import equal_tensors
from listen_cluster import launch, transport

WORK = ROOT/'experiments/p2_j_av_only_benchmark_20261101'


def main():
    import torch
    from p2_listen.cache import atomic_torch
    from p2_matrix16.checkpoint import checked
    torch.set_num_threads(4)
    WORK.mkdir(parents=True, exist_ok=True)
    cluster = read(CLUSTER); base = read(DEFAULT); backend = install_backend()
    def status(stage, **kw):
        value = dict(stage=stage, at=time.time(), **kw)
        atomic_json(WORK/'status.json', value); print(json.dumps(value), flush=True)
    def free(node):
        s = subprocess.check_output(transport(cluster,node,['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits']), text=True, timeout=30)
        values = [int(x) for x in s.splitlines() if x.strip()]
        return len(values)==8 and min(values)>135*1024
    status('waiting_checkpoint4200_and_release')
    deadline=time.time()+900
    while True:
        if time.time()>deadline:raise TimeoutError('AV+Event checkpoint/release timeout')
        state=read(ROOT/'experiments/p2_j_av_event_resume_after_lora2_20261101/state.json')
        if state['stage']=='stopped' and (REFERENCE/'step_004200.complete.json').exists():
            with ThreadPoolExecutor(max_workers=4) as pool:
                if all(pool.map(free,cluster['nodes'])):break
        time.sleep(3)
    last=read(REFERENCE/'status.json')['step']; stopped=REFERENCE/f'step_{last:06d}.pt'
    paused=checked(stopped)
    atomic_json(WORK/'ORIGINAL_SAVED.json',dict(step=last,checkpoint=str(stopped),sha256=sha(stopped)))
    del paused
    initial_path=ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_20261101/step_000000.pt'
    initial=checked(initial_path)
    assert initial['step']==0 and not initial['optimizer']['state']
    assert initial['identity']['data_identity']==backend.data_identity(base)
    pre=read(Path(base['output'])/'preflight.json')
    source=backend.source_identity()
    assert source==pre['source']
    atomic_json(WORK/'BENCHMARK_ONLY.json',dict(initial=str(initial_path),
        initial_sha256=sha(initial_path),formal_steps_counted=0,
        note='All runs begin with the same step0 model and empty optimizer. Offset100 is synthetic for testing the actual aligned sample schedule.'))
    sequence=0
    def prepare(name,policy,start):
        c=copy.deepcopy(base);c['output']=str(WORK/name);c['training']['gradient_checkpointing']=policy
        path=WORK/(name+'.json');atomic_json(path,c,immutable=True)
        identity=dict(initial['identity'],experiment=c,source=source)
        out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
        saved=dict(initial,step=start,identity=identity)
        checkpoint=out/f'step_{start:06d}.pt';atomic_torch(checkpoint,saved)
        atomic_json(checkpoint.with_suffix('.complete.json'),dict(step=start,sha256=sha(checkpoint)))
        atomic_json(out/'run.json',identity,immutable=True)
        atomic_json(out/'preflight.json',dict(pre,source=source),immutable=True)
        return c,path,checkpoint
    def execute(name,policy,start,count,segments=None):
        nonlocal sequence
        c,path,checkpoint=prepare(name,policy,start)
        for stop in segments or [start+count]:
            sequence+=1; cc=copy.deepcopy(cluster);cc['master_port']=29900+sequence
            status('benchmark',name=name,policy=policy,start=start,end=stop,world=32)
            launch(cc,commands(cc,path,resume=checkpoint,stop_after=stop),WORK/'transport'/f'{name}_{stop}')
            checkpoint=Path(c['output'])/f'step_{stop:06d}.pt'
        metrics=[json.loads(x) for x in (Path(c['output'])/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in metrics]==list(range(start+1,start+count+1))
        assert all(r['event']==0 and r['weighted_total']==r['av'] and r['grad_norm']['events']==0 for r in metrics)
        assert all(r['grad_norm']['connector']>0 and r['grad_norm']['text_lora']>0 for r in metrics)
        end=checked(checkpoint)
        assert all(torch.equal(v,end['model'][k]) for k,v in initial['model'].items() if k.startswith('event_heads.'))
        assert any(not torch.equal(v,end['model'][k]) for k,v in initial['model'].items() if k.startswith('connector.'))
        assert any(not torch.equal(v,end['model'][k]) for k,v in initial['model'].items() if k.startswith('text_lora.'))
        result=dict(name=name,policy=policy,start=start,count=count,
            mean_seconds=statistics.mean(r['seconds'] for r in metrics),
            warm_mean_seconds=statistics.mean(r['seconds'] for r in metrics[2:]),
            peak_reserved_gib=max(r['peak_reserved_gib'] for r in metrics),
            av_only_gradients_verified=True,checkpoint=str(checkpoint))
        atomic_json(WORK/(name+'.result.json'),result);return result
    results=[]
    # Longest11.95s sample occurs at actual step8; all32 GPUs participate.
    results.append(execute('alternating_longest','alternating',0,16))
    results.append(execute('all_aligned','all',100,32))
    results.append(execute('alternating_aligned','alternating',100,32))
    # Same sequence split across a real strict restore.
    results.append(execute('alternating_resume','alternating',100,32,[116,132]))
    left=checked(results[1]['checkpoint']);right=checked(results[2]['checkpoint'])
    policy_equal={k:equal_tensors(left[k],right[k]) for k in ('model','optimizer','rank_rng')}
    del left
    resumed=checked(results[3]['checkpoint'])
    resume_equal={k:equal_tensors(right[k],resumed[k]) for k in ('model','optimizer','rank_rng')}
    assert all(resume_equal.values()), resume_equal
    result=dict(success=True,world=32,formal_training_started=False,results=results,
        recomputation_bitwise_equal=policy_equal,strict_resume_bitwise_equal=resume_equal,
        time_reduction=1-results[2]['warm_mean_seconds']/results[1]['warm_mean_seconds'],
        selected='alternating' if all(policy_equal.values()) and results[2]['warm_mean_seconds']<results[1]['warm_mean_seconds'] else 'all')
    atomic_json(WORK/'RESULT.json',result);status('complete',result=result)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        WORK.mkdir(parents=True,exist_ok=True)
        atomic_json(WORK/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
