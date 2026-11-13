"""Paired new-seed rerun of two completed6000 models, node11 GPU0 only."""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
from p2_testsets_eval.runtime import ready
from p2_combined_whisper_eval import link

WORK=ROOT/'experiments/p2_seamless_whisper_joint_seed238_node11gpu0_20261108'
SEED=238
EXPORTS=ROOT/'experiments/p2_current_feature_whisper6000_eval32_20261108'
PARENTS={
    'A':ROOT/'experiments/p2_current500h_whisper6000_sft2752_dual200_20261108',
    'B':ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107',
}


def items_for_seamless(parent):
    p=read(parent/'protocol.json');assert sha(parent/'shared_items.json')==p['shared_items_sha256']
    items=[x for x in read(parent/'shared_items.json') if x['dataset']=='seamless']
    assert len(items)==len({x['key'] for x in items})==200
    return items


def gpu0(report):
    for line in report.splitlines():
        index,used,total,util=[float(x.strip()) for x in line.split(',')]
        if int(index)==0:return dict(used=used,free=total-used,util=util)
    raise ValueError('Physical GPU0 missing')


def verify_metadata(m,p):
    arm=p['arms'][0]
    assert m['seed']==SEED==p['seed'] and m['step']==6000
    assert m['checkpoint_sha256']==arm['original_sha256']
    assert m['planning_adapter']==p['planner_adapter_identity']
    assert m['generation']==p['generation']
    assert not m['listener_future_input'] and not m['target_description_access']


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    parents={k:read(v/'protocol.json') for k,v in PARENTS.items()}
    a,b=parents['A'],parents['B']
    assert a['seed']==b['seed']==137 and a['generation']==b['generation']
    assert a['planner_adapter_identity']==b['planner_adapter_identity']
    assert a['datasets']['seamless']==b['datasets']['seamless']
    ia=items_for_seamless(PARENTS['A']);ib=items_for_seamless(PARENTS['B'])
    assert {x['key']:x for x in ia}=={x['key']:x for x in ib}
    refs=Path(a['datasets']['seamless']['prepared'])/'references.jsonl'
    assert sha(refs)==a['datasets']['seamless']['references_sha256']
    assert (PARENTS['A']/'conditions/seamless').resolve()==(PARENTS['B']/'conditions/seamless').resolve()
    assert (PARENTS['A']/'whisper/seamless').resolve()==(PARENTS['B']/'whisper/seamless').resolve()
    for name,parent in PARENTS.items():
        assert read(parent/'FINAL_AUDIT.json')['success']
        p=parents[name];work=root/name;work.mkdir(exist_ok=True)
        if (work/'protocol.json').exists():
            old=read(work/'protocol.json');assert old['seed']==SEED and old['rerun_code_sha256']==sha(SELF)
            assert old['arms']==p['arms']
            continue
        for folder in ['conditions/seamless','whisper/seamless','negative/00','negative/07']:
            link(parent/folder,work/folder)
        if name=='A':link(parent/'DATA_CONFIG.json',work/'DATA_CONFIG.json')
        for x in ia:
            folder=work/'conditions'/x['key'];assert ready(folder/'qwen') and ready(folder/'text')
            assert ready(work/'whisper'/x['key'])
            assert read(folder/'qwen/plan.json')['adapter']==p['planner_adapter_identity']
        cluster=read(PARENTS['B']/'cluster.json');cluster['nodes'][0]['gpus']=[0]
        cluster.update(training_script=str(SELF),minimum_free_gib=80)
        save(work/'cluster.json',cluster);save(work/'shared_items.json',ia)
        for rank,items in enumerate(partition(ia,8)):save(work/'shards'/f'{rank:02d}.json',items)
        p=copy.deepcopy(p);p.update(seed=SEED,workers=8,total_samples=200,total_outputs=200,
            datasets={'seamless':p['datasets']['seamless']},cluster=str(work/'cluster.json'),
            metrics_cluster=str(work/'cluster.json'),metrics_workers_per_gpu=8,
            metrics_inference_workers_per_gpu=8,metrics_use_released_inference_nodes=False,
            worker_cli=str(SELF),shared_items_manifest=str(work/'shared_items.json'),
            shared_items_sha256=sha(work/'shared_items.json'),rerun_code_sha256=sha(SELF),
            rerun_parent=str(parent),rerun_parent_seed=137,rerun_condition_cache='Unchanged SFT2752 plans/H/Gemma and speaker Whisper',
            authorized_scope='node11 physical GPU0 only; training on GPUs1-7 untouched',
            execution_order='render one model ->4 decode workers ->8 pipelined metric workers; repeat other model')
        save(work/'protocol.json',p)
        save(work/'status.json',dict(stage='prepared',at=time.time(),seed=SEED))
    save(root/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),seed=SEED,previous_seed=137,
        same_200_ids_and_geometry=True,same_plans_hidden_gemma_and_whisper=True,
        generation=a['generation'],planner=a['planner_adapter_identity'],physical_gpu=0,outputs=400,
        rerun_code_sha256=sha(SELF),weights={k:p['arms'] for k,p in parents.items()}))


def worker(root,name,stage,slot):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert '172.29.189.119' in subprocess.check_output(['hostname','-I'],text=True).split()
    work=root/name;p=read(work/'protocol.json');assert p['rerun_code_sha256']==sha(SELF)
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.shared_items import all_items
    save(work/'devices'/f'{stage}_{slot:02d}.json',dict(at=time.time(),pid=os.getpid(),host=os.uname().nodename,
        physical_gpu=0,seed=SEED,stage=stage,slot=slot))
    lock=work/'locks'/f'{stage}_{slot}.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if name=='A':
            import p2_current_dual_eval_20261108 as native
            native.WHISPER=work;native.INFO['whisper']['work']=work
            context=native.runtime('whisper')
        else:
            import p2_ltx6000_latest_sft_dual200_20261107 as native
            context=native.runtime(work)
        with context as cfg:
            if stage=='render':
                if name=='A':native.render_whisper(EXPORTS,0,cfg)
                else:native.render(work,all_items(work,[],p),0,p)
            else:rt.decode(work,all_items(work,[],p),slot,p,p['arms'])
        save(work/'progress/standalone'/f'{stage}_{slot}.json',dict(status='complete',at=time.time()))


def count(work,stage):
    p=read(work/'protocol.json');arm=p['arms'][0]['name']
    return sum(ready(work/stage/arm/x['key']) for x in read(work/'shared_items.json'))


def stage(root,name,kind,workers):
    work=root/name;active={};completed=set();attempts={};limit=workers
    while len(completed)<workers:
        if (root/'STOP').exists():save(work/'STOP',dict(reason='seed238 user stop',at=time.time()));return False
        for slot,p in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            del active[slot]
            if rc==0:completed.add(slot)
            else:
                log=work/'logs'/f'{kind}_{slot}_attempt{attempts[slot]}.log'
                if kind=='decode' and 'out of memory' in log.read_text(errors='replace').lower():limit=2
                if attempts[slot]>=3:raise RuntimeError('Stage failed after retries: '+str(log))
        for slot in range(workers):
            if slot in active or slot in completed or len(active)>=limit:continue
            attempts[slot]=attempts.get(slot,0)+1
            path=work/'logs'/f'{kind}_{slot}_attempt{attempts[slot]}.log';path.parent.mkdir(exist_ok=True)
            cmd=[str(PYTHON),'-u',str(SELF),'worker','--work',str(root),'--model',name,'--stage',kind,'--slot',str(slot)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
                TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True')
            with path.open('ab',buffering=0) as out:active[slot]=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            save(work/'processes'/f'{kind}_{slot}.json',dict(pid=active[slot].pid,at=time.time(),slot=slot,physical_gpu=0,log=str(path)))
        save(root/'status.json',dict(stage=kind,model=name,completed=count(work,'latents' if kind=='render' else 'generated'),total=200,active=len(active),at=time.time(),seed=SEED))
        time.sleep(5)
    assert count(work,'latents' if kind=='render' else 'generated')==200
    return True


def finalize(root,name):
    work=root/name;p=read(work/'protocol.json');arm=p['arms'][0]['name'];results=read(work/'RESULTS.json')
    for x in read(work/'shared_items.json'):
        out=work/'generated'/arm/x['key'];assert ready(out);verify_metadata(read(out/'metadata.json'),p)
    s=results[arm+'/seamless'];assert s['requested_pairs']==s['audio']['requested_pairs']==200
    assert s['audio']['pairs_with_errors']==0
    assert all(s['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=200,seed=SEED,step=6000,physical_gpu=0))


def queue(root):
    assert '172.29.189.119' in subprocess.check_output(['hostname','-I'],text=True).split()
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),host=os.uname().nodename,physical_gpu=0))
        prepare(root)
        from p2_testsets_eval import cluster,metrics
        from p2_current_av6000_score_overlay_20261108 import launch_budget
        for name in ['A','B']:
            work=root/name
            if (work/'FINAL_AUDIT.json').exists():continue
            while True:
                if (root/'STOP').exists():return
                report=gpu0(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True))
                if report['used']<2048 and report['util']<10:break
                save(root/'status.json',dict(stage='waiting_gpu0',model=name,resource=report,at=time.time()));time.sleep(10)
            if not stage(root,name,'render',1):return
            if not stage(root,name,'decode',4):return
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=200,at=time.time()))
            cluster.bind(work);metrics.prepare_smoke(work)
            save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=200,at=time.time()))
            latest={}
            def available(report):
                latest.clear();latest.update(gpu0(report))
                return [0] if launch_budget(latest['free'],0)>0 else []
            def slots(reports,active):
                eligible=[(n,g)for n,cap,gpus in reports for g in gpus if g==0]
                return [eligible[0]]*launch_budget(latest.get('free',0),len(active)) if eligible else []
            save(root/'status.json',dict(stage='metrics',model=name,seed=SEED,at=time.time()))
            with ExitStack() as stack:
                stack.enter_context(patch.object(metrics,'CLI',ROOT/'scripts/p2_pipelined_testsets_eval.py'))
                stack.enter_context(patch.object(metrics,'idle_gpus',available))
                stack.enter_context(patch.object(metrics,'balanced_metric_slots',slots))
                metrics.queue(work)
            if (root/'STOP').exists() or (work/'STOP').exists():return
            finalize(root,name)
        results={};comparison={}
        for name,parent in PARENTS.items():
            current=read(root/name/'RESULTS.json');original=read(parent/'RESULTS.json')
            results.update(current)
            for key,s in current.items():comparison[name]=dict(seed137=original[key],seed238=s)
        save(root/'RESULTS.json',results);save(root/'SEED_COMPARISON.json',comparison)
        save(root/'FINAL_AUDIT.json',dict(success=True,outputs=400,seed=SEED,at=time.time(),physical_gpu=0))
        save(root/'status.json',dict(stage='complete',outputs=400,seed=SEED,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','queue','worker'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--model',choices=['A','B'])
    parser.add_argument('--stage',choices=['render','decode']);parser.add_argument('--slot',type=int,default=0)
    a=parser.parse_args()
    try:
        if a.action=='prepare':prepare(a.work.resolve())
        elif a.action=='worker':worker(a.work.resolve(),a.model,a.stage,a.slot)
        else:queue(a.work.resolve())
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
