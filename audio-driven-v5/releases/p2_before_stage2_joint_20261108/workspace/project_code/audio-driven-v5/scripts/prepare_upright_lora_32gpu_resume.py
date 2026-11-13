"""Prepare a separately sealed, checkpoint-preserving 32-GPU continuation."""
import array
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
PARENT=ROOT/'experiments/p2_planning_sft_upright_3b_gpu7_20261105'
WORK=ROOT/'experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json


def main():
    if (WORK/'PREPARED.json').exists():
        print(json.dumps(read(WORK/'PREPARED.json')));return
    if WORK.exists():
        raise FileExistsError('Inspect partially prepared migration before repeating')
    old=read(PARENT/'config.json');prep=read(PARENT/'PREPARED.json')
    assert sha(PARENT/'config.json')==prep['config_sha256']
    for p,h in prep['code'].items():assert sha(p)==h,p
    audit=read(PARENT/'monitor/checkpoint_audits/checkpoint-1500.json')
    assert audit['passed'] and audit['step']==1500 and audit['optimizer_steps']==[1500.0]
    source=PARENT/'training/checkpoint-1500'
    WORK.mkdir();vendor=WORK/'vendor';vendor.mkdir();package=vendor/'p2_planning_sft'
    shutil.copytree(PARENT/'vendor/p2_planning_sft',package,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PARENT/'vendor/upright_lora_media.py',vendor/'upright_lora_media.py')
    shutil.copy2(ROOT/'scripts/upright_lora_global_batch.py',package/'global_batch.py')
    shutil.copy2(ROOT/'scripts/upright_lora_world_resume.py',package/'performance_resume.py')
    train=package/'train.py';text=train.read_text()
    a='batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,'
    b="batch_size=getattr(self, 'sampler_group_batch', None) or self.args.train_batch_size * self.args.gradient_accumulation_steps,"
    assert text.count(a)==1;text=text.replace(a,b)
    a="    trainer.fixed_global_batch = t.get('global_batch_size')"
    assert text.count(a)==1;text=text.replace(a,a+"\n    trainer.sampler_group_batch = t.get('sampler_group_batch')")
    a='        self.written = True\n'
    assert text.count(a)==1
    b="        atomic_json(self.output / 'gradient_audits' / f'rank-{args.process_index:02d}.json', dict(rank=args.process_index, finite=finite, norm=norm, trainable_tensors=len(rows), trainable_parameters=sum(p.numel() for _,p in rows), frozen_gradient=frozen_gradient, world_size=args.world_size, step_before_update=state.global_step))\n"+a
    text=text.replace(a,b);train.write_text(text)
    entry=r'''import argparse,datetime,json,os,socket,subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from p2_listen.contracts import atomic_json
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--stop-after',type=int);a=p.parse_args()
config=json.loads(Path(a.config).read_text());rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==32
assert not a.smoke and a.output is None
assert config['training']['micro_batch']==8 and config['training']['gradient_accumulation']==1
torch.set_num_threads(config.get('cpu_threads',2));torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
x=torch.tensor(float(rank),device=f'cuda:{local}');dist.all_reduce(x);assert x.item()==496.0
uuid=subprocess.check_output(['nvidia-smi','-i',str(local),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
atomic_json(Path(config['output'])/'communication'/f'rank-{rank:02d}.json',dict(rank=rank,local_rank=local,world_size=world,host=socket.gethostname(),gpu_uuid=uuid,nccl_sum=x.item(),pid=os.getpid()))
from p2_planning_sft.train import train
train(a.config,a.resume,a.smoke,a.output,a.stop_after)
'''
    (vendor/'train_entry.py').write_text(entry)
    launcher=(ROOT/'scripts/p2_planning_cluster.py').read_text()
    a='ROOT = Path(__file__).resolve().parents[1]';assert launcher.count(a)==1
    launcher=launcher.replace(a,"ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')")
    a="        cmd += [str(ROOT/'scripts/p2_planning_sft.py'), '--config', str(Path(config).resolve())]";assert launcher.count(a)==1
    b=f"        cmd += [{str(vendor/'train_entry.py')!r}, '--config', str(Path(config).resolve())]\n        if cluster.get('stop_after') is not None:\n            cmd += ['--stop-after', str(cluster['stop_after'])]"
    launcher=launcher.replace(a,b)
    (vendor/'cluster_launcher.py').write_text(launcher)
    for p in vendor.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    seal=read(PARENT/'data_seal.json');derived=copy.deepcopy(seal)
    derived['reader_code']={k.replace(str(PARENT/'vendor'),str(vendor)):v for k,v in seal['reader_code'].items()}
    for p,h in derived['reader_code'].items():
        path=Path(p) if p.startswith('/') else ROOT/'src'/p
        assert sha(path)==h,p
    atomic_json(WORK/'data_seal.json',derived,immutable=True)
    dest=WORK/'load_checkpoint/checkpoint-1500';dest.mkdir(parents=True)
    records={}
    for p in source.iterdir():
        if p.is_file():
            st=p.stat();value=sha(p);assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
            if p.name in audit['files']:assert value==audit['files'][p.name]['sha256']
            os.link(p,dest/p.name);records[p.name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=value)
    atomic_json(WORK/'checkpoint-1500-receipt.json',dict(checkpoint=str(source),parent_run_sha256=sha(PARENT/'training/run.json'),files=records),immutable=True)
    import torch
    torch.set_num_threads(1)
    rng=torch.load(source/'rng_state.pth',map_location='cpu',weights_only=False)
    cuda=rng['cuda'];cuda=cuda[0] if isinstance(cuda,list) else cuda
    assert cuda.dtype==torch.uint8 and cuda.ndim==1
    expanded=dict(rng,cuda=[cuda.clone() for _ in range(8)])
    torch.save(expanded,dest/'rng_state_0.pth')
    for rank in range(1,32):os.link(dest/'rng_state_0.pth',dest/f'rng_state_{rank}.pth')
    spec=importlib.util.spec_from_file_location('migration_partition',package/'global_batch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from transformers.trainer_pt_utils import LengthGroupedSampler
    import sqlite3
    with sqlite3.connect(Path(seal['directory'])/'index.sqlite') as conn:
        durations=conn.execute("select sid,duration from samples where split='train' order by sid").fetchall()
    assert len(durations)==704368
    lengths=[round(d*25)+256 for _,d in durations]
    torch.manual_seed(old['seed']);indices=list(LengthGroupedSampler(batch_size=256,lengths=lengths))
    ranks=[list(mod.FixedGlobalBatchShard(indices,256,32,r)) for r in range(32)]
    assert all(len(b)==2752 for b in ranks)
    for step in range(2752):
        merged=[i for rank in range(32) for i in ranks[rank][step]]
        assert merged==indices[step*256:(step+1)*256],step
    report=dict(passed=True,examples=len(indices),steps=2752,global_batch=256,world_size=32,
        grouping_batch=256,grouping_matches_parent_micro_times_accumulation=True,
        every_global_batch_reconstructed_exactly=True,checkpoint_skip_global_batches=1500,
        checkpoint_skip_examples=384000,tail_examples=112,tail_rank_sizes=sorted(set(len(r[-1]) for r in ranks)),
        no_added_or_dropped_samples=True,tail_padding=False,code_sha256=sha(package/'global_batch.py'))
    atomic_json(WORK/'BATCH_PARTITION.json',report,immutable=True)
    policy=dict(parent_checkpoint=str(source),load_checkpoint=str(dest),step=1500,
        parent_run_sha256=sha(PARENT/'training/run.json'),parent_vendor=str(PARENT/'vendor'),new_vendor=str(vendor),
        checkpoint_receipt=str(WORK/'checkpoint-1500-receipt.json'),checkpoint_receipt_sha256=sha(WORK/'checkpoint-1500-receipt.json'),
        rank_rng_sha256={str(r):sha(dest/f'rng_state_{r}.pth') for r in range(32)},
        rng_scope='Single-rank Python/NumPy/CPU state preserved; single CUDA RNG expanded to 8 visible devices per rank. Distributed floating-point trajectory may differ.',
        batch_partition_report=str(WORK/'BATCH_PARTITION.json'),batch_partition_report_sha256=sha(WORK/'BATCH_PARTITION.json'),
        input_equivalence=str(PARENT/'INPUT_EQUIVALENCE.json'),input_equivalence_sha256=sha(PARENT/'INPUT_EQUIVALENCE.json'),
        derived_seal_sha256=sha(WORK/'data_seal.json'),validated_code={p.name:sha(p) for p in package.glob('*.py')})
    config=copy.deepcopy(old);config.update(output=str(WORK/'training'),initialization='resume_upright_checkpoint_with_optimizer_and_scheduler')
    config['training'].update(gradient_accumulation=1,global_batch_size=256,sampler_group_batch=256)
    config['data']['seal']=str(WORK/'data_seal.json');config['performance_resume']=policy
    atomic_json(WORK/'config.json',config,immutable=True)
    cluster=read(ROOT/'configs/p2_planning_cluster_filtered_20261102.json')
    cluster['master_port']=29791;cluster['minimum_free_gib']=139
    cluster['network']['PYTHONPATH']=str(vendor)+':'+str(ROOT/'src')+':'+str(ROOT/'scripts')
    cluster['network']['MKL_NUM_THREADS']='2'
    for node in cluster['nodes']:
        if node['name']=='node8':node['host']='172.29.104.237'
        node['ssh_key']=cluster['ssh_key']
    atomic_json(WORK/'cluster.json',cluster,immutable=True)
    atomic_json(WORK/'cluster_canary.json',dict(cluster,stop_after=1503),immutable=True)
    atomic_json(WORK/'PREPARED.json',dict(passed=True,parent=str(PARENT),parent_checkpoint=str(source),
        output=config['output'],train_examples=704368,total_steps=2752,resume_step=1500,
        global_batch=256,micro_batch=8,gradient_accumulation=1,world_size=32,
        code={str(p):sha(p) for p in vendor.rglob('*.py')},config_sha256=sha(WORK/'config.json'),
        cluster_sha256=sha(WORK/'cluster.json'),parent_checkpoint_preserved=True,at=time.time()),immutable=True)
    print(json.dumps(dict(prepared=True,work=str(WORK),partition=report)),flush=True)

if __name__=='__main__':main()
