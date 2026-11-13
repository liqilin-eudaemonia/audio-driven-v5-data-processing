"""CPU preparation: source checkpoint links, explicit RNG partition, data identity."""
import copy,json,math,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK/'vendor'),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_planning_sft.reference_data import datasets
from p2_planning_sft.data import SYSTEM,USER
from p2_planning_sft.continuation import validate_extension,align_loaded_learning_rate

torch.set_num_threads(1)
config=read(WORK/'config.json');cluster=read(WORK/'cluster.json')
source=Path(config['extension_policy']['parent_checkpoint']);destination=Path(config['extend_from'])
destination.mkdir(parents=True,exist_ok=True)
files={}
for p in source.iterdir():
    if p.name.startswith('rng_state_'):continue
    if p.is_file():
        if not (destination/p.name).exists():
            (destination/p.name).symlink_to(p)
        assert (destination/p.name).resolve() == p.resolve()
        files[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
rng=[];rank=0
for node in cluster['nodes']:
    for local in range(len(node['gpus'])):
        name=f'rng_state_{rank}.pth';s=torch.load(source/name,map_location='cpu',weights_only=False)
        original_count=len(s['cuda']);assert original_count==8
        s['cuda']=s['cuda'][:len(node['gpus'])]
        torch.save(s,destination/name)
        rng.append(dict(rank=rank,source_rank=rank,node=node['name'],local_rank=local,source_cuda_states=original_count,
                        target_cuda_states=len(s['cuda']),source_sha256=sha(source/name),mapped_sha256=sha(destination/name)))
        rank+=1
assert rank==20
atomic_json(WORK/'CHECKPOINT_MAPPING.json',dict(source=str(source),load=str(destination),files=files,rng=rng,
    note='Model/Adam/scheduler unchanged. Python/NumPy/CPU RNG retained per rank0..27; CUDA RNG truncated to visible local devices. Topology changes floating-point trajectory.',at=time.time()))
train,dev,full=datasets({k:v for k,v in config.items() if k != 'extend_from'},False)
identity=dict(config=config,smoke=False,train_manifest=train.manifest_sha256,dev_manifest=dev.manifest_sha256,
    train_ids=digest(train.ids),train_count=len(train),dev_count=len(dev),system=SYSTEM,user=USER,
    code={p.name:sha(p) for p in (WORK/'vendor/p2_planning_sft').glob('*.py')},world_size=20,
    reference_data=dict(train=train.identity,diagnostic_dev=dev.identity,full_dev=full.identity),full_dev_count=len(full))
extension=validate_extension(config,identity,WORK/'training')
assert math.ceil(len(train)/256)==2752
for n in (128,112):
    parts=[list(range(n))[n*r//20:n*(r+1)//20] for r in range(20)]
    assert [i for p in parts for i in p]==list(range(n)) and all(parts)
optimizer=torch.load(source/'optimizer.pt',map_location='cpu',weights_only=True)
assert len(optimizer['state'])==504
assert {int(v['step']) for v in optimizer['state'].values()}=={2752}
assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in optimizer['state'].values())
sched=torch.load(source/'scheduler.pt',map_location='cpu',weights_only=True);assert sched['last_epoch']==2752
from transformers import get_cosine_schedule_with_warmup
dummy=torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.zeros(()))]) for _ in sched['base_lrs']],lr=config['training']['learning_rate'])
schedule=get_cosine_schedule_with_warmup(dummy,num_warmup_steps=extension['warmup_steps'],num_training_steps=8256)
schedule.load_state_dict(sched);lr=align_loaded_learning_rate(dummy,schedule)
assert lr[0]>0 and schedule.last_epoch==2752
assert not torch.cuda.is_initialized()
atomic_json(WORK/'PREPARED.json',dict(passed=True,parent_step=2752,target_step=8256,additional_epochs=2,
    steps_per_epoch=2752,train_count=len(train),global_batch_size=256,world_size=20,
    full_rank_batch_sizes=[6,7],tail_rank_batch_sizes=[5,6],gradient_accumulation=2,global_micro_batch_size=128,tail_padding=False,warmup_steps=extension['warmup_steps'],
    starting_learning_rates=lr,config_sha256=sha(WORK/'config.json'),code=identity['code'],extension=extension,at=time.time()))
print(json.dumps(read(WORK/'PREPARED.json')),flush=True)
