import json
import pathlib
import sys
import time
import tempfile
import copy
R = pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0] = [str(R/'src'), str(R/'scripts'), str(pathlib.Path(__file__).parent)]
import torch
from baseline_training.glare_cached import Registry
from baseline_training.glare_batched import BatchedCachedFLOAT, length_batches
from baseline_training.training import build, seed, set_train
from prefetch import PreparedFLOAT, prefetched

out = pathlib.Path(__file__).parent
torch.set_num_threads(2)
seed(137)
dataset = Registry(R/'experiments/glare_cache_28gpu_20261110/production_gpu_fer/TRAINING_CACHES_LATEST.json')
by_length = {}
with dataset.path.open() as f:
    for i, line in enumerate(f):
        n = json.loads(line)['glare_caches']['view']['frames']
        entries = by_length.setdefault(n, [])
        if len(entries) < 128: entries.append(i)
keys = sorted(n for n, rows in by_length.items() if len(rows) == 128)
chosen = [keys[len(keys)//2], keys[-1]]
ids = [i for n in chosen for i in by_length[n]]
data = [dataset[i] for i in ids]
# Small contiguous-shard probe, exact CPU tensors with independent temporary storage.
with tempfile.TemporaryDirectory(prefix='glare-shard-probe-') as folder:
    shard = pathlib.Path(folder)/'samples.pt'
    torch.save(data[:64], shard)
    shard_bytes = shard.stat().st_size
    read_times = []
    for _ in range(3):
        tick = time.perf_counter()
        loaded = torch.load(shard, map_location='cpu', weights_only=True)
        read_times.append(time.perf_counter()-tick)
    assert all(torch.equal(a['wav2vec'], b['wav2vec']) for a,b in zip(data, loaded))
    tick = time.perf_counter()
    direct = [dataset[i] for i in ids[:64]]
    individual_time = time.perf_counter()-tick
    io_result = dict(samples=64, bytes=shard_bytes, shard_reads_s=read_times,
                     individual_read_s=individual_time, tensors_equal=True,
                     note='Warm page-cache comparison. Full Weka cold-read gain remains to be measured.')
print(json.dumps(dict(io_probe=io_result)), flush=True)
free, total = torch.cuda.mem_get_info()
# Keep headroom for active tasks on the allocated card.
limit_bytes = min(int(total*.35), max(8*1024**3, free-8*1024**3))
torch.cuda.set_per_process_memory_fraction(limit_bytes/total)
model, _ = build('glare', 'cuda:0')
model.opt.reaction_loss_weight = .05
set_train(model, 'glare')
params = [p for p in model.parameters() if p.requires_grad]
results = []
for size, overlap in [(32,False),(64,False),(128,False),(64,True),(128,True),(32,False)]:
    seed(512)
    wrapper = PreparedFLOAT(model) if overlap else BatchedCachedFLOAT(model)
    optimizer = torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    times = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            groups = list(length_batches(data,size))
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            with torch.autocast('cuda', dtype=torch.float16):
                inputs = prefetched(groups,'cuda:0') if overlap else groups
                for value in inputs:
                    count = len(value['motion']) if overlap else len(value)
                    loss = wrapper(value)['loss']
                    with torch.autocast('cuda',enabled=False):
                        scaler.scale(loss*count/256).backward()
                    losses.append(loss.detach())
            assert torch.isfinite(torch.stack(losses)).all()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-tick)
        result = dict(microbatch=size,overlap=overlap,seconds=times,
                      peak_gib=torch.cuda.max_memory_allocated()/1024**3,finite=True)
    except torch.cuda.OutOfMemoryError as exc:
        optimizer.zero_grad(set_to_none=True)
        result = dict(microbatch=size,overlap=overlap,oom=True,error=str(exc)[:500])
        torch.cuda.empty_cache()
    results.append(result)
    print(json.dumps(result),flush=True)
# Check prepared and ordinary paths with identical native random draws.
seed(731)
model.eval()
group=data[:8]
with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
    standard=BatchedCachedFLOAT(model)(group)['loss']
    seed(731)
    prepared=next(prefetched([group],'cuda:0'))
    actual=PreparedFLOAT(model)(prepared)['loss']
assert torch.allclose(standard,actual,atol=1e-6,rtol=1e-5), (standard,actual)
result = dict(results=results, io=io_result, frame_counts=chosen, ids=ids,
              prefetch_loss_matches=True, memory_cap_gib=limit_bytes/1024**3)
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
