"""Real torchrun/DDP plumbing gate; explicitly not a pretrained P2 GPU test."""
from datetime import timedelta
import copy
import os
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omni_av.data import write_json_new
from .checkpoint import load
from .distributed import parameter_digest, save_distributed
from .launch import agree, establish_output
from .performance import global_sample_index
from .validation import distributed_mean


def run(args):
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    device = torch.device("cpu" if args.probe == "cpu" else f"cuda:{local}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("gloo" if device.type == "cpu" else "nccl", timeout=timedelta(seconds=120))
    writer = None
    try:
        establish_output(args.output, rank)
        if args.checkpoint_dir and args.checkpoint_dir != args.output:
            establish_output(args.checkpoint_dir, rank)
        checkpoint_dir = args.checkpoint_dir or args.output
        agree(dict(seed=args.seed, steps=args.steps, accumulation=args.accumulation, resume=str(args.resume)))
        identity = dict(probe=True, world=world, seed=args.seed, accumulation=args.accumulation)
        torch.manual_seed(args.seed)
        raw = torch.nn.Module()
        raw.add_module("connector", torch.nn.Linear(3, 2))
        raw.forward = raw.connector.forward
        raw.to(device)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=.01)
        start = load(args.resume, raw, identity, optimizer, rank=rank)["step"] if args.resume else 0
        model = DDP(raw, device_ids=[local] if device.type == "cuda" else None)
        # Every rank independently computes the serial global-batch reference.
        reference = torch.nn.Linear(3, 2).to(device)
        reference.load_state_dict(raw.connector.state_dict())
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01)
        ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        if rank == 0 and args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.output / "tensorboard"))
        def data(ordinal):
            x = torch.tensor([[ordinal % 7 / 7, ordinal % 5 / 5, 1.]], device=device)
            return x, torch.stack((x[:, 0] - x[:, 1], x[:, 0] + x[:, 1]), dim=-1)
        maximum_error = 0.
        for step in range(start + 1, start + args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            from contextlib import nullcontext
            for micro in range(args.accumulation):
                ordinal = global_sample_index(step, micro, rank, world, args.accumulation)
                x, target = data(ordinal)
                with model.no_sync() if micro + 1 < args.accumulation else nullcontext():
                    loss = torch.nn.functional.mse_loss(model(x), target)
                    (loss / args.accumulation).backward()
            for micro in range(args.accumulation):
                for source_rank in range(world):
                    ordinal = global_sample_index(step, micro, source_rank, world, args.accumulation)
                    x, target = data(ordinal)
                    (torch.nn.functional.mse_loss(reference(x), target) / (world * args.accumulation)).backward()
            optimizer.step()
            ref_optimizer.step()
            for got, expected in zip(raw.parameters(), reference.parameters()):
                maximum_error = max(maximum_error, float((got - expected).abs().max().detach()))
                torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-6)
            if writer:
                writer.add_scalar("probe/local_loss", float(loss.detach()), step)
            if step % args.save_every == 0 or step == start + args.steps:
                save_distributed(checkpoint_dir / f"extractor_step_{step:06d}.pt", raw, optimizer, step, identity, rank, world)
        # Fewer Dev items than ranks: zero-item ranks must still join the reduction.
        count = max(1, world - 1)
        values = list(range(rank, count, world))
        mean, actual = distributed_mean([sum(values)], len(values), device)
        assert actual == count and mean == [(count - 1) / 2]
        hashes = [None] * world
        dist.all_gather_object(hashes, parameter_digest(raw))
        assert len(set(hashes)) == 1
        hosts = [None] * world
        dist.all_gather_object(hosts, socket.gethostname())
        if rank == 0:
            write_json_new(args.output / "completed.json", dict(probe=True, backend=dist.get_backend(),
                world_size=world, nnodes=args.nnodes, start_step=start, end_step=start + args.steps,
                serial_max_abs_error=maximum_error, ranks_identical=True, uneven_dev_count=actual,
                physical_multinode_verified=len(set(hosts)) > 1, hosts=sorted(set(hosts)), pretrained_p2_test=False))
    finally:
        if writer:
            writer.close()
        dist.destroy_process_group()
