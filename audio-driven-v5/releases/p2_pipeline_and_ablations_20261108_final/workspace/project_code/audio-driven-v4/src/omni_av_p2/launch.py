"""One torchrun agent per node; explicit Python, rendezvous and isolated logs."""
import json
import os
import re
import subprocess
import sys


def idle_gpus(gpus):
    status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = {int(v[0]): (int(v[1]), int(v[2])) for line in status.splitlines() if (v := line.split(","))}
    if any(g not in rows or rows[g][0] > 1000 or rows[g][1] > 5 for g in gpus):
        raise RuntimeError(f"Requested GPUs are not idle: {rows}")


def launch_command(args, nproc, script, argv):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id) or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a simple path-safe identifier")
    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", "--max_restarts=0"]
    if args.nnodes == 1:
        command += ["--standalone"]
    else:
        command += [f"--nnodes={args.nnodes}", f"--node_rank={args.node_rank}",
                    f"--master_addr={args.master_addr}", f"--master_port={args.master_port}",
                    f"--rdzv_id={args.run_id}"]
    log_dir = args.output.parent / (args.output.name + ".torchrun") / args.run_id / f"node_{args.node_rank}"
    return command + ["--tee=3", f"--log_dir={log_dir}", str(script), *argv, "--worker"]


def launch_environment(args, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.probe == "cpu" else ",".join(map(str, gpus)),
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000000")
    if args.network_config:
        values = json.loads(args.network_config.read_text())
        if any(not k.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_")) for k in values):
            raise ValueError("Network profile accepts NCCL/GLOO variables only")
        env.update({k: str(v) for k, v in values.items()})
    return env


def establish_output(path, rank):
    # Broadcast filesystem errors instead of leaving non-master ranks at a barrier.
    import torch.distributed as dist
    error = [None]
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    dist.barrier()
    if not path.is_dir():
        raise RuntimeError("All nodes must share the output filesystem")


def agree(value):
    import hashlib
    import torch.distributed as dist
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, digest)
    if len(set(hashes)) != 1:
        raise ValueError("Rank configuration/data/resume mismatch")


def source_signature():
    """Reject stale per-node code before loading the expensive frozen model."""
    from pathlib import Path
    import hashlib
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("omni_av*/*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
