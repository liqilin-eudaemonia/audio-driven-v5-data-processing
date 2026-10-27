"""P3 launch helpers: explicit topology, matching sources and idle GPU checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def signature():
    files = [*ROOT.glob("src/omni_av_p3/*.py"), *ROOT.glob("scripts/pipeline3*.py")]
    files += [ROOT / p for p in ("src/omni_av/lora.py", "src/omni_av/masking.py", "src/omni_av/omni.py",
        "src/omni_av/data.py", "src/omni_av_p2/decoder.py", "src/omni_av_p2/connector.py",
        "src/omni_av_p2/checkpoint.py", "src/omni_av_p2/launch.py", "scripts/pipeline2_cluster.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def cluster_nodes(config):
    nodes = config["nodes"]
    masters = [n for n in nodes if n.get("role") == "master"]
    if len(masters) != 1 or len({n["host"] for n in nodes}) != len(nodes):
        raise ValueError("Exactly one master and distinct hosts required")
    if len({n["name"] for n in nodes}) != len(nodes):
        raise ValueError("Duplicate node names")
    for n in nodes:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", n["name"]) or n["name"] in {".", ".."}:
            raise ValueError("Unsafe node name")
        if not n.get("gpus") or len(set(n["gpus"])) != len(n["gpus"]) or any(g < 0 for g in n["gpus"]):
            raise ValueError("Invalid GPU list")
    if len({len(n["gpus"]) for n in nodes}) != 1:
        raise ValueError("Static torchrun requires equal GPU counts per node")
    return masters + [n for n in nodes if n is not masters[0]]


def environment(args, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.cpu else ",".join(map(str, gpus)),
               OMP_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000000")
    if args.network_config:
        profile = json.loads(args.network_config.read_text())
        if any(not k.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_")) for k in profile):
            raise ValueError("Only NCCL/GLOO network variables are accepted")
        env.update({k: str(v) for k, v in profile.items()})
    return env


def preflight(args):
    from omni_av_p2.launch import idle_gpus
    from .distributed import sha, validate_index
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        raise ValueError("Invalid GPU IDs")
    if not args.cpu:
        idle_gpus(gpus)
        # Even a low-memory background process makes a card non-idle.
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        selected = {line.split(",")[1].strip() for line in inventory.splitlines() if int(line.split(",")[0]) in gpus}
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        if selected & set(active.splitlines()):
            raise RuntimeError("A selected GPU has a compute process")
    if args.output.exists():
        raise FileExistsError(args.output)
    artifacts = {}
    for path in (args.cache, args.resume, args.initialize_from, args.network_config):
        if path:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts[str(path.resolve())] = sha(path)
    if not args.tiny:
        from .data import contract
        from .model import Config
        index = json.loads(args.cache.read_text())
        validate_index(index)
        cfg = Config(**index["contract"]["config"])
        if index["contract"] != json.loads(json.dumps(contract(cfg))):
            raise ValueError("Cache contract mismatch")
        for split in ("train", "dev"):
            for entry in index[split]:
                if not Path(entry["path"]).is_file():
                    raise FileNotFoundError(entry["path"])
    if args.resume and not args.resume.with_suffix(".complete.json").is_file():
        raise ValueError("DDP resume needs a completion marker; use --initialize-from for old single-GPU weights")
    return dict(source=signature(), artifacts=artifacts, python=sys.executable,
                gpus=gpus, master=args.master_addr, node_rank=args.node_rank,
                world_size=args.nnodes * len(gpus))
