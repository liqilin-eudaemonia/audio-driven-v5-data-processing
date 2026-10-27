"""YAML-driven Stage-2 SSH/torchrun orchestration with run-scoped process cleanup."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from ltx_trainer import logger

ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / ".venv/bin/python"
ENTRY = ROOT / "packages/ltx-trainer/scripts/train_omni_stage2.py"
MODULE = "ltx_trainer.omni_stage2_cluster"


def build_plan(config_path: Path, env: dict[str, str]) -> dict[str, Any]:
    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.omni_stage2 import OmniJointConfig
    from ltx_trainer.omni_stage2_trainer import validate_joint_configuration

    raw = yaml.safe_load(config_path.read_text())
    joint = OmniJointConfig(**raw.pop("omni_joint"))
    config = LtxTrainerConfig(**raw)
    validate_joint_configuration(config, joint)
    cluster = config.cluster
    if cluster is None or not cluster.nodes:
        raise ValueError("Multi-node launcher requires cluster.nodes (a single master also supports one node)")
    masters = [node for node in cluster.nodes if node.role == "master"]
    if len(masters) != 1:
        raise ValueError("cluster.nodes must have exactly one master")
    nodes = [masters[0], *(node for node in cluster.nodes if node.role != "master")]
    if len({node.name for node in nodes}) != len(nodes) or len({node.host for node in nodes}) != len(nodes):
        raise ValueError("Cluster node names and hosts must be unique")
    nproc = int(env.get("NPROC_PER_NODE", cluster.nproc_per_node))
    port = int(env.get("MASTER_PORT", cluster.master_port))
    if nproc < 1 or not 0 < port < 65536:
        raise ValueError("Invalid nproc_per_node or master_port")
    fsdp = env.get("FSDP_ENABLE", "0")
    if fsdp not in {"0", "1"}:
        raise ValueError("FSDP_ENABLE must be 0 or 1")
    if fsdp == "1" and nproc < 2:
        raise ValueError("HYBRID_SHARD requires at least two ranks per node")
    devices = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(nproc))))
    if len(devices.split(",")) != nproc or len(set(devices.split(","))) != nproc:
        raise ValueError("CUDA_VISIBLE_DEVICES must name nproc_per_node distinct devices")
    run_id = "omni_stage2_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(config.output_dir).resolve() / "cluster_runs" / run_id
    launch_env = {
        "PYTHONPATH": ":".join(
            str(ROOT / f"packages/{package}/src") for package in ("ltx-trainer", "ltx-core", "ltx-pipelines")
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": devices,
        "WANDB_MODE": "offline",
        "FSDP_ENABLE": fsdp,
        "ACCELERATE_USE_FSDP": "true" if fsdp == "1" else "false",
        "ACCELERATE_USE_DEEPSPEED": "false",
        "ACCELERATE_USE_MEGATRON_LM": "false",
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "NCCL_IB_TIMEOUT": "20",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_CROSS_NIC": "0",
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    for key, value in cluster.environment.items():
        if key != "LD_LIBRARY_PATH" and not key.startswith(
            ("NCCL_", "GLOO_", "TORCH_NCCL_")
        ):
            raise ValueError(
                "cluster.environment only accepts NCCL/Gloo settings and "
                f"LD_LIBRARY_PATH, got {key}"
            )
        launch_env[key] = value
    # Explicit communication overrides are propagated identically to every node.
    for key in list(launch_env):
        if key.startswith(("NCCL_", "GLOO_")) or key in {"OMP_NUM_THREADS", "WANDB_MODE"}:
            launch_env[key] = env.get(key + "_VALUE", env.get(key, launch_env[key]))
    for key, value in env.items():
        if key.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_", "OMNI_STAGE2_")) and not key.endswith("_VALUE"):
            launch_env[key] = value
    if fsdp == "1":
        launch_env.update(
            {
                "FSDP_VERSION": "1",
                "FSDP_SHARDING_STRATEGY": "HYBRID_SHARD",
                "FSDP_AUTO_WRAP_POLICY": "TRANSFORMER_BASED_WRAP",
                "FSDP_TRANSFORMER_CLS_TO_WRAP": "BasicAVTransformerBlock",
                "FSDP_USE_ORIG_PARAMS": "true",
                "FSDP_STATE_DICT_TYPE": "FULL_STATE_DICT",
                "FSDP_OFFLOAD_PARAMS": "false",
                "FSDP_SYNC_MODULE_STATES": "false",
                "FSDP_CPU_RAM_EFFICIENT_LOADING": "false",
            }
        )
    identity = env.get("SSH_IDENTITY_FILE", "")
    if not identity:
        candidate = Path(f"/private/ssh_keys/root_{socket.gethostname()}_ed25519")
        identity = str(candidate) if candidate.is_file() else ""
    if identity and not Path(identity).is_file():
        raise FileNotFoundError(identity)
    return {
        "config": str(config_path.resolve()),
        "python": str(PYTHON),
        "entry": str(ENTRY),
        "root": str(ROOT),
        "nodes": [node.model_dump() for node in nodes],
        "nproc": nproc,
        "master_addr": env.get("MASTER_ADDR", masters[0].host),
        "master_port": port,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "env": launch_env,
        "identity": identity,
        "global_batch": len(nodes)
        * nproc
        * config.optimization.batch_size
        * config.optimization.gradient_accumulation_steps
        // config.acceleration.context_parallel_size,
    }


def torchrun_command(plan: dict[str, Any], rank: int) -> list[str]:
    return [
        plan["python"],
        "-m",
        "torch.distributed.run",
        f"--nnodes={len(plan['nodes'])}",
        f"--nproc_per_node={plan['nproc']}",
        f"--node_rank={rank}",
        f"--master_addr={plan['master_addr']}",
        f"--master_port={plan['master_port']}",
        f"--rdzv_id={plan['run_id']}",
        "--max_restarts=0",
        "--tee=3",
        f"--log_dir={Path(plan['run_dir']) / f'node_{rank}' / 'torchrun'}",
        plan["entry"],
        plan["config"],
        "--disable-progress-bars",
    ]


def node_command(plan: dict[str, Any], rank: int, action: str) -> list[str]:
    command = [plan["python"], "-m", MODULE, action, str(Path(plan["run_dir"]) / "plan.json"), "--rank", str(rank)]
    if rank == 0:
        return command
    node = plan["nodes"][rank]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]
    if plan["identity"]:
        ssh += ["-i", plan["identity"], "-o", "IdentitiesOnly=yes"]
    ssh += ["-p", str(node["ssh_port"]), f"{node['ssh_user']}@{node['host']}"]
    # SSH passes a shell command; quote every argument, never concatenate raw config values.
    remote = ["env", *(f"{key}={value}" for key, value in plan["env"].items()), *command]
    return ssh + [shlex.join(remote)]


def child_env(plan: dict[str, Any]) -> dict[str, str]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FSDP_", "ACCELERATE_", "NCCL_", "GLOO_", "TORCH_NCCL_"))
        and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    }
    return {**inherited, **plan["env"]}


def preflight_nodes(plan: dict[str, Any]) -> None:
    # Run on the configured master, and refuse a rendezvous port used by another job.
    with socket.socket() as probe:
        probe.bind((plan["master_addr"], plan["master_port"]))
    reports = []
    for rank, node in enumerate(plan["nodes"]):
        result = subprocess.run(
            node_command(plan, rank, "check-node"),
            env=child_env(plan),
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        (Path(plan["run_dir"]) / f"preflight_node_{rank}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Preflight failed on {node['name']}: {result.stderr or result.stdout}")
        report = json.loads((Path(plan["run_dir"]) / f"preflight_node_{rank}.json").read_text())
        if reports and report["software"] != reports[0]["software"]:
            raise ValueError(f"Different code/dependency/config versions on node {node['name']}")
        reports.append(report)
    uuids = [gpu for report in reports for gpu in report["gpu_uuids"]]
    if len(uuids) != len(set(uuids)):
        raise ValueError("The cluster maps more than one node to the same physical GPU")


def process_start(pid: int) -> str | None:
    try:
        # /proc comm may contain spaces; field 22 is index 19 after its closing parenthesis.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_node(plan: dict[str, Any], rank: int) -> None:
    record_path = Path(plan["run_dir"]) / f"node_{rank}.pid.json"
    if not record_path.is_file():
        return
    record = json.loads(record_path.read_text())
    pid = record["pid"]
    if record["run_id"] != plan["run_id"] or record["start"] is None or process_start(pid) != record["start"]:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def run_node(plan: dict[str, Any], rank: int) -> int:
    run_dir = Path(plan["run_dir"])
    record_path = run_dir / f"node_{rank}.pid.json"
    stopping = False
    process = None

    def forward_signal(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    if stopping:
        return 130
    process = subprocess.Popen(
        torchrun_command(plan, rank), env=child_env(plan), cwd=plan["root"], start_new_session=True
    )
    temporary = record_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"run_id": plan["run_id"], "pid": process.pid, "start": process_start(process.pid)})
    )
    temporary.replace(record_path)
    if stopping:
        stop_node(plan, rank)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            stop_node(plan, rank)
            process.wait(timeout=60)
        record_path.unlink(missing_ok=True)


def launch_nodes(plan: dict[str, Any]) -> int:
    processes = []
    streams = []
    interrupted = False

    def on_signal(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    prior = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = 0
    try:
        for rank in range(len(plan["nodes"])):
            if interrupted:
                break
            stream = (Path(plan["run_dir"]) / f"node_{rank}.log").open("w")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    node_command(plan, rank, "run-node"),
                    env=child_env(plan),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while processes and not interrupted:
            statuses = [process.poll() for process in processes]
            failures = [status for status in statuses if status not in (None, 0)]
            if failures:
                result = failures[0]
                break
            if all(status == 0 for status in statuses):
                break  # a successful rank/node exiting first must not kill its peers
            time.sleep(0.2)
        if interrupted:
            result = 130
    except BaseException:
        result = 1
        raise
    finally:
        if result:
            for rank, process in enumerate(processes):
                try:
                    subprocess.run(node_command(plan, rank, "stop-node"), env=child_env(plan), timeout=20, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    logger.warning(f"Cleanup could not reach node {rank}: {exc}")
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "run-node", "stop-node", "check-node"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action != "launch":
        plan = json.loads(args.path.read_text())
        if args.action == "stop-node":
            stop_node(plan, args.rank)
        elif args.action == "run-node":
            raise SystemExit(run_node(plan, args.rank))
        else:
            from ltx_trainer.omni_stage2_preflight import check_node

            report = check_node(Path(plan["config"]), plan["nproc"], child_env(plan))
            (Path(plan["run_dir"]) / f"preflight_node_{args.rank}.json").write_text(json.dumps(report, indent=2))
        return
    plan = build_plan(args.path.resolve(), dict(os.environ))
    if args.dry_run:
        sys.stdout.write(
            json.dumps({**plan, "commands": [torchrun_command(plan, r) for r in range(len(plan["nodes"]))]}, indent=2) + "\n"
        )
        return
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    preflight_nodes(plan)
    logger.info(
        f"Stage 2: {len(plan['nodes'])} nodes x {plan['nproc']} GPUs, global batch={plan['global_batch']}; logs={run_dir}"
    )
    if args.preflight:
        logger.info("All nodes passed preflight; no training launched.")
        return
    code = launch_nodes(plan)
    (run_dir / "result.json").write_text(json.dumps({"returncode": code, "run_id": plan["run_id"]}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
