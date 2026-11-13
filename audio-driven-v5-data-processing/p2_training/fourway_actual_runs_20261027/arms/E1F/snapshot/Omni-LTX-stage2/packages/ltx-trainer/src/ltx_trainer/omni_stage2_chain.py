"""Chain a completed Planner SFT run into four-node Stage-2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / ".venv/bin/python"
CLUSTER_MODULE = "ltx_trainer.omni_stage2_cluster"
EXPECTED_HOSTS = [
    "172.29.252.24",
    "172.29.104.194",
    "172.29.112.43",
    "172.29.15.126",
]


@dataclass(frozen=True)
class PlannerCandidate:
    step: int
    valid_generated_plans: int
    generated_samples: int
    plan_ce: float
    checkpoint: str
    validation: str
    sha256: str

    @property
    def score(self) -> tuple[int, float, int]:
        return (self.valid_generated_plans, -self.plan_ce, self.step)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_from_validation(path: Path) -> PlannerCandidate:
    result = json.loads(path.read_text())
    step = int(result["step"])
    generated = int(result["generated_samples"])
    valid = int(result["valid_generated_plans"])
    plan_ce = float(result["plan_ce"])
    if generated <= 0 or not 0 <= valid <= generated or not math.isfinite(plan_ce):
        raise ValueError(f"Invalid Planner validation result: {path}")
    checkpoint = (
        path.parent.parent
        / "checkpoints"
        / f"omni_planner_sft_step_{step:05d}.safetensors"
    )
    checksum = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not checkpoint.is_file() or not checksum.is_file():
        raise FileNotFoundError(f"Validation has no complete checkpoint: {path}")
    expected = checksum.read_text().strip()
    actual = file_sha256(checkpoint)
    if expected != actual:
        raise ValueError(f"Planner checkpoint checksum mismatch: {checkpoint}")
    return PlannerCandidate(
        step=step,
        valid_generated_plans=valid,
        generated_samples=generated,
        plan_ce=plan_ce,
        checkpoint=str(checkpoint.resolve()),
        validation=str(path.resolve()),
        sha256=actual,
    )


def select_best_checkpoint(outputs: list[Path]) -> PlannerCandidate:
    candidates = []
    for output in outputs:
        for validation in sorted((output / "validation").glob("step_*.json")):
            candidates.append(candidate_from_validation(validation))
    if not candidates:
        raise FileNotFoundError("No complete Planner validation/checkpoint pairs found")
    expected_samples = max(item.generated_samples for item in candidates)
    complete = [
        item for item in candidates if item.generated_samples == expected_samples
    ]
    return max(complete, key=lambda item: item.score)


def build_stage2_config(
    template: Path,
    destination: Path,
    stage2_output: Path,
    candidate: PlannerCandidate,
    sft_config: Path,
    master_port: int,
) -> dict[str, Any]:
    raw = yaml.safe_load(template.read_text())
    sft = yaml.safe_load(sft_config.read_text())
    if int(raw["seed"]) == int(sft["seed"]):
        raise ValueError("Planner SFT and Stage 2 must use different seeds")
    stage2_metadata = [str(Path(path).resolve()) for path in raw["data"]["metadata_jsonls"]]
    if str(Path(sft["metadata_jsonl"]).resolve()) not in stage2_metadata:
        raise ValueError("Planner SFT and Stage 2 do not use the same training metadata")
    hosts = [node["host"] for node in raw["cluster"]["nodes"]]
    if hosts != EXPECTED_HOSTS or int(raw["cluster"]["nproc_per_node"]) != 8:
        raise ValueError("Stage 2 must use the audited four-node, 32-GPU cluster")
    raw["cluster"]["master_port"] = master_port
    raw["output_dir"] = str(stage2_output.resolve())
    raw["optimization"]["steps"] = 1000
    raw["checkpoints"]["no_resume"] = True
    raw["omni_joint"]["planner_init_checkpoint"] = candidate.checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    return raw


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sft_node_is_alive(
    host: str, port: int, output: Path, identity: Path
) -> bool:
    command = ["pgrep", "-af", "torch.distributed.run"]
    if host != EXPECTED_HOSTS[0]:
        command = [
            "ssh",
            "-i",
            str(identity),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            f"root@{host}",
            "pgrep -af torch.distributed.run",
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    marker = f"--master_port={port}"
    return any(
        marker in line and str(output.resolve()) in line
        for line in result.stdout.splitlines()
    )


def sft_cluster_is_alive(port: int, output: Path, identity: Path) -> bool:
    return all(
        sft_node_is_alive(host, port, output, identity) for host in EXPECTED_HOSTS
    )


def sft_cluster_is_released(port: int, output: Path, identity: Path) -> bool:
    return not any(
        sft_node_is_alive(host, port, output, identity) for host in EXPECTED_HOSTS
    )


def latest_logged_step(log: Path) -> int:
    if not log.is_file():
        return 0
    matches = re.findall(r'"step":\s*(\d+)', log.read_text(errors="replace"))
    return int(matches[-1]) if matches else 0


def available_master_port(preferred: int) -> int:
    for port in range(preferred, min(65536, preferred + 100)):
        with socket.socket() as probe:
            try:
                probe.bind((EXPECTED_HOSTS[0], port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No free Stage-2 rendezvous port from {preferred}")


def wait_for_sft(
    output: Path,
    log: Path,
    master_port: int,
    identity: Path,
    poll_seconds: int,
    status_path: Path,
) -> None:
    completed = output / "completed.json"
    while True:
        if completed.is_file():
            state = json.loads(completed.read_text())
            if state.get("status") != "completed" or int(state.get("step", 0)) != 1000:
                raise ValueError(f"Unexpected Planner completion record: {state}")
            release_deadline = time.monotonic() + 600
            while not sft_cluster_is_released(master_port, output, identity):
                atomic_json(
                    status_path,
                    {
                        "state": "waiting_for_planner_gpu_release",
                        "latest_step": 1000,
                        "sft_output": str(output.resolve()),
                        "updated_at": time.time(),
                    },
                )
                if time.monotonic() >= release_deadline:
                    raise RuntimeError(
                        "Planner SFT completed but did not release the cluster"
                    )
                time.sleep(5)
            return
        step = latest_logged_step(log)
        atomic_json(
            status_path,
            {
                "state": "waiting_for_planner_sft",
                "latest_step": step,
                "sft_output": str(output.resolve()),
                "updated_at": time.time(),
            },
        )
        if not sft_cluster_is_alive(master_port, output, identity):
            time.sleep(10)
            if not completed.is_file() and not sft_cluster_is_alive(
                master_port, output, identity
            ):
                raise RuntimeError(
                    f"Planner SFT stopped before completion at approximately step {step}"
                )
        time.sleep(poll_seconds)


def launch_stage2(
    config: Path,
    stage2_output: Path,
    identity: Path,
    controller_log: Path,
    status_path: Path,
) -> int:
    environment = dict(os.environ)
    environment.update(
        {
            "SSH_IDENTITY_FILE": str(identity),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NPROC_PER_NODE": "8",
            "FSDP_ENABLE": "1",
            "OMNI_STAGE2_AUDIT_GRADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    before = set((stage2_output / "cluster_runs").glob("*"))
    with controller_log.open("a") as stream:
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                CLUSTER_MODULE,
                "launch",
                str(config),
            ],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        run_dir = None
        deadline = time.monotonic() + 1200
        while process.poll() is None and time.monotonic() < deadline:
            created = set((stage2_output / "cluster_runs").glob("*")) - before
            if created:
                run_dir = max(created, key=lambda path: path.stat().st_mtime)
                node_records = [
                    run_dir / f"node_{rank}.pid.json" for rank in range(4)
                ]
                if all(record.is_file() for record in node_records):
                    break
            time.sleep(2)
        if run_dir is None or not all(
            (run_dir / f"node_{rank}.pid.json").is_file() for rank in range(4)
        ):
            process.terminate()
            process.wait(timeout=60)
            raise RuntimeError("Stage 2 controller did not launch all four nodes")
        atomic_json(
            status_path,
            {
                "state": "stage2_running",
                "controller_pid": process.pid,
                "cluster_run": str(run_dir.resolve()),
                "stage2_output": str(stage2_output.resolve()),
                "updated_at": time.time(),
            },
        )
        return_code = process.wait()
    atomic_json(
        status_path,
        {
            "state": "stage2_completed" if return_code == 0 else "stage2_failed",
            "returncode": return_code,
            "cluster_run": str(run_dir.resolve()),
            "stage2_output": str(stage2_output.resolve()),
            "updated_at": time.time(),
        },
    )
    return return_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-output", type=Path, required=True)
    parser.add_argument("--watch-log", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, action="append", default=[])
    parser.add_argument("--sft-config", type=Path, required=True)
    parser.add_argument("--sft-master-port", type=int, required=True)
    parser.add_argument("--stage2-template", type=Path, required=True)
    parser.add_argument("--stage2-output", type=Path, required=True)
    parser.add_argument("--stage2-master-port", type=int, default=29862)
    parser.add_argument("--ssh-identity", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    status = args.state_dir / "status.json"
    try:
        wait_for_sft(
            args.watch_output,
            args.watch_log,
            args.sft_master_port,
            args.ssh_identity,
            args.poll_seconds,
            status,
        )
        outputs = [*args.candidate_output, args.watch_output]
        candidate = select_best_checkpoint(outputs)
        selection = args.state_dir / "selected_planner_checkpoint.json"
        atomic_json(selection, asdict(candidate))
        generated = args.state_dir / "stage2_selected_config.yaml"
        stage2_master_port = available_master_port(args.stage2_master_port)
        build_stage2_config(
            args.stage2_template,
            generated,
            args.stage2_output,
            candidate,
            args.sft_config,
            stage2_master_port,
        )
        atomic_json(
            status,
            {
                "state": "launching_stage2",
                "selected_checkpoint": candidate.checkpoint,
                "selected_step": candidate.step,
                "stage2_master_port": stage2_master_port,
                "updated_at": time.time(),
            },
        )
        code = launch_stage2(
            generated,
            args.stage2_output,
            args.ssh_identity,
            args.state_dir / "stage2_controller.log",
            status,
        )
        raise SystemExit(code)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            atomic_json(
                status,
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                },
            )
        raise


if __name__ == "__main__":
    main()
