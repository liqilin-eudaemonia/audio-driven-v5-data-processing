#!/usr/bin/env python3
"""Run the frozen native AV prompt pilot on one explicitly selected idle GPU."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def gpu_snapshot():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    return [dict(index=int(r[0]), uuid=r[1].strip(), free_mib=float(r[2]),
                 total_mib=float(r[3]), utilization=float(r[4])) for r in csv.reader(io.StringIO(text))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    root = Path(a.run).resolve()
    code = root / "code"
    sys.path.insert(0, str(code / "src"))
    from p2_listen.contracts import atomic_json, sha
    for name, expected in json.loads((root / "code_hashes.json").read_text()).items():
        if sha(code / name) != expected:
            raise ValueError("Frozen code changed: " + name)
    cfg = json.loads((root / "config.json").read_text())
    devices = gpu_snapshot()
    device = next(g for g in devices if g["index"] == a.gpu)
    occupied = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True).splitlines()
    ready = (device["uuid"] not in [x.strip() for x in occupied] and device["utilization"] < 5
             and device["free_mib"] >= device["total_mib"] * .86 + 1024)
    stamp = datetime.now(timezone.utc).isoformat()
    atomic_json(root / "resource_check.json", dict(utc=stamp, host=os.uname().nodename,
                gpu=device, idle_and_sufficient=ready, check_only=a.check_only))
    if not ready:
        print("Selected GPU is occupied or lacks sufficient free memory; no inference started.")
        return 2
    if a.check_only:
        print("GPU preflight passed; no inference started.")
        return 0
    systems = ",".join(s["name"] for s in cfg["systems"])
    worker = f"prompt-pilot-g{a.gpu}"
    cmd = [cfg["systems"][0]["python"], str(code / "scripts/run_listen_benchmark_vllm.py"),
           "--config", str(root / "config.json"), "--manifest", str(root / "manifest.json"),
           "--output", str(root / "results"), "--systems", systems,
           "--worker-id", worker, "--batch-size", "1", "--clients", "4", "--memory", ".86"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HF_ENABLE_PARALLEL_LOADING="true",
               HF_PARALLEL_LOADING_WORKERS="4")
    env["LD_LIBRARY_PATH"] = "/private/software/conda/envs/vllm/lib:" + env.get("LD_LIBRARY_PATH", "")
    logfile = root / ("worker-" + str(time.time_ns()) + ".log")
    with logfile.open("x") as log:
        proc = subprocess.Popen(cmd, cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(root / "process.json", dict(pid=proc.pid, command=cmd, host=os.uname().nodename,
                    gpu=a.gpu, log=str(logfile), started_utc=stamp))
        atomic_json(root / "status.json", dict(state="running", gpu_inference=True))
        try:
            with (root / "resources.jsonl").open("a", buffering=1) as resources:
                while proc.poll() is None:
                    resources.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                    gpus=gpu_snapshot(), loadavg=os.getloadavg())) + "\n")
                    time.sleep(5)
        except BaseException:
            proc.terminate()
            proc.wait(timeout=60)
            raise
    summaries = list((root / "results/items").glob("*/*/summary.json"))
    results = [r for path in summaries for r in json.loads(path.read_text())["results"]]
    expected = json.loads((root / "plan.json").read_text())["expected_requests"]
    complete = proc.returncode == 0 and len(results) == expected
    atomic_json(root / "status.json", dict(state="completed" if complete else "incomplete",
                gpu_inference=True, exit_code=proc.returncode, completed_requests=len(results),
                expected_requests=expected, schema_pass=sum(r["status"] == "schema_pass" for r in results),
                human_verified=False, training_admitted=False))
    print((root / "status.json").read_text())
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
