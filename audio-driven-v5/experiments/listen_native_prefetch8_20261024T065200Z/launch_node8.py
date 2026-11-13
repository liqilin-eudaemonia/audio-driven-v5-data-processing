"""Eight isolated, work-stealing native AV replicas for the authorized node8 run."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "src"))
from p2_listen.contracts import atomic_json, sha


def snapshot():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw", "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(("index", "memory_used_mib", "memory_total_mib", "gpu_percent", "memory_percent", "power_w"), map(float, line.split(",")))) for line in text.strip().splitlines()]


def main():
    assert socket.gethostname() == "dedicated-developjob-dev-machine-hi3wi", "Wrong node"
    with (ROOT / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, expected in json.loads((ROOT / "code_hashes.json").read_text()).items():
            assert sha(CODE / name) == expected, name
        before = snapshot()
        assert len(before) == 8 and all(g["memory_used_mib"] < 256 for g in before), before
        atomic_json(ROOT / "resources_before.json", before)
        cfg = json.loads((ROOT / "config.json").read_text())
        workers = []
        for gpu in range(8):
            worker = f"node8-g{gpu}"
            cmd = [cfg["systems"][0]["python"], str(CODE / "scripts/run_listen_benchmark_vllm.py"),
                   "--config", str(ROOT / "config.json"), "--manifest", str(ROOT / "manifest.json"),
                   "--output", str(ROOT / "results" / f"gpu{gpu}"), "--systems", ",".join(s["name"] for s in cfg["systems"]),
                   "--worker-id", worker, "--clients", "1" if gpu < 4 else "4", "--batch-size", "1", "--memory", ".86"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HF_ENABLE_PARALLEL_LOADING="true",
                       HF_PARALLEL_LOADING_WORKERS="4")
            env["LD_LIBRARY_PATH"] = "/private/software/conda/envs/vllm/lib:" + env.get("LD_LIBRARY_PATH", "")
            logfile = ROOT / (worker + ".log")
            with logfile.open("x") as log:
                proc = subprocess.Popen(cmd, env=env, cwd=CODE, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((proc, dict(pid=proc.pid, gpu=gpu, worker=worker, command=cmd, log=str(logfile))))
        atomic_json(ROOT / "launch.json", [r for _, r in workers])
        atomic_json(ROOT / "status.json", dict(state="running", gpu_inference=True, workers=8, expected_requests=64))
        previous = None
        with (ROOT / "resources.jsonl").open("a", buffering=1) as log:
            while any(proc.poll() is None for proc, _ in workers):
                cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
                current = (sum(cpu[:8]), cpu[3] + cpu[4])
                percent = None if previous is None else 100 * (1 - (current[1]-previous[1]) / max(1,current[0]-previous[0]))
                previous = current
                record = dict(utc=datetime.now(timezone.utc).isoformat(), gpus=snapshot(), cpu_percent=percent,
                              worker_states={str(gpu):json.loads((ROOT / "results" / f"gpu{gpu}" / "workers" / f"node8-g{gpu}.json").read_text()) for gpu in range(8) if (ROOT / "results" / f"gpu{gpu}" / "workers" / f"node8-g{gpu}.json").exists()}, loadavg=os.getloadavg(), workers_running=sum(proc.poll() is None for proc, _ in workers))
                log.write(json.dumps(record) + "\n")
                atomic_json(ROOT / "resources_latest.json", record)
                time.sleep(5)
        results = [r for path in (ROOT / "results").glob("gpu*/items/*/*/summary.json")
                   for r in json.loads(path.read_text())["results"]]
        exits = {row["worker"]: proc.returncode for proc, row in workers}
        atomic_json(ROOT / "status.json", dict(state="completed" if len(results)==64 and not any(exits.values()) else "incomplete",
                    completed_requests=len(results), expected_requests=64, gpu_inference=True, worker_exits=exits,
                    schema_pass=sum(r["status"]=="schema_pass" for r in results), human_verified=False, training_admitted=False))
        atomic_json(ROOT / "resources_after.json", snapshot())
        print((ROOT / "status.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
