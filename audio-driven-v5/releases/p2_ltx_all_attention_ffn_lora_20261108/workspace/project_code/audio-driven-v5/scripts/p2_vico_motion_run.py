"""Own the four-worker extraction lifecycle and CPU-only scoring/audit."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/"experiments/p2_vico_motion8000_20261106"


def save(name, data):
    temp = WORK/(name+".tmp");temp.write_text(json.dumps(data, indent=2)+"\n");temp.replace(WORK/name)


def run():
    WORK.mkdir(exist_ok=True)
    with (WORK/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert json.loads((WORK/"SMOKE_COMPLETE.json").read_text())["streams"] == 3
        assert json.loads((WORK/"BATCH_PARITY.json").read_text())["passed"] is True
        gpu = subprocess.check_output(["nvidia-smi", "-i", "7", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        assert int(gpu.strip()) < 2000, f"GPU7 unexpectedly occupied: {gpu}"
        start = time.time();stage = ["extracting"];stop = threading.Event();telemetry = []
        def monitor():
            with (WORK/"gpu7.jsonl").open("a", buffering=1) as f:
                while not stop.is_set():
                    text = subprocess.check_output(["nvidia-smi", "-i", "7", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True)
                    mem, util, power = map(float, text.strip().split(","))
                    row = dict(at=time.time(), stage=stage[0], memory_mib=mem, utilization_pct=util, power_w=power)
                    telemetry.append(row);f.write(json.dumps(row)+"\n");stop.wait(1)
        thread = threading.Thread(target=monitor, daemon=True);thread.start()
        processes, handles = [], []
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="7", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
            for i in range(4):
                handle = (WORK/f"extract_{i:02d}.log").open("w");handles.append(handle)
                cmd = [sys.executable, str(ROOT/"scripts/p2_vico_motion_extract.py"), "--work", str(WORK), "--shard", str(i), "--shards", "4"]
                processes.append(subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
            save("LAUNCH.json", dict(at=start, controller_pid=os.getpid(), worker_pids=[p.pid for p in processes], workers=4, physical_gpu=7))
            while any(p.poll() is None for p in processes):
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:raise RuntimeError(f"Extraction worker failed: {failures}")
                states = []
                for i in range(4):
                    path = WORK/f"status_{i:02d}.json"
                    if path.exists():states.append(json.loads(path.read_text()))
                save("status.json", dict(stage=stage[0], workers=states, elapsed_s=time.time()-start, at=time.time()))
                time.sleep(2)
            assert all(p.returncode == 0 for p in processes)
            receipts = [json.loads((WORK/f"EXTRACTION_COMPLETE_{i:02d}.json").read_text()) for i in range(4)]
            assert sum(x["streams"] for x in receipts) == 129
            save("EXTRACTION_COMPLETE.json", dict(streams=129, frames=sum(x["frames"] for x in receipts),
                missing_face_frames=sum(x["missing_face_frames"] for x in receipts), seconds=time.time()-start))
            stage[0] = "scoring";save("status.json", dict(stage=stage[0], at=time.time()))
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            for script, log in [("p2_vico_motion_metrics.py", "metrics.log"), ("audit_p2_vico_motion.py", "audit.log")]:
                stage[0] = "scoring" if log == "metrics.log" else "auditing"
                save("status.json", dict(stage=stage[0], at=time.time()))
                with (WORK/log).open("w") as f:
                    subprocess.run([sys.executable, str(ROOT/"scripts"/script), "--work", str(WORK)],
                                   env=cpu_env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            assert json.loads((WORK/"FINAL_AUDIT.json").read_text())["passed"] is True
            save("COMPLETE.json", dict(at=time.time(), seconds=time.time()-start, clips=43, streams=129,
                frames_per_stream=16288, final_audit="FINAL_AUDIT.json", results="RESULTS.json"))
            stage[0] = "complete";save("status.json", dict(stage="complete", at=time.time(), seconds=time.time()-start))
        except Exception as e:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            stage[0] = "failed";save("CONTROLLER_FAILURE.json", dict(error=str(e), at=time.time()))
            save("status.json", dict(stage="failed", error=str(e), at=time.time()))
            raise
        finally:
            stop.set();thread.join(timeout=10)
            for handle in handles:handle.close()
            rows = [r for r in telemetry if r["stage"] == "extracting"]
            if rows:save("GPU_USAGE.json", dict(samples=len(rows), mean_utilization_pct=statistics.mean(r["utilization_pct"] for r in rows),
                median_utilization_pct=statistics.median(r["utilization_pct"] for r in rows), peak_memory_mib=max(r["memory_mib"] for r in rows),
                peak_power_w=max(r["power_w"] for r in rows), scope="four extraction workers, physical GPU7 only"))


if __name__ == "__main__":run()
