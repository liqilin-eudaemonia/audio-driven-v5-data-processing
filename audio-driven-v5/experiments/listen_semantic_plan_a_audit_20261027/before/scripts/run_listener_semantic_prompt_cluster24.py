#!/usr/bin/env python3
"""Run listener semantic prompt consolidation across 3 nodes x 8 GPUs (24 GPUs total)."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_benchmark.semantic_prompt import (
    PLAN_PROMPT,
    REVIEW_PROMPT,
    VERSION,
    compile_plan,
    digest,
    materialize,
    render_prompt,
    source_payload,
)

KEY = "/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519"
NODES = [
    ("node17", "172.29.15.91"),
    ("node12", "172.29.252.11"),
    ("node4", "172.29.225.51"),
]
VLLM_PYTHON = "/private/software/conda/envs/vllm/bin/python"
WORKER_SCRIPT = ROOT / "scripts/run_listener_semantic_prompt.py"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run_shard(rank, node, host, gpu, shard_input, shard_out, model, batch, memory, attempts, log_path):
    cmd = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 "
        f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "
        f"{shlex.quote(VLLM_PYTHON)} {shlex.quote(str(WORKER_SCRIPT))} "
        f"--inputs {shlex.quote(str(shard_input))} "
        f"--output {shlex.quote(str(shard_out))} "
        f"--model {shlex.quote(str(model))} "
        f"--batch {batch} --memory {memory} --attempts {attempts}"
    )
    ssh_cmd = [
        "ssh",
        "-i",
        KEY,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        f"root@{host}",
        cmd,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(ssh_cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return {
        "rank": rank,
        "node": node,
        "host": host,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - t0,
        "log": str(log_path),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/private/ckpts/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--memory", type=float, default=0.48)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--reuse-from", type=Path)
    args = p.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in args.inputs.open() if line.strip()]
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job ID")
    for j in jobs:
        source_payload(j["record"])
        if j["provenance"]["role"] == "current_postprocessed" and "audio_postprocess" not in j["record"]:
            raise ValueError("Current prompt lacks postprocessed source")

    code = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [WORKER_SCRIPT, ROOT / "src/listen_benchmark/semantic_prompt.py"]
    }
    model = Path(args.model).resolve()
    model_files = [f for f in model.iterdir() if f.is_file() and f.suffix in (".safetensors", ".json")]
    identity = dict(
        version=VERSION,
        code=code,
        model=str(model),
        model_files={f.name: dict(size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in sorted(model_files)},
        prompt_sha256=digest(PLAN_PROMPT),
        review_prompt_sha256=digest(REVIEW_PROMPT),
        input_sha256=digest(jobs),
        temperature=0,
        attempts=args.attempts,
    )
    proto_sha = digest(identity)
    write_json(out / "protocol.json", identity)
    for name in code:
        destination = out / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())

    slots = []
    for node, host in NODES:
        for gpu in range(8):
            slots.append((len(slots), node, host, gpu))
    assert len(slots) == 24

    shards_dir = out / "shards"
    shards_dir.mkdir(exist_ok=True)
    tasks = []
    for rank, node, host, gpu in slots:
        shard_jobs = jobs[rank::24]
        if not shard_jobs:
            continue
        sdir = shards_dir / f"shard_{rank:02d}_{node}_gpu{gpu}"
        sdir.mkdir(parents=True, exist_ok=True)
        sinput = sdir / "inputs.jsonl"
        sinput.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in shard_jobs))
        sout = sdir / "run"
        if args.reuse_from and (args.reuse_from / "attempts").is_dir():
            from listen_benchmark.semantic_prompt import validate_review
            for j in shard_jobs:
                jid = j["job_id"]
                rec = j["record"]
                for att in (0, 1, 2):
                    pp = args.reuse_from / "attempts" / f"{jid}.proposal.{att}.json"
                    if pp.is_file():
                        try:
                            p_entry = json.loads(pp.read_text())
                            ans = compile_plan(rec, json.loads(p_entry["raw"]))
                            p_entry["attempt"] = 0
                            write_json(sout / "attempts" / f"{jid}.proposal.0.json", p_entry)
                            rp = args.reuse_from / "attempts" / f"{jid}.review.{att}.json"
                            if rp.is_file():
                                r_entry = json.loads(rp.read_text())
                                r_obj = json.loads(r_entry["raw"])
                                if validate_review(rec, ans, r_obj):
                                    r_entry["attempt"] = 0
                                    write_json(sout / "attempts" / f"{jid}.review.0.json", r_entry)
                            break
                        except Exception:
                            pass
        slog = sdir / "worker.log"
        tasks.append((rank, node, host, gpu, sinput, sout, args.model, args.batch, args.memory, args.attempts, slog))

    write_json(
        out / "cluster_launch.json",
        {
            "nodes": [{"node": n, "host": h, "gpus": 8} for n, h in NODES],
            "total_gpus": 24,
            "active_shards": len(tasks),
            "samples": len(jobs),
            "model": str(model),
            "protocol_sha256": proto_sha,
            "started_at": time.time(),
        },
    )
    print(f"Launching {len(tasks)} GPU workers across 3 nodes (24 GPUs)...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(run_shard, *t) for t in tasks]
        worker_reports = [f.result() for f in futures]

    write_json(out / "cluster_workers.json", worker_reports)
    failed_workers = [w for w in worker_reports if w["returncode"] != 0]
    if failed_workers:
        raise RuntimeError(f"Some GPU workers failed: {failed_workers}")

    # Merge attempts, results, prompts, and rewrite protocol_sha256 to match top-level identity
    (out / "attempts").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)

    states = {}
    for rank, node, host, gpu, sinput, sout, *_ in tasks:
        shard_summary = json.loads((sout / "summary.json").read_text())
        for rej in shard_summary.get("rejected", []):
            states[rej["job_id"]] = {k: v for k, v in rej.items() if k != "job_id"}
        for att_file in (sout / "attempts").glob("*.json"):
            att = json.loads(att_file.read_text())
            att["protocol_sha256"] = proto_sha
            write_json(out / "attempts" / att_file.name, att)
        if (sout / "results").is_dir():
            for res_file in (sout / "results").glob("*.json"):
                res = json.loads(res_file.read_text())
                jid = res_file.stem
                att_idx = res["provenance"]["attempt"]
                prop_raw = json.loads((out / "attempts" / f"{jid}.proposal.{att_idx}.json").read_text())
                rev_raw = json.loads((out / "attempts" / f"{jid}.review.{att_idx}.json").read_text())
                prov = dict(res["provenance"], protocol_sha256=proto_sha)
                job_rec = next(j["record"] for j in jobs if j["job_id"] == jid)
                rebuilt = materialize(
                    job_rec,
                    compile_plan(job_rec, json.loads(prop_raw["raw"])),
                    json.loads(rev_raw["raw"]),
                    prov,
                )
                write_json(out / "results" / res_file.name, rebuilt)
                (out / "prompts" / f"{jid}.txt").write_text(render_prompt(rebuilt))
                states[jid] = dict(status="accepted", result=rebuilt)

    elapsed = time.time() - t0
    summary = dict(
        status="passed" if all(s.get("status") == "accepted" for s in states.values()) and len(states) == len(jobs) else "review_required",
        samples=len(jobs),
        accepted=sum(s.get("status") == "accepted" for s in states.values()),
        rejected=[dict(job_id=k, **v) for k, v in states.items() if v.get("status") != "accepted"],
        input_visual=sum(len(source_payload(j["record"])["visual_sources"]) for j in jobs),
        output_visual_accepted=sum(
            sum(e["modality"] == "visual" for e in s["result"]["events"])
            for s in states.values()
            if s.get("status") == "accepted"
        ),
        elapsed_seconds=elapsed,
        cluster_gpus=24,
        raw_annotations_modified=False,
        canonical_cache_updated=False,
        model_review_is_not_human_validation=True,
    )
    write_json(out / "summary.json", summary)
    if summary["status"] == "passed":
        write_json(out / "COMPLETE.json", dict(**summary, protocol_sha256=proto_sha))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
