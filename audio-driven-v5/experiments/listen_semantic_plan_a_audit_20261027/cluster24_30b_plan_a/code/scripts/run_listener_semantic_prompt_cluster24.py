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
    if args.reuse_from:
        raise ValueError("Cross-protocol response reuse is disabled; resume only the same immutable output directory")

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
        for f in [WORKER_SCRIPT, ROOT / "src/listen_benchmark/semantic_prompt.py", ROOT / "src/p3_listen/prompts.py"]
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
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    proto_sha = digest(identity)
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != identity:
        raise ValueError("Output identity differs; use a fresh directory")
    write_json(out / "protocol.json", identity)
    controller_copy=out/'code/scripts/run_listener_semantic_prompt_cluster24.py'
    controller_copy.parent.mkdir(parents=True,exist_ok=True)
    controller_copy.write_bytes(Path(__file__).read_bytes())
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
        print(f"{len(failed_workers)} workers failed; collecting successful shards without marking full completion",flush=True)

    # Validate worker identities before creating explicitly attributed aggregate artifacts.
    (out / "attempts").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)

    states = {}
    for rank, node, host, gpu, sinput, sout, *_ in tasks:
        if not (sout/'summary.json').exists():
            report=next(w for w in worker_reports if w['rank']==rank)
            for job in (json.loads(line) for line in sinput.open() if line.strip()):
                states[job['job_id']]=dict(status='rejected',stage='worker',error=f"Worker failed or incomplete; inspect {report['log']}")
            continue
        worker_protocol=json.loads((sout / "protocol.json").read_text())
        worker_hash=digest(worker_protocol)
        if worker_protocol['code']!=code or worker_protocol['model']!=identity['model'] or worker_protocol['prompt_sha256']!=identity['prompt_sha256'] or worker_protocol['review_prompt_sha256']!=identity['review_prompt_sha256']:
            raise ValueError('Worker code/model/prompt differs from aggregation protocol')
        worker_jobs=[json.loads(line) for line in sinput.open() if line.strip()]
        if worker_protocol['input_sha256']!=digest(worker_jobs):raise ValueError('Worker input identity mismatch')
        shard_summary = json.loads((sout / "summary.json").read_text())
        for rej in shard_summary.get("rejected", []):
            states[rej["job_id"]] = {k: v for k, v in rej.items() if k != "job_id"}
        for att_file in (sout / "attempts").glob("*.json"):
            att = json.loads(att_file.read_text())
            if att['protocol_sha256']!=worker_hash:raise ValueError('Attempt worker identity mismatch')
            att['worker_protocol_sha256']=worker_hash
            att['worker_protocol_path']=str(sout / 'protocol.json')
            att["protocol_sha256"] = proto_sha
            write_json(out / "attempts" / att_file.name, att)
        if (sout / "results").is_dir():
            for res_file in (sout / "results").glob("*.json"):
                res = json.loads(res_file.read_text())
                jid = res_file.stem
                att_idx = res["provenance"]["attempt"]
                prop_raw = json.loads((out / "attempts" / f"{jid}.proposal.{att_idx}.json").read_text())
                rev_raw = json.loads((out / "attempts" / f"{jid}.review.{att_idx}.json").read_text())
                if res['provenance']['protocol_sha256']!=worker_hash:raise ValueError('Result worker identity mismatch')
                prov = dict(res["provenance"], protocol_sha256=proto_sha,worker_protocol_sha256=worker_hash)
                job_rec = next(j["record"] for j in jobs if j["job_id"] == jid)
                rebuilt = materialize(
                    job_rec,
                    compile_plan(job_rec, json.loads(prop_raw["raw"])),
                    json.loads(rev_raw["raw"]),
                    prov,
                )
                original_replay=dict(rebuilt,provenance=res['provenance'])
                if original_replay!=res:raise ValueError('Worker result differs from strict replay')
                write_json(out / "results" / res_file.name, rebuilt)
                (out / "prompts" / f"{jid}.txt").write_text(render_prompt(rebuilt))
                states[jid] = dict(status="accepted", result=rebuilt)

    elapsed = time.time() - t0
    summary = dict(
        status="passed" if not failed_workers and all(s.get("status") == "accepted" for s in states.values()) and len(states) == len(jobs) else "review_required",
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
        failed_workers=failed_workers,
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
