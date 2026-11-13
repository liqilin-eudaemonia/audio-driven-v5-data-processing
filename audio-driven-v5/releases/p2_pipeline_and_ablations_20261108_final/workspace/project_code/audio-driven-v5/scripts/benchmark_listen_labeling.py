#!/usr/bin/env python3
"""Prepare, run and evaluate isolated listener annotation comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/listen_benchmark.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Read-only model shard and environment inventory; no GPU/network")
    check.add_argument("--output")
    for name in ("prepare", "review-template", "run", "worker", "processor-check"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--limit", type=int, default=50)
        if name in ("run", "worker", "processor-check"):
            sub.add_argument("--system", required=True)
        if name in ("run", "worker"):
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--retry-failed", action="store_true")
        if name == "run":
            sub.add_argument("--gpu", required=True, help="Explicit physical GPU ID(s), e.g. 2; no automatic GPU allocation")
            sub.add_argument("--timeout-seconds", type=int, default=21600)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--root", required=True)
    summarize.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--system-dir", required=True)
    evaluate.add_argument("--ground-truth", required=True)
    evaluate.add_argument("--partition", choices=("tuning", "heldout"), default="heldout")
    evaluate.add_argument("--output", required=True)
    paired = commands.add_parser("evaluate-pair")
    paired.add_argument("--visual-system-dir", required=True)
    paired.add_argument("--audio-system-dir", required=True)
    paired.add_argument("--ground-truth", required=True)
    paired.add_argument("--partition", choices=("tuning", "heldout"), default="heldout")
    paired.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    # A project-specific overlay is used only by this benchmark, never shared production services.
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    selected_system = next((s for s in config["systems"] if s["name"] == getattr(args, "system", None)), {})
    dependency_path = selected_system.get("dependency_path", config.get("dependency_path"))
    if dependency_path and Path(dependency_path).is_dir():
        sys.path.insert(0, dependency_path)
    from p2_listen.contracts import atomic_json, read
    from listen_benchmark.runner import validate_config, select_samples, review_template, run_system
    validate_config(config)

    if args.command == "check":
        from listen_benchmark.assets import preflight
        report = preflight(config)
        if args.output:
            atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["all_ready"] else 2
    if args.command == "summarize":
        from listen_benchmark.evaluate import summarize
        atomic_json(args.output, summarize(args.root))
        return 0
    if args.command == "evaluate":
        from listen_benchmark.evaluate import evaluate_system
        root = Path(args.system_dir)
        report = evaluate_system(read(root / "summary.json"), read(root / "inputs.json"),
                                 read(args.ground_truth), args.partition)
        atomic_json(args.output, report)
        return 0
    if args.command == "evaluate-pair":
        from listen_benchmark.evaluate import evaluate_pair
        atomic_json(args.output, evaluate_pair(args.visual_system_dir, args.audio_system_dir,
                                              read(args.ground_truth), args.partition))
        return 0
    if args.command == "run":
        system = next(s for s in config["systems"] if s["name"] == args.system)
        if args.timeout_seconds <= 0:
            raise ValueError("timeout-seconds must be positive")
        paths = [str(ROOT / "src")]
        if dependency_path:
            paths.insert(0, dependency_path)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONPATH=os.pathsep.join(paths))
        command = [system["python"], str(Path(__file__).resolve()), "--config", str(config_path), "worker",
                   "--manifest", str(Path(args.manifest).resolve()), "--output", str(Path(args.output).resolve()),
                   "--limit", str(args.limit), "--system", args.system, "--device", args.device]
        if args.retry_failed:
            command.append("--retry-failed")
        try:
            return subprocess.run(command, env=env, timeout=args.timeout_seconds).returncode
        except subprocess.TimeoutExpired:
            print("Benchmark worker exceeded deadline and was stopped; completed attempts are retained", file=sys.stderr)
            return 124

    samples = select_samples(args.manifest, args.limit)
    if args.command == "review-template":
        atomic_json(args.output, review_template(samples), immutable=True)
        return 0
    if args.command == "worker":
        report = run_system(config, args.system, samples, args.output, args.device, args.retry_failed)
        return 1 if any(r["status"] == "failed" for r in report["results"]) else 0
    if args.command == "processor-check":
        from listen_benchmark.processor_check import check_processor
        system = next(s for s in config["systems"] if s["name"] == args.system)
        atomic_json(args.output, check_processor(config, system, samples))
        return 0
    if args.command == "prepare":
        from listen_benchmark.media import prepare, uniform_video
        results = []
        for row in samples:
            try:
                media = prepare(row, "omni", config["media"])
                _, native = uniform_video(media, config["media"]["fps"])
                results.append({"sample_id": row["sample_id"], "status": "ready",
                                "media_hashes": row["media_hashes"], "input_evidence": media["meta"],
                                "qwen_native_timeline": native})
            except Exception as exc:
                results.append({"sample_id": row["sample_id"], "status": "failed", "error": str(exc)})
        atomic_json(args.output, {"results": results, "models_loaded": False, "network_called": False}, immutable=True)
        return 1 if any(r["status"] == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
