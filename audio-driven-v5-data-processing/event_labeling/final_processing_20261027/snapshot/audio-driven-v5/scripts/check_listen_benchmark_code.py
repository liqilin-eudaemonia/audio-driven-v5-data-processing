#!/usr/bin/env python3
"""Offline tests and optional CPU processor checks; never starts a GPU annotator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_listen.contracts import atomic_json, read, sha
from listen_benchmark.assets import preflight
from listen_benchmark.runner import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processors", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "experiments/listen_benchmark_code_validation"))
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / "configs/listen_benchmark.json")
    code = """import unittest
loader=unittest.TestLoader(); suite=unittest.TestSuite()
for pattern in ('test_listen_benchmark.py','test_listen_labeling.py','test_listen_labeling_full.py'):
    suite.addTests(loader.discover('tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    unit = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
    (output / "unit_tests.log").write_text(unit.stdout + unit.stderr)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "unit_tests_passed": unit.returncode == 0,
              "code_sha256": source_identity(), "config_sha256": sha(ROOT / "configs/listen_benchmark.json"),
              "real_model_inference": False, "gpu_tasks_started": False, "processors": []}
    report["code_sha256"]["scripts/check_listen_benchmark_code.py"] = sha(__file__)
    report["code_sha256"]["tests/test_listen_benchmark.py"] = sha(ROOT / "tests/test_listen_benchmark.py")
    atomic_json(output / "assets.json", preflight(config))
    if args.processors:
        def check(system):
            result_path = output / (system["name"] + "_processor.json")
            command = [system["python"], str(ROOT / "scripts/benchmark_listen_labeling.py"), "processor-check",
                       "--manifest", str(ROOT / "experiments/listen_labeling_seamless_sample50_512/av_manifest.json"),
                       "--limit", "1", "--system", system["name"], "--output", str(result_path)]
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
                (output / (system["name"] + "_processor.log")).write_text(process.stdout + process.stderr)
                return {"system": system["name"], "passed": process.returncode == 0, "report": str(result_path)}
            except subprocess.TimeoutExpired:
                return {"system": system["name"], "passed": False, "error": "CPU processor deadline exceeded"}
        with ThreadPoolExecutor(max_workers=3) as pool:
            report["processors"] = list(pool.map(check, [s for s in config["systems"] if s["tier"] == "main"]))
    report["passed"] = report["unit_tests_passed"] and all(r["passed"] for r in report["processors"])
    atomic_json(output / "result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
