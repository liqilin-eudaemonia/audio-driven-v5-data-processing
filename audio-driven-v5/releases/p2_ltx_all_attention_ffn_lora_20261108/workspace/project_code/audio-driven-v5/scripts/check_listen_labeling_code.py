#!/usr/bin/env python3
"""Run offline annotation integration tests and save code-bound validation evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_listen.contracts import atomic_json, sha


def main():
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_listen_labeling*.py", "-v"],
                          cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
    paths = sorted((ROOT / "src/listen_labeling").glob("*.py")) + [
        ROOT / "scripts/label_listen_events.py", ROOT / "scripts/check_listen_labeling_code.py",
        ROOT / "scripts/check_listen_labeling_package.py", ROOT / "scripts/serve_listen_labeling_models.py",
        ROOT / "tests/test_listen_labeling.py", ROOT / "tests/test_listen_labeling_full.py",
        ROOT / "scripts/compare_listen_labeling_models.py", ROOT / "scripts/index_listen_labeling_full.py",
        ROOT / "scripts/run_listen_labeling_full.py", ROOT / "scripts/watch_listen_labeling_full.py",
        ROOT / "scripts/export_listen_labeling_full.py", ROOT / "configs/listen_labeling.json",
        ROOT / "src/p2_listen/annotations.py", ROOT / "src/p2_listen/contracts.py", ROOT / "src/p2_listen/dataset.py"]
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), passed=proc.returncode == 0,
                  code_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                  real_models_tested=False, training_started=False, synthetic_media=True,
                  transport="local mock HTTP server only", stdout=proc.stdout, stderr=proc.stderr)
    destination = ROOT / "experiments/listen_labeling_code_validation/result.json"
    atomic_json(destination, report)
    print(proc.stdout + proc.stderr)
    print(json.dumps(dict(passed=report["passed"], report=str(destination)), ensure_ascii=False))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
