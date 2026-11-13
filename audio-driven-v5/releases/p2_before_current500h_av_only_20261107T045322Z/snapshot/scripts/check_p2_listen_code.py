#!/usr/bin/env python3
"""Run only P2 code tests, with CUDA hidden, and save machine-readable evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import sys
import unittest
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tests"))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/p2_listen_v2_migration_validation/unit_tests.json")
    args=parser.parse_args()
    from p2_listen.contracts import atomic_json,sha
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_p2_listen_*.py")
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files=list((ROOT/"src/p2_listen").glob("*.py"))+list((ROOT/"tests").glob("test_p2_listen_*.py"))
    files+=list((ROOT/"scripts").glob("*p2_listen*.py"))+list((ROOT/"configs").glob("p2_listen*.json"))
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),python=sys.version,tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful(),
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],real_models_loaded=False,real_training_started=False,
                code={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    atomic_json(args.output,report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
