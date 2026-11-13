#!/usr/bin/env python3
"""Freeze a small native audiovisual prompt comparison without starting inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_benchmark.prompts import annotation_prompt
from listen_benchmark.runner import select_samples, validate_config
from p2_listen.contracts import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feedback", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out, baseline = Path(a.output).resolve(), Path(a.baseline).resolve()
    if out.exists():
        raise ValueError("Use a new experiment directory")
    feedback = json.loads(Path(a.feedback).read_text())
    samples = select_samples(baseline / "manifest50.json", None)
    by_id = {r["sample_id"]: r for r in samples}
    ids = list(dict.fromkeys(r["sample_id"] for r in feedback["records"]))
    if not ids or any(sid not in by_id for sid in ids):
        raise ValueError("Feedback samples must belong to the baseline manifest")
    if feedback["run_id"] != baseline.name or feedback["human_ground_truth"] is not False:
        raise ValueError("Expected comparison preferences for the named baseline, not event ground truth")
    extra = random.Random(20261024).sample([r for r in samples if r["sample_id"] not in ids], 4)
    selected = [by_id[sid] for sid in ids] + extra
    plan = json.loads((baseline / "full50_plan.json").read_text())
    cfg = json.loads(Path(plan["config"]).read_text())
    original = next(s for s in cfg["systems"] if s["name"] == "qwen3_omni_native")
    cfg["systems"] = [dict(original, name=name, prompt_variant=variant, annotation_stages=["discovery"])
                      for name, variant in (("native_baseline", "baseline"),
                                            ("native_occurrences_v1", "backchannel_occurrences_v1"))]
    validate_config(cfg)
    out.mkdir(parents=True)
    atomic_json(out / "human_feedback.json", feedback, immutable=True)
    atomic_json(out / "manifest.json", selected, immutable=True)
    atomic_json(out / "config.json", cfg, immutable=True)
    atomic_json(out / "plan.json", dict(
        created_at=datetime.now(timezone.utc).isoformat(), baseline=str(baseline),
        experiment="prompt_only_native_av_pilot", prompt_variants=[s["prompt_variant"] for s in cfg["systems"]],
        expected_requests=len(selected) * 2, feedback_sample_ids=ids,
        random_sample_ids=[r["sample_id"] for r in extra], random_seed=20261024,
        human_ground_truth=False, heldout_evaluation=False, training_admitted=False,
        limitations=["All samples belong to the previously inspected development pool",
                     "Feedback is approximate preference/correction text, not complete event ground truth",
                     "Prompt changes cannot by themselves establish accurate temporal localization"],
        comparison=["per-occurrence audio splitting", "head-movement omissions", "audio/visual separation",
                    "whole-clip intervals", "duplicate fragments", "raw finish and schema failures"],
        unchanged=["weights", "8 fps listener crop", "independent listener WAV", "native AV time grid",
                   "4096 token budget", "temperature zero", "JSON schema"],
        user_feedback_sent_to_model=False), immutable=True)
    for row in selected:
        for system in cfg["systems"]:
            dest = out / "prompts" / row["sample_id"] / (system["name"] + ".txt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(annotation_prompt(system, row["duration_s"], "discovery"))
    code = out / "code"
    shutil.copytree(ROOT / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "scripts").mkdir()
    for name in ("run_listen_benchmark_vllm.py", "benchmark_listen_labeling.py",
                 "prepare_listen_prompt_pilot.py", "run_listen_prompt_pilot.py"):
        shutil.copy2(ROOT / "scripts" / name, code / "scripts" / name)
    (code / "tests").mkdir()
    shutil.copy2(ROOT / "tests/test_listen_benchmark.py", code / "tests/test_listen_benchmark.py")
    (code / "configs").mkdir()
    shutil.copy2(ROOT / "configs/listen_benchmark.json", code / "configs/listen_benchmark.json")
    hashes = {str(f.relative_to(code)): sha(f) for f in code.rglob("*") if f.is_file()}
    atomic_json(out / "code_hashes.json", hashes, immutable=True)
    atomic_json(out / "status.json", dict(state="prepared_not_started", gpu_inference=False))
    print(json.dumps(dict(output=str(out), samples=len(selected), requests=len(selected)*2,
                          gpu_inference=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
