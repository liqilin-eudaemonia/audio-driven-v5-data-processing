#!/usr/bin/env python3
"""Compare completed matched single-GPU/DDP runs without relaxing tolerances."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", type=Path, required=True, help="Run root containing runs/s0_<seed>")
    p.add_argument("--distributed", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--arm", choices=("E0", "E1", "both"), default="both")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    from p2_listen.contracts import read, atomic_json
    from p2_listen.training import checked_checkpoint
    torch.set_num_threads(4)
    results = {}
    for arm in (("E0", "E1") if a.arm == "both" else (a.arm,)):
        one = a.single / "runs" / ("s0_" + str(a.seed)) / arm
        multi = a.distributed / "runs" / ("s0_" + str(a.seed)) / arm
        ident1, identn = read(one / "run.json"), read(multi / "run.json")
        for key in ("seed", "s0", "budget", "observations", "targets", "labels", "statistics", "K_train",
                    "initialization", "source", "assets", "numerics", "debug_positive_events"):
            if ident1[key] != identn[key]:
                raise ValueError("Run identity differs: " + key)
        if ident1["world"] != 1 or identn["world"] < 2:
            raise ValueError("Expected single GPU vs multi GPU")
        c1, cn = dict(ident1["config"]), dict(identn["config"])
        c1.pop("output"); cn.pop("output")
        if c1 != cn:
            raise ValueError("Configs differ beyond output directory")
        def logs(folder, rank):
            rr = [json.loads(s) for s in (folder / f"train_rank{rank}.jsonl").read_text().splitlines() if s.strip()]
            if [r["step"] for r in rr] != list(range(1, a.steps + 1)):
                raise ValueError("Missing, duplicate or reordered steps")
            for r in rr:
                if not all(math.isfinite(v) for v in (r["base"], r["event"], *r["grad_norm"].values())):
                    raise ValueError("Nonfinite training statistics")
            return rr
        single = logs(one, 0)
        ranks = [logs(multi, rank) for rank in range(identn["world"])]
        for step, row in enumerate(single):
            if sorted(n for rr in ranks for n in rr[step]["ordinals"]) != row["ordinals"]:
                raise ValueError("Global sample ordinals differ")
            if any(rr[step]["event_known_bins"] != row["event_known_bins"] for rr in ranks):
                raise ValueError("Global supervision counts differ")
            if any(rr[step]["grad_norm"] != ranks[0][step]["grad_norm"] for rr in ranks):
                raise ValueError("Rank gradients were not synchronized")
        loss_error = abs(single[0]["event"] - ranks[0][0]["event"])
        x = checked_checkpoint(one / f"step_{a.steps:06d}.pt")
        y = checked_checkpoint(multi / f"step_{a.steps:06d}.pt")
        if x["identity"] != ident1 or y["identity"] != identn or x["step"] != a.steps or y["step"] != a.steps:
            raise ValueError("Checkpoint identity/step differs from run metadata")
        if set(x["model"]) != set(y["model"]):
            raise ValueError("Model parameter keys differ")
        errors = []
        for key, v in x["model"].items():
            w = y["model"][key]
            if v.shape != w.shape or v.dtype != w.dtype or not torch.isfinite(v).all() or not torch.isfinite(w).all():
                raise ValueError("Invalid parameter comparison: " + key)
            error = float((v - w).abs().max()) if v.is_floating_point() else float(not torch.equal(v, w))
            errors.append((key, error))
        errors.sort(key=lambda pair: pair[1], reverse=True)
        results[arm] = dict(passed=errors[0][1] <= 2e-5 and loss_error < 2e-6, steps=a.steps,
            world=identn["world"], global_batch=c1["training"]["global_batch"],
            same_initialization=True, same_sample_ordinals=True, same_global_class_counts=True,
            rank_grad_norms_equal=True, numerics=ident1["numerics"],
            first_step_event_loss_error=loss_error, model_max_abs_error=errors[0][1],
            model_absolute_tolerance=2e-5, largest_parameter_errors=errors[:10])
        del x, y
    report = dict(passed=all(v["passed"] for v in results.values()), arms=results)
    atomic_json(a.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
