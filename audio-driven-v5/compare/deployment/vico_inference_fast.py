"""Resume ViCo inference before loading torch or requesting a GPU.

Official models/precision are unchanged. New work goes to a separate directory.
"""
import argparse
import json
import time
from pathlib import Path
from vico_more_common import ROOT, WORK, read, save, sha, test_samples

FAST = ROOT / "experiments/p2_vico_inference_fast_20261110"


def verified_records(marker, expected_ids):
    try:
        return _verified_records(marker, expected_ids)
    except (OSError,ValueError,KeyError,TypeError):
        return None


def _verified_records(marker, expected_ids):
    if not marker.exists():
        return None
    data = read(marker)
    records = data.get("records", [])
    if not data.get("success") or len(records) != len(expected_ids):
        return None
    if {r["id"] for r in records} != set(expected_ids):
        return None
    for r in records:
        path = Path(r["path"])
        if not path.is_file() or sha(path) != r["sha256"]:
            return None
    # Verify linked manifests too; a completion flag alone is insufficient.
    for key in ("train_complete", "weight_audit", "demo_parity", "metrics"):
        digest = data.get(key + "_sha256")
        if digest and sha(marker.parent / (key.upper() + ".json")) != digest:
            return None
    return records


def clean_dualtalk_partial(out, weight_sha, samples):
    """Discard invalid local receipts; unlink archived symlinks without touching targets."""
    audit=out/'WEIGHT_AUDIT.json'
    try:compatible=read(audit)['checkpoint_sha256']==weight_sha
    except (OSError,ValueError,KeyError):compatible=False
    for sample in samples:
        dest=out/'predictions'/(sample['id']+'.npz');receipt=dest.with_suffix('.json')
        try:
            record=read(receipt)
            valid=compatible and record['id']==sample['id'] and dest.is_file() and sha(dest)==record['sha256']
            valid=valid and Path(record['path']).is_file() and sha(record['path'])==record['sha256']
        except (OSError,ValueError,KeyError):valid=False
        if not valid:
            receipt.unlink(missing_ok=True);dest.unlink(missing_ok=True)


def reusable(method, stage, out):
    if method in ('dystream', 'learning2listen'):
        from vico_fast_registry import reuse
        return reuse(method, out)
    expected = [s["id"] for s in test_samples()]
    requested_drivers = None
    if stage == "render":
        generation = out / "GENERATION_COMPLETE.json"
        if not generation.exists():
            generation = WORK / method / "GENERATION_COMPLETE.json"
        requested_drivers = {r["id"]: r["sha256"] for r in read(generation)["records"]}
    name = "RENDER_COMPLETE.json" if stage == "render" else (
        "COMPLETE.json" if method == "dualtalk" else "GENERATION_COMPLETE.json")
    for directory in dict.fromkeys([out, WORK / method]):
        records = verified_records(directory / name, expected)
        if records is not None:
            if stage == "render":
                candidate_drivers = {r["id"]: r["sha256"] for r in read(directory / "GENERATION_COMPLETE.json")["records"]}
                if candidate_drivers != requested_drivers:
                    continue
                # Archived render protocol links these videos to their drivers.
                proto = read(directory / "protocol.json")
                if sha(directory / "GENERATION_COMPLETE.json") != proto["generation_complete_sha256"]:
                    continue
                audit_path = directory / "RENDER_WEIGHT_AUDIT.json"
                if not audit_path.exists():
                    audit_path = WORK / "renderer/WEIGHT_AUDIT.json"
                renderer = WORK / "assets/renderer/face/epoch_00190_iteration_000400000_checkpoint.pt"
                if sha(renderer) != read(audit_path)["checkpoint_sha256"]:
                    continue
            elif method == "listenformer":
                checkpoint_sha = sha(WORK / method / "best.pt")
                if any(r["checkpoint_sha256"] != checkpoint_sha for r in records):
                    continue
            else:
                audit = read(directory / "WEIGHT_AUDIT.json")
                if sha(WORK / "assets/dualtalk/DualTalk.pth") != audit["checkpoint_sha256"]:
                    continue
            return records, directory / name
    return None, None


def generate_listenformer(out):
    import numpy as np
    import torch
    import vico_more_train as original
    from vico_more_features import OUT as FEATURES, dynamic
    source = WORK / "listenformer"
    complete = read(source / "TRAIN_COMPLETE.json")
    assert complete["success"] and complete["epochs"] == 200
    assert complete["formal_training_budget_complete"]
    checkpoint_sha = sha(source / "best.pt")
    assert checkpoint_sha == complete["best_checkpoint_sha256"]
    assert sha(FEATURES / "train_stats.npz") == read(FEATURES / "COMPLETE.json")["stats_sha256"]
    stats = dict(np.load(FEATURES / "train_stats.npz"))
    inputs = {r["id"]: r for r in read(FEATURES / "TEST_INPUTS.json")["records"]}
    model = None
    records = []
    for sample in test_samples():
        uid = sample["id"]
        record = None
        for directory in (out, source):
            receipt = directory / "predictions" / (uid + ".json")
            if receipt.exists():
                r = read(receipt)
                if r["checkpoint_sha256"] == checkpoint_sha and Path(r["path"]).is_file() and sha(r["path"]) == r["sha256"]:
                    record = r
                    break
        if record is not None:
            records.append(record)
            continue
        # First missing clip triggers a single model load for the whole run.
        if model is None:
            original.gpu_guard(12)
            model, _ = original.model_for("listenformer")
            ck = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
            assert ck["protocol_sha256"] == sha(source / "TRAIN_PROTOCOL.json")
            model.load_state_dict(ck["model"], strict=True)
            model.cuda().eval()
        t = time.perf_counter()
        inp = inputs[uid]
        assert sha(inp["speaker"]) == inp["speaker_sha256"]
        assert sha(inp["initial"]) == inp["initial_sha256"]
        speaker = np.load(inp["speaker"])
        initial = dict(np.load(inp["initial"]))
        audio = np.load(FEATURES / "test" / uid / "audio.npy")
        def norm(a, key):
            return torch.from_numpy(((a-stats[key+"_mean"])/stats[key+"_std"]).astype(np.float32)).cuda()
        with torch.inference_mode():
            pred = original.inference(model, "listenformer", norm(audio, "audio"),
                norm(dynamic(speaker), "dynamic"), norm(dynamic(initial), "dynamic")[0]).cpu().numpy()
        pred = pred * stats["dynamic_std"] + stats["dynamic_mean"]
        assert pred.shape == (len(audio), 73) and np.isfinite(pred).all()
        dest = out / "predictions" / (uid + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp, dynamic=pred, motion=pred[:, np.r_[3:67, 0:3, 67:70]],
            initial_coeff=initial["coeff"], initial_transform_params=initial["transform_params"])
        temp.replace(dest)
        r = dict(id=uid, path=str(dest), sha256=sha(dest), frames=len(audio),
            listener_frames_used=1, future_listener_dynamic=False, future_listener_crop=False,
            checkpoint_sha256=checkpoint_sha, seconds=time.perf_counter()-t)
        save(dest.with_suffix(".json"), r)
        records.append(r)
    # Keep the source provenance explicitly; do not alter the archived training run.
    import shutil
    shutil.copyfile(source / "TRAIN_COMPLETE.json", out / "TRAIN_COMPLETE.json")
    save(out / "GENERATION_COMPLETE.json", dict(success=True, outputs=len(records), records=records,
        train_complete_sha256=sha(source / "TRAIN_COMPLETE.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from vico_fast_registry import UNAVAILABLE, BASE, ALIASES
    p.add_argument("method", choices=["all", "listenformer", "dualtalk", "dystream", "learning2listen", *UNAVAILABLE, *ALIASES])
    p.add_argument("stage", choices=["generate", "render"])
    p.add_argument("--output-root", type=Path, default=FAST)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    a.method=ALIASES.get(a.method,a.method)
    if a.method in UNAVAILABLE:
        print(json.dumps(dict(method=a.method, state="awaiting_assets", reason=UNAVAILABLE[a.method],
            gpu_requested=False, optimized_inference=False)))
        return
    if a.method == "all":
        if a.stage != "generate":p.error("all uses generate; DyStream render is fused, L2L/DualTalk output motion")
        import subprocess
        import sys
        results=[]
        for method in ['listenformer', 'dualtalk', 'dystream', 'learning2listen', *UNAVAILABLE]:
            command=[sys.executable, str(Path(__file__).resolve()), method, a.stage, '--output-root', str(a.output_root.resolve())]
            if a.check_only:command.append('--check-only')
            result=subprocess.run(command, text=True, capture_output=True)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr)
                raise SystemExit(result.returncode)
            results.append(dict(method=method, result=json.loads(result.stdout.splitlines()[-1])))
        save(a.output_root / 'INVENTORY.json', dict(methods=results, excluded=['Avatar Forcing']))
        return
    if a.method != "listenformer" and a.stage == "render":
        p.error("render is for ListenFormer; DyStream render is fused into generate, L2L/DualTalk output native motion")
    out = a.output_root.resolve() / a.method
    if out in (WORK / a.method, BASE / a.method):
        p.error("use a new output root; archived results are read-only")
    start = time.perf_counter()
    records, marker = reusable(a.method, a.stage, out)
    if records is not None:
        print(json.dumps(dict(state="reused_verified", clips=len(records), manifest=str(marker),
            seconds=time.perf_counter()-start, model_loaded=False, gpu_requested=False)))
        return
    if a.check_only:
        print(json.dumps(dict(state="pending", seconds=time.perf_counter()-start)))
        return
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / (a.stage + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.method == "dystream":
            from vico_dystream_fast import generate
            generate(out)
        elif a.method == "learning2listen":
            import vico_l2l_fast
            vico_l2l_fast.prepare(out)
            vico_l2l_fast.generate()
        elif a.stage == "render":
            from vico_render_fast import render
            render(out)
        elif a.method == "listenformer":
            generate_listenformer(out)
        else:
            import vico_more_dualtalk as original
            import torch
            original.OUT = out
            # Preserve valid archived predictions when resuming a partial run.
            weight_sha = sha(original.WEIGHT)
            clean_dualtalk_partial(out, weight_sha, test_samples())
            audit_path = WORK / "dualtalk/WEIGHT_AUDIT.json"
            if audit_path.exists() and read(audit_path)["checkpoint_sha256"] == weight_sha:
                for sample in test_samples():
                    uid = sample["id"]
                    dest = out / "predictions" / (uid + ".npz")
                    receipt = WORK / "dualtalk/predictions" / (uid + ".json")
                    if not dest.exists() and receipt.exists():
                        record = read(receipt)
                        if Path(record["path"]).is_file() and sha(record["path"]) == record["sha256"]:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.unlink(missing_ok=True)
                            dest.symlink_to(record["path"])
                            save(dest.with_suffix(".json"), record)
            original.gpu_guard(16)
            torch.manual_seed(137)
            model = original.construct().eval().cuda()
            original.parity(model)
            from vico_compute_fast import install_dualtalk
            compute = install_dualtalk(model)
            original.generate(model)
            save(out / 'COMPUTE_FAST.json', dict(silent_feature_hits=compute.hits,
                silent_feature_misses=compute.misses, fallback_calls=compute.fallbacks,
                cached_lengths=len(compute.cache), precision='unchanged FP32'))
    print(json.dumps(dict(state="completed", seconds=time.perf_counter()-start, output=str(out))))


if __name__ == "__main__":
    main()
