"""Unified measurement/reclassification entry, bounded work queue and atomic exports."""
import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import time
import uuid
from pathlib import Path
from . import VERSION
from .io import read, write, file_hash, canonical_hash, input_rows
from .rules import analyze_focus, propose_crop
from .decision import decide, evidence_index

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task"


def load_policy(path):
    p = read(path)
    if p.get("version") != VERSION or p.get("backend") != "cpu" or p.get("clarity_operator") != "OR":
        raise ValueError("This release requires the CPU / OR policy")
    # Frame selection and detector timestamps are frozen at the verified sampling rate.
    if p["sampling_fps"] != 5 or p["median_samples"] != 3:
        raise ValueError("This release is verified at 5 fps with a 3-sample median")
    if p["min_continuous_low_s"] != 2 or p["min_low_fraction"] != .25:
        raise ValueError("Changing the accepted 2 s / 25% rule requires a new version")
    if not 1 <= p["crop_max_side_growth"] <= 1.05:
        raise ValueError("Crop enlargement exceeds the agreed experiment cap")
    return p


def result_path(out, sid):
    return out / "results" / canonical_hash(sid)[:2] / (sid + ".json")


def check_source(row, checked):
    s = row["source"]
    path, expected = s["source_result"], s["source_result_sha256"]
    if path not in checked:
        checked[path] = file_hash(path)
    if checked[path] != expected:
        raise ValueError("Original J result changed: " + path)


def work(task):
    row, policy, model, out, measurement_hash = task
    path = result_path(Path(out), row["sample_id"])
    input_hash = canonical_hash(row["source"])
    if path.exists():
        cached = read(path)
        if cached.get("measurement_hash") != measurement_hash or cached.get("source_sha256") != input_hash:
            raise ValueError("Incompatible cached measurement: " + str(path))
        if cached["status"] == "ok":
            return row, cached, True
    from .measurement import measure_row
    record = measure_row(row, policy, str(ROOT), model)
    record.update(measurement_hash=measurement_hash, source_sha256=input_hash)
    write(path, record)
    return row, record, False


class Exports:
    """Per-row exports remain bounded in memory; summary is the completion receipt."""
    names = ("decisions", "retained_rows", "excluded_rows", "review_rows", "crop_proposals")

    def __init__(self, out, policy, evidence):
        self.out, self.policy, self.evidence = out, policy, evidence
        self.files, self.temps = {}, {}
        self.counts = {"retain": 0, "exclude": 0, "review": 0}
        self.errors = self.reused = self.total = 0
        for name in self.names:
            temp = out / (name + "." + uuid.uuid4().hex + ".tmp")
            self.temps[name] = temp
            self.files[name] = temp.open("w")

    def add(self, row, record, reused=False):
        decision = decide(record, self.policy, self.evidence)
        self.counts[decision["decision"]] += 1
        self.total += 1
        self.reused += int(reused)
        self.errors += int(record["status"] != "ok")
        self.files["decisions"].write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
        destination = {"retain": "retained_rows", "exclude": "excluded_rows", "review": "review_rows"}[decision["decision"]]
        value = row if decision["decision"] == "retain" else {"row": row, "decision": decision}
        self.files[destination].write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        if record["status"] == "ok":
            self.files["crop_proposals"].write(json.dumps({"sample_id": row["sample_id"], "proposal": record["crop"],
                "quality_decision": decision["decision"], "applied": False}, ensure_ascii=False, allow_nan=False) + "\n")

    def close(self, commit):
        for f in self.files.values():
            f.close()
        if commit:
            for name, temp in self.temps.items():
                os.replace(temp, self.out / (name + ".jsonl"))
        else:
            for temp in self.temps.values():
                temp.unlink(missing_ok=True)


def measure(args, policy, exports, fingerprint):
    checked = {}
    def tasks():
        for row in input_rows(args.input):
            check_source(row, checked)
            yield row, policy, str(args.model.resolve()), str(args.output), fingerprint
    iterator = iter(tasks())
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        while True:
            while len(pending) < args.workers * 2:
                try:
                    pending.add(pool.submit(work, next(iterator)))
                except StopIteration:
                    break
            if not pending:
                break
            finished, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                row, record, reused = future.result()
                exports.add(row, record, reused)
                if exports.total % 10 == 0:
                    write(args.output / "progress.json", {"processed": exports.total, "decisions": exports.counts, "reused": exports.reused})


def classify(args, policy, exports):
    value = read(args.measurements)
    records = value["results"] if isinstance(value, dict) else value
    index = {}
    for record in records:
        if record["id"] in index:
            raise ValueError("Duplicate input measurement")
        index[record["id"]] = record
    checked = {}
    for row in input_rows(args.input):
        check_source(row, checked)
        record = index.pop(row["sample_id"])
        if record["source"] != row["source"]:
            raise ValueError("Measurement/source row mismatch")
        if record["status"] == "ok":
            record["focus"] = analyze_focus(record["focus_samples"], row["source"]["duration_s"], policy)
            record["crop"] = propose_crop(record["raw_measurements"], row["source"], policy)
        write(result_path(args.output, record["id"]), record)
        exports.add(row, record, True)
    if index:
        raise ValueError("Measurements outside supplied input manifest")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure", "classify"))
    parser.add_argument("--input", type=Path, required=True, help="J annotation rows JSONL, rows list, or {rows:[...]}")
    parser.add_argument("--output", type=Path, required=True, help="Independent output directory; inputs are never rewritten")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/listen_visual_quality_v1.json")
    parser.add_argument("--evidence", type=Path, default=ROOT / "configs/listen_visual_quality_reviewed_evidence_20261030.json")
    parser.add_argument("--measurements", type=Path, help="Existing aggregate measurement JSON; required for classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.command == "classify" and args.measurements is None:
        parser.error("classify requires --measurements")
    policy = load_policy(args.policy)
    reviewed = evidence_index(read(args.evidence))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # A single writer owns a run; parallelism is inside its bounded worker pool.
    import fcntl
    with (args.output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        entry = ROOT / "scripts/run_listen_visual_quality.py"
        code[str(entry.relative_to(ROOT))] = file_hash(entry)
        contract = {"version": VERSION, "command": args.command, "input_sha256": file_hash(args.input),
                    "policy": policy, "evidence_sha256": file_hash(args.evidence), "code_sha256": code,
                    "backend": "cpu", "renders_review_media": False}
        if args.command == "measure":
            vendor = ROOT / "related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py"
            contract.update(model_sha256=file_hash(args.model), detector_sha256=file_hash(vendor),
                            package_versions={n: importlib.metadata.version(n) for n in ("av", "numpy", "mediapipe", "opencv-contrib-python")})
        else:
            contract["measurements_sha256"] = file_hash(args.measurements)
        fingerprint = canonical_hash(contract)
        contract_path = args.output / "contract.json"
        if contract_path.exists() and read(contract_path)["sha256"] != fingerprint:
            raise ValueError("Output belongs to another input/policy/runtime. Use a new output directory.")
        write(contract_path, {"sha256": fingerprint, "contract": contract})
        (args.output / "summary.json").unlink(missing_ok=True)
        started = time.monotonic()
        exports = Exports(args.output, policy, reviewed)
        complete = False
        try:
            if args.command == "measure":
                measure(args, policy, exports, fingerprint)
            else:
                classify(args, policy, exports)
            complete = True
        finally:
            exports.close(complete)
        summary = {"version": VERSION, "contract_sha256": fingerprint, "processed": exports.total,
                   "decisions": exports.counts, "measurement_errors": exports.errors,
                   "reused_measurements": exports.reused, "elapsed_s": time.monotonic() - started,
                   "source_inputs_unchanged": True, "crop_proposals_applied": False, "production_applied": False,
                   "outputs": {name: str(args.output / (name + ".jsonl")) for name in Exports.names}}
        write(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
