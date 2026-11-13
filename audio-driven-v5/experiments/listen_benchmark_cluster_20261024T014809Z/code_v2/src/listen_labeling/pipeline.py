"""Resumable branches, review packets, and deliberate P2 label export."""
import copy
from datetime import datetime
import fcntl
from pathlib import Path
from p2_listen.contracts import atomic_json, digest, read, rows, sha
from p2_listen.annotations import validate
from . import VERSION
from .client import call, prompt, validate_endpoint
from .media import frames, prepare_row, waveform
from .schema import fuse, review_reasons, validate_response


def code_hash():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / "p2_listen" / n for n in ("annotations.py", "contracts.py")]
    return digest({str(p.relative_to(Path(__file__).parents[1])): sha(p) for p in sorted(paths)})


def branches_for_mode(mode):
    return {"split": ["visual", "audio"], "omni": ["omni"], "visual": ["visual"]}[mode]


def load_manifest(path, mode="split"):
    path = Path(path).resolve()
    return [prepare_row(r, path.parent, mode) for r in rows(path)]


def branch_inputs(row, branch, media_cfg):
    content, meta = [], {}
    if branch in ("visual", "omni"):
        part, info = frames(row["target_video"], row["duration_s"], media_cfg, row.get("video_transform"))
        content.extend(part); meta["video"] = info
    if branch in ("audio", "omni"):
        part, info = waveform(row["target_audio"], row["duration_s"], media_cfg)
        content.extend(part); meta["audio"] = info
    return content, meta


def preflight(manifest, cfg, mode):
    samples = load_manifest(manifest, mode)
    branches = branches_for_mode(mode)
    for branch in branches:
        validate_endpoint(cfg["endpoints"][branch], branch)
    checked = []
    for row in samples:
        # Never send mixed audio to a listener-only audio annotator.
        _, meta = branch_inputs(row, "visual" if mode == "visual" else "omni", cfg["media"])
        checked.append(dict(sample_id=row["sample_id"], media_hashes=row["media_hashes"], media=meta))
    return dict(schema=VERSION, network_called=False, samples=checked)


def run(manifest, cfg, output, mode="split"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another annotation writer is using this output directory") from None
        return _run(manifest, cfg, output, mode)


def _run(manifest, cfg, output, mode):
    samples = load_manifest(manifest, mode)
    branches = branches_for_mode(mode)
    for branch in branches:
        validate_endpoint(cfg["endpoints"][branch], branch)
    code = code_hash()
    run_id = digest(dict(samples=samples, config=cfg, mode=mode, code=code))
    root = output / "runs" / run_id
    atomic_json(root / "inputs.json", dict(samples=samples, config=cfg, mode=mode, code=code), immutable=True)
    report = dict(schema=VERSION, run_id=run_id, mode=mode, items=[], failed=0, succeeded=0)
    for row in samples:
        sid = row["sample_id"]
        result, errors, stage_paths, discovery_paths = {}, {}, {}, {}
        # Two distinct requests over the original media. Audit never sees discovery text.
        for stage in ("discovery", "audit"):
            for branch in branches:
                if stage == "audit" and "discovery" not in result.get(branch, {}):
                    continue
                hashes = {k: v for k, v in row["media_hashes"].items()
                          if branch == "omni" or k == ("target_video" if branch == "visual" else "target_audio")}
                identity = dict(code=code, stage=stage, branch=branch, sample_id=sid, duration_s=row["duration_s"],
                                media_hashes=hashes, media_config=cfg["media"], endpoint=cfg["endpoints"][branch],
                                prompt=prompt(branch, row["duration_s"], stage))
                if branch in ("visual", "omni"):
                    identity["video_transform"] = row.get("video_transform")
                fingerprint = digest(identity)
                folder = output / "branches" / sid / branch / stage / fingerprint
                try:
                    if (folder / "result.json").exists():
                        value = read(folder / "result.json")
                        if value["fingerprint"] != fingerprint or value.get("stage") != stage:
                            raise ValueError("Cache fingerprint/stage mismatch")
                        validate_response(value["parsed"], branch, row["duration_s"], stage)
                    else:
                        content, meta = branch_inputs(row, branch, cfg["media"])
                        if any(sha(row[k]) != v for k, v in hashes.items()):
                            raise ValueError("Media changed during preparation")
                        value = call(cfg["endpoints"][branch], branch, row["duration_s"], content,
                                     folder, fingerprint, meta, stage=stage)
                        atomic_json(folder / "result.json", value, immutable=True)
                    result.setdefault(branch, {})[stage] = value
                    stage_paths.setdefault(branch, {})[stage] = str(folder / "result.json")
                    if stage == "discovery":
                        discovery_path = root / "open_events" / sid / (branch + ".json")
                        atomic_json(discovery_path, value["parsed"], immutable=True)
                        discovery_paths[branch] = str(discovery_path)
                except Exception as exc:
                    # Retain all completed stages, including discovery when audit fails.
                    errors[branch + "." + stage] = {"type": type(exc).__name__, "message": str(exc)}
        item = dict(sample_id=sid, experiment_split=row["experiment_split"], media_hashes=row["media_hashes"],
                    target_video=row["target_video"], target_audio=row["target_audio"], errors=errors,
                    training_admitted=row.get("training_admitted", True), video_transform=row.get("video_transform"),
                    stage_results=stage_paths, open_events=discovery_paths)
        if errors:
            item["status"] = "failed"
            report["failed"] += 1
        else:
            label = fuse(row, result, mode)
            label_path = root / "drafts" / (sid + ".json")
            atomic_json(label_path, label, immutable=True)
            # Rich events are retained independently, not discarded by the two-class mapping.
            atomic_json(root / "rich" / (sid + ".json"), {b: {s: v["parsed"] for s, v in r.items()} for b, r in result.items()}, immutable=True)
            item.update(status="draft", draft=str(label_path), draft_sha256=digest(label),
                        review_reasons=review_reasons(label, result))
            report["succeeded"] += 1
        report["items"].append(item)
        atomic_json(root / "summary.json", report)
    reviews = [dict(sample_id=x["sample_id"], draft_sha256=x["draft_sha256"], human_verified=False,
                    reviewer="", reviewed_at="", open_events=x["open_events"], label=read(x["draft"])) for x in report["items"] if x["status"] == "draft"]
    # Never overwrite a person's edited review file on resume.
    template = root / ("review_template-" + digest(reviews)[:16] + ".json")
    if not template.exists():
        atomic_json(template, reviews, immutable=True)
    report["review_template"] = str(template)
    atomic_json(root / "summary.json", report)
    atomic_json(root / "review_queue.json", report["items"])
    atomic_json(output / "latest.json", dict(summary=str(root / "summary.json"), run_id=run_id))
    return dict(summary=str(root / "summary.json"), succeeded=report["succeeded"], failed=report["failed"])


def export(summary_path, output, reviews_path=None, allow_weak_train=False):
    summary = read(summary_path)
    items = {x["sample_id"]: x for x in summary["items"] if x["status"] == "draft"}
    reviews = {x["sample_id"]: x for x in rows(reviews_path)} if reviews_path else {}
    if set(reviews) - set(items):
        raise ValueError("Review references samples outside this run")
    labels, skipped = [], []
    for sid, item in items.items():
        if item.get("training_admitted") is False or item["experiment_split"] == "unassigned":
            skipped.append(sid)
            continue
        label = read(item["draft"])
        if digest(label) != item["draft_sha256"]:
            raise ValueError("Draft modified; edit a review copy instead")
        for key, expected in item["media_hashes"].items():
            if sha(item[key]) != expected:
                raise ValueError("Media changed since annotation")
        record = reviews.get(sid)
        if record and record.get("human_verified") is True:
            if (record.get("draft_sha256") != item["draft_sha256"] or
                not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip() or
                not isinstance(record.get("reviewed_at"), str)):
                raise ValueError("Review must bind draft and name reviewer/date")
            date = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
            if date.tzinfo is None:
                raise ValueError("Review date requires timezone")
            corrected = copy.deepcopy(record["label"])
            validate(corrected, {"sample_id": sid, "requested_duration_s": label["duration_s"]})
            if corrected["provenance"] != label["provenance"]:
                raise ValueError("Edit events/coverage/unknown in review copy, not provenance")
            corrected["provenance"].update(human_verified=True, reviewer=record["reviewer"],
                                            reviewed_at=record["reviewed_at"], review_sha256=digest(record),
                                            source_draft_sha256=item["draft_sha256"])
            for event in corrected["events"]:
                event["verification"] = "human"
            labels.append(corrected)
        elif allow_weak_train and item["experiment_split"] == "train":
            labels.append(label)
        else:
            skipped.append(sid)
    if not labels:
        raise ValueError("No eligible labels: supply completed reviews or explicitly export weak train labels")
    atomic_json(output, labels, immutable=True)
    return dict(exported=len(labels), skipped=skipped, failed_samples=summary["failed"],
                output=str(Path(output).resolve()), human_verified=sum(x["provenance"]["human_verified"] for x in labels))
