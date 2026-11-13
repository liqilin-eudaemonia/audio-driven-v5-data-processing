"""Immutable experiment identity, independent stages, and explicit failed retries."""
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time

from listen_labeling.client import prompt
from listen_labeling.pipeline import load_manifest
from listen_labeling.schema import parse, validate_response
from p2_listen.contracts import atomic_json, digest, read, safe_id, sha
from . import VERSION
from .assets import inspect_weights
from .media import prepare


def validate_config(config):
    if config.get("schema") != VERSION:
        raise ValueError("Unsupported benchmark config version")
    names = []
    supported = {"joy": {"visual"}, "qwen3_vl": {"visual"}, "qwen2_audio": {"audio"},
                 "qwen3_omni": {"visual", "audio", "omni"}, "minicpm": {"visual", "audio", "omni"},
                 "captioner": {"audio"}, "asr": {"audio"}, "mediapipe": {"visual"}}
    for system in config["systems"]:
        names.append(safe_id(system["name"]))
        if system["branch"] not in supported.get(system["backend"], set()):
            raise ValueError("Unsupported backend/modality combination")
        auxiliary = system["backend"] in ("captioner", "asr", "mediapipe")
        if system["auxiliary"] != auxiliary:
            raise ValueError("Auxiliary tools must not masquerade as full event annotators")
        if system["representation"] not in ("separate", "native"):
            raise ValueError("Unknown input representation")
        if system["representation"] == "native" and system["branch"] != "omni":
            raise ValueError("Native audiovisual mode requires both modalities")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Expected unique nonempty system names")
    generation = config["generation"]
    for key in ("max_new_tokens", "context_limit", "max_generation_seconds"):
        if type(generation[key]) is not int or generation[key] <= 0:
            raise ValueError("Invalid generation budget")
    if type(generation["guided_json"]) is not bool:
        raise ValueError("guided_json must be boolean")
    return config


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/listen_benchmark").glob("*.py"))
    paths += [root / "src/listen_labeling" / name for name in
              ("client.py", "media.py", "pipeline.py", "schema.py", "guided.py")]
    paths += [root / "src/p2_listen" / name for name in ("contracts.py", "annotations.py")]
    paths += [root / "scripts/benchmark_listen_labeling.py"]
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def select_samples(manifest, limit):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    samples = load_manifest(manifest)
    return samples[:limit] if limit is not None else samples


def environment_identity():
    versions = {}
    for name in ("torch", "transformers", "numpy", "scipy", "av", "Pillow", "soundfile",
                 "lm-format-enforcer", "qwen-asr", "mediapipe", "minicpmo"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def run_system(config, system_name, samples, output, device, retry_failed=False, backend_factory=None):
    from .backends import load_backend
    validate_config(config)
    system = next(s for s in config["systems"] if s["name"] == system_name)
    root = Path(output).resolve() / system_name
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if backend_factory is None:
            asset = ({"sha256": sha(system["model_path"]), "ready": True} if system["backend"] == "mediapipe"
                     else inspect_weights(system["model_path"]))
            if not asset["ready"]:
                raise ValueError(f"Model assets not ready: {asset}")
        else:
            asset = {"test_backend": True}
        identity = {"schema": VERSION, "system": system, "media": config["media"],
                    "generation": config["generation"], "samples": samples,
                    "code": source_identity(), "weights": asset, "versions": environment_identity()}
        run_id = digest(identity)
        atomic_json(root / "inputs.json", dict(identity, run_id=run_id), immutable=True)
        backend, backend_error, results = None, None, []
        stages = ["auxiliary"] if system["auxiliary"] else ["discovery", "audit"]
        for row in samples:
            media, media_error = None, None
            sid = row["sample_id"]
            for stage in stages:
                folder = root / "requests" / sid / stage
                attempts = sorted(folder.glob("attempt-*.json"))
                previous = [read(p) for p in attempts]
                if previous and (previous[-1]["status"] != "failed" or not retry_failed):
                    result = previous[-1]
                else:
                    sequence = len(attempts) + 1
                    destination = folder / f"attempt-{sequence:04d}.json"
                    text = None if system["auxiliary"] else prompt(system["branch"], row["duration_s"], stage)
                    request = {"run_id": run_id, "sample_id": sid, "stage": stage, "prompt": text,
                               "system": system, "generation": config["generation"],
                               "media_hashes": row["media_hashes"], "video_transform": row.get("video_transform")}
                    atomic_json(folder / "request.json", request, immutable=True)
                    result = {"sample_id": sid, "system": system_name, "stage": stage,
                              "attempt": sequence, "status": "failed", "path": str(destination),
                              "utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                              "human_verified": False, "training_admitted": False}
                    began = time.monotonic()
                    try:
                        if media_error:
                            raise ValueError(media_error)
                        if media is None:
                            try:
                                media_cfg = dict(config["media"], **system.get("media_override", {}))
                                media = prepare(row, system["branch"], media_cfg)
                                if any(sha(row[k]) != value for k, value in row["media_hashes"].items()):
                                    raise ValueError("Source media changed after manifest validation")
                            except Exception as exc:
                                media_error = str(exc)
                                raise
                        result["input_evidence"] = media["meta"]
                        raw_path = folder / f"raw-{sequence:04d}.json"
                        if raw_path.exists():
                            # Recover a crash after raw persistence but before outcome persistence.
                            raw = read(raw_path)
                            result["recovered_raw_without_regeneration"] = True
                        else:
                            if backend is None:
                                if backend_error:
                                    raise RuntimeError("Backend initialization previously failed: " + backend_error)
                                try:
                                    backend = (backend_factory or load_backend)(system, device, config["generation"])
                                except Exception as exc:
                                    backend_error = str(exc)
                                    raise
                            inference_start = time.monotonic()
                            raw = backend.infer(media, text, stage, media_cfg["fps"])
                            result["inference_wall_seconds"] = time.monotonic() - inference_start
                        result["raw_response"] = raw
                        # Preserve the raw answer before any parsing or semantic validation.
                        atomic_json(raw_path, raw, immutable=True)
                        if system["auxiliary"]:
                            if system["backend"] == "captioner" and raw.get("finish_reason") != "stop":
                                raise ValueError("Captioner did not finish; raw output retained")
                            result["status"] = "auxiliary_only"
                        else:
                            if raw.get("finish_reason") != "stop":
                                raise ValueError("Incomplete/non-stop generation; raw output retained")
                            result["parsed"] = validate_response(parse(raw["text"]), system["branch"],
                                                                  row["duration_s"], stage)
                            result["status"] = "schema_pass"
                    except Exception as exc:
                        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    result["wall_seconds_including_preparation_and_load"] = time.monotonic() - began
                    atomic_json(destination, result, immutable=True)
                    previous.append(result)
                results.append(dict(result, first_attempt_status=previous[0]["status"], attempts=len(previous)))
                report = {"schema": VERSION, "run_id": run_id, "system": system_name,
                          "branch": system["branch"], "auxiliary": system["auxiliary"],
                          "expected_requests": len(samples) * len(stages), "completed": len(results),
                          "complete": len(results) == len(samples) * len(stages), "results": results,
                          "human_ground_truth": False, "training_admitted": False}
                atomic_json(root / "summary.json", report)
                print(json.dumps({"system": system_name, "sample_id": sid, "stage": stage,
                                  "status": result["status"], "completed": len(results)}, ensure_ascii=False), flush=True)
        return report


def review_template(samples):
    """Blind template deliberately contains no model answers or implied negatives."""
    return [{"sample_id": row["sample_id"], "duration_s": row["duration_s"],
             "media_hashes": row["media_hashes"], "target_video": row["target_video"],
             "target_audio": row["target_audio"], "video_transform": row.get("video_transform"),
             "source_reference": row.get("source_reference"),
             "reviewer": "", "reviewed_at": "", "human_verified": False,
             "evaluation_partition": "unassigned", "stratum": "unassigned",
             "label": {"reviewed": {"head_nod": [], "listener_vocalization": []},
                       "unknown": {"head_nod": [], "listener_vocalization": []}, "events": [], "notes": ""}}
            for row in samples]
