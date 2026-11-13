"""Read-only local asset checks. Never downloads or loads model tensors."""
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess

from p2_listen.contracts import digest, sha


def inspect_weights(directory):
    root = Path(directory)
    result = {"path": str(root), "ready": False, "errors": [], "files": []}
    if not (root / "config.json").is_file():
        result["errors"].append("Missing config.json (an empty directory is not a model)")
        return result
    try:
        result["model_type"] = json.loads((root / "config.json").read_text()).get("model_type")
        indices = list(root.glob("*.safetensors.index.json"))
        expected = {}
        if indices:
            for index in indices:
                for key, name in json.loads(index.read_text())["weight_map"].items():
                    expected.setdefault(name, set()).add(key)
            paths = [root / name for name in sorted(expected)]
        else:
            paths = sorted(root.glob("*.safetensors"))
        if not paths:
            raise ValueError("No safetensors weights")
        for path in paths:
            if path.parent.resolve() != root.resolve():
                raise ValueError("Weight index must reference files in the model directory")
            stat = path.stat()
            with path.open("rb") as stream:
                size_bytes = stream.read(8)
                if len(size_bytes) != 8:
                    raise ValueError(f"Truncated header: {path.name}")
                header_size = struct.unpack("<Q", size_bytes)[0]
                if not 0 < header_size <= min(stat.st_size - 8, 100_000_000):
                    raise ValueError(f"Invalid header size: {path.name}")
                header = json.loads(stream.read(header_size))
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            if not tensors or not expected.get(path.name, set()) <= tensors.keys():
                raise ValueError(f"Index/header tensor mismatch: {path.name}")
            ends = []
            for tensor in tensors.values():
                start, end = tensor["data_offsets"]
                if not 0 <= start <= end <= stat.st_size - 8 - header_size:
                    raise ValueError(f"Truncated tensor data: {path.name}")
                ends.append(end)
            if max(ends) != stat.st_size - 8 - header_size:
                raise ValueError(f"Unexpected weight file length: {path.name}")
            result["files"].append({"name": path.name, "bytes": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns, "header_sha256": digest(header)})
        metadata = sorted(set(root.glob("*.json")) | set(root.glob("*.py")))
        result["metadata_sha256"] = {p.name: sha(p) for p in metadata}
        result["bytes"] = sum(f["bytes"] for f in result["files"])
        result["ready"] = True
        result["verification"] = "index/header/length; tensor payload and upstream revision not verified"
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        result["errors"].append(str(exc))
    result["identity"] = digest(result)
    return result


def probe_python(python, modules, dependency_path=None):
    code = """import importlib.util,importlib.metadata,json,sys
names=json.loads(sys.argv[1]); result={}
if sys.argv[2]: sys.path.insert(0,sys.argv[2])
for name in names:
    try: result[name]=bool(importlib.util.find_spec(name))
    except (ImportError,ValueError): result[name]=False
versions={}
for name in ('torch','transformers','numpy','av','soundfile','scipy','mediapipe','qwen-asr'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
print(json.dumps(dict(modules=result,versions=versions,python=sys.version)))
"""
    try:
        process = subprocess.run([python, "-c", code, json.dumps(modules), dependency_path or ""],
                                 text=True, capture_output=True, timeout=30)
        if process.returncode:
            return {"ready": False, "error": process.stderr[-2000:]}
        result = json.loads(process.stdout)
        result["ready"] = all(result["modules"].values())
        result["verification"] = "module discovery only; binary imports checked separately"
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"ready": False, "error": str(exc)}


def preflight(config):
    weights, environments, systems = {}, {}, []
    for system in config["systems"]:
        path, python = system["model_path"], system["python"]
        if path not in weights:
            if system["backend"] == "mediapipe":
                file = Path(path)
                weights[path] = {"path": path, "ready": file.is_file() and file.stat().st_size > 0,
                                 "sha256": sha(file) if file.is_file() else None}
            else:
                weights[path] = inspect_weights(path)
        modules = ["numpy", "PIL", "av", "soundfile", "scipy", "requests"]
        modules += ["mediapipe"] if system["backend"] == "mediapipe" else ["torch", "transformers"]
        if system["backend"] == "asr":
            modules += ["qwen_asr"]
        if system["backend"] == "minicpm" and system["representation"] == "native":
            modules += ["minicpmo"]
        if config["generation"]["guided_json"] and not system["auxiliary"]:
            modules += ["lmformatenforcer"]
        key = digest([python, modules])
        if key not in environments:
            environments[key] = dict(path=python, **probe_python(python, modules, system.get("dependency_path", config.get("dependency_path"))))
        systems.append({"name": system["name"], "weight_path": path, "environment": key,
                        "ready": weights[path]["ready"] and environments[key]["ready"]})
    return {"weights": weights, "environments": environments, "systems": systems,
            "all_ready": all(s["ready"] for s in systems), "models_loaded": False,
            "network_called": False}
