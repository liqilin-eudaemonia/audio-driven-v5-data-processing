"""Atomic, independently fingerprinted cache parts."""
import os
from pathlib import Path
import tempfile
import torch
from .contracts import atomic_json, digest, read, sha


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    return value


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    return value


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_part(folder, identity, device="cpu", missing_ok=False):
    folder = Path(folder)
    marker = folder / "complete.json"
    if not marker.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"Cache incomplete: {folder}")
    m = read(marker)
    if m["identity"] != digest(identity):
        raise ValueError(f"Cache identity differs: {folder}; use a new experiment/cache version")
    if sha(folder / "payload.pt") != m["payload_sha256"]:
        raise ValueError("Corrupt cache payload")
    return torch.load(folder / "payload.pt", map_location=device, weights_only=True)


def save_part(folder, identity, payload):
    folder = Path(folder)
    if load_part(folder, identity, missing_ok=True) is not None:
        return
    atomic_torch(folder / "payload.pt", cpu(payload))
    atomic_json(folder / "complete.json", dict(identity=digest(identity), contract=identity,
                payload_sha256=sha(folder / "payload.pt")))
