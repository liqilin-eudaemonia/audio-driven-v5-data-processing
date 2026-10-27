import random
import os
import json
from pathlib import Path

import numpy as np
import torch

from omni_av.data import save_new
from . import VERSION


def capture_rng():
    state = np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        python_rng=random.getstate(), numpy_rng=[state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]])


def restore_rng(data):
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] is not None:
        torch.cuda.set_rng_state(data["cuda_rng"])
    random.setstate(data["python_rng"])
    state = data["numpy_rng"]
    np.random.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        # Flush through a Python file object before publishing on shared storage.
        # A successful pathname-based torch.save alone is not a durability barrier.
        with temp.open('xb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # Publish only a complete file; refuse to overwrite.
    finally:
        temp.unlink(missing_ok=True)


def atomic_json_new(path, value):
    """Publish a complete JSON file without replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.partial.{os.getpid()}')
    try:
        with temp.open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save(path, model, optimizer, step, identity, rank_rng=None):
    state = np.random.get_state()
    atomic_save(path, dict(version=VERSION, identity=identity, step=step, rank_rng=rank_rng,
        model={n: t.detach().cpu().clone() for n, t in model.state_dict().items()},
        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() and rank_rng is None else [],
        python_rng=random.getstate(), numpy_rng=[state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]]))


def load(path, model, identity, optimizer=None, connector_only=False, rank=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["version"] != VERSION or (identity is not None and data["identity"] != identity):
        raise ValueError("P2 checkpoint identity mismatch")
    state = data["model"]
    if connector_only:
        state = {n.removeprefix("connector."): t for n, t in state.items() if n.startswith("connector.")}
    expected = model.state_dict()
    if expected.keys() != state.keys():
        raise ValueError("P2 checkpoint parameter keys mismatch")
    if any(expected[n].shape != t.shape or not torch.isfinite(t).all() for n, t in state.items()):
        raise ValueError("P2 checkpoint invalid tensor")
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["torch_rng"])
        if data["cuda_rng"]:
            torch.cuda.set_rng_state_all(data["cuda_rng"])
        random.setstate(data["python_rng"])
        state = data["numpy_rng"]
        np.random.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
        if rank is not None and data.get("rank_rng") is not None:
            restore_rng(data["rank_rng"][rank])
    return data


def export_connector(source, target):
    """Deployment file contains no decoder tensors or optimizer state."""
    data = torch.load(source, map_location="cpu", weights_only=True)
    if data["version"] != VERSION:
        raise ValueError("Not a Pipeline 2 checkpoint")
    state = {n: t for n, t in data["model"].items() if n.startswith("connector.")}
    if not state or any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Invalid connector state")
    atomic_save(target, dict(version=VERSION, identity=data["identity"], step=data["step"],
                          model=state, deployment_only=True))
