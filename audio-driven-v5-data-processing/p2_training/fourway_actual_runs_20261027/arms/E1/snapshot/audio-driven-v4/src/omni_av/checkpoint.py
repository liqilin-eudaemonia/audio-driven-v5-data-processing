"""New-file-only delta checkpoints with optimizer and RNG restoration."""
import random
from pathlib import Path

import numpy as np
import torch

from . import VERSION
from .data import save_new


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path, model, optimizer, step, contract):
    save_new(path, dict(version=VERSION, contract=contract, step=step,
        trainable=trainable_state(model), optimizer=optimizer.state_dict(),
        rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        rng_python=random.getstate(), rng_numpy=(np.random.get_state()[0],
            torch.from_numpy(np.random.get_state()[1].astype(np.int64)), *np.random.get_state()[2:])))


def restore_checkpoint(path, model, optimizer, contract):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch")
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if params.keys() != saved["trainable"].keys():
        raise ValueError("Trainable parameter keys mismatch")
    for name, param in params.items():
        value = saved["trainable"][name]
        if value.shape != param.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor {name}")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved["trainable"][name])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng_torch"])
    if saved["rng_cuda"]:
        torch.cuda.set_rng_state_all(saved["rng_cuda"])
    random.setstate(saved["rng_python"])
    state = saved["rng_numpy"]
    np.random.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    return saved["step"]


def initialize_delta(path, model, expected_model_contract, prefix="", expected_names=None):
    """Initialize a stage from verified weights; do not restore optimizer/RNG."""
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    identity = saved["contract"].get("model", saved["contract"].get("config"))
    if saved["version"] != VERSION or identity != expected_model_contract:
        raise ValueError("Initialization model contract mismatch")
    params = dict(model.named_parameters())
    state = {k.removeprefix(prefix): v for k, v in saved["trainable"].items()
             if not prefix or k.startswith(prefix)}
    if not state:
        raise ValueError("No matching initialization parameters")
    expected = (set(expected_names) if expected_names is not None else
                {name for name, p in params.items() if p.requires_grad})
    if set(state) != expected:
        raise ValueError("Initialization parameter set is incomplete or belongs to another stage")
    for name, value in state.items():
        if name not in params or value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid initialization tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    return list(state)
