"""P2.1 archives always pair Connector with text-attention LoRA."""
import json
from pathlib import Path
import re

import torch

from omni_av_p2.checkpoint import atomic_save, capture_rng, restore_rng
from . import VERSION


def architecture(model):
    return dict(model=model.cfg.to_dict(), lora=model.lora.to_dict(), targets=model.targets)


def validate_state(module, state):
    expected = module.state_dict()
    if expected.keys() != state.keys():
        raise ValueError("P2.1 checkpoint parameter keys mismatch")
    if any(expected[n].shape != t.shape or not torch.isfinite(t).all() for n, t in state.items()):
        raise ValueError("P2.1 checkpoint invalid tensor")


def require_complete(path):
    path = Path(path)
    match = re.fullmatch(r'(extractor_step_(\d+))(?:_deployment)?\.pt', path.name)
    if match:
        marker = path.with_name(match[1] + '.complete.json')
        if not marker.is_file() or json.loads(marker.read_text())['step'] != int(match[2]):
            raise ValueError("P2.1 checkpoint not completely published")


def save(path, model, optimizer, step, identity, rank_rng=None):
    model.assert_parameter_ownership()
    state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    validate_state(model, state)
    atomic_save(path, dict(version=VERSION, identity=identity, architecture=architecture(model), step=step,
        model=state, optimizer=optimizer.state_dict(), rng=capture_rng(), rank_rng=rank_rng))


def load(path, model, identity=None, optimizer=None, rank=None):
    require_complete(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("version") != VERSION or data.get("architecture") != architecture(model)
            or (identity is not None and data["identity"] != identity)):
        raise ValueError("P2.1 checkpoint identity/architecture mismatch")
    if data.get("deployment_only"):
        raise ValueError("Deployment bundle cannot resume training")
    validate_state(model, data["model"])
    model.load_state_dict(data["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
        states = data.get("rank_rng")
        if rank is not None and states is not None:
            restore_rng(states[rank])
        else:
            restore_rng(data["rng"])
    return data


def export_deployment(source, target):
    data = torch.load(source, map_location="cpu", weights_only=True)
    if data.get("version") != VERSION:
        raise ValueError("Not a Pipeline 2.1 checkpoint")
    state = {k: v for k, v in data["model"].items()
             if k.startswith(("extractor.connector.", "adapters."))}
    if (not any(k.startswith("adapters.") for k in state)
            or not any(k.startswith("extractor.connector.") for k in state)
            or any(not torch.isfinite(t).all() for t in state.values())):
        raise ValueError("Deployment requires finite Connector AND text LoRA tensors")
    atomic_save(target, dict(version=VERSION, identity=data["identity"], architecture=data["architecture"],
        step=data["step"], model=state, deployment_only=True))


def initialize_from_p2(path, model, expected_contract, recipe):
    """Explicit warm-start of all Extractor weights; zero LoRA and fresh optimizer."""
    from omni_av_p2 import VERSION as P2_VERSION
    require_complete(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("version") != P2_VERSION or data.get("deployment_only")
            or data["identity"]["contract"] != expected_contract
            or data["identity"]["recipe"] != recipe):
        raise ValueError("P2 warm-start requires a full matching no-time/recipe checkpoint")
    validate_state(model.extractor, data["model"])
    model.extractor.load_state_dict(data["model"], strict=True)
    return data["step"]
