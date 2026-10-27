"""Low-rank Stage-2 adapters over learned Stage-1 LTX injection weights."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def named_optimizer_group_grad_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Return pre-clipping L2 grad norms for named replicated parameter groups."""
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if not name:
            continue
        parameter_norms = [
            torch.linalg.vector_norm(parameter.grad.detach().float())
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if parameter_norms:
            total_norm = torch.linalg.vector_norm(torch.stack(parameter_norms))
            result[str(name)] = total_norm.item()
    return result


class Stage2LoRALinear(nn.Module):
    """Freeze a learned Linear as the base and train a zero-init low-rank delta."""

    def __init__(
        self, base_layer: nn.Linear, *, rank: int, alpha: int, dropout: float
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.stage2_lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.stage2_lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.stage2_lora_A.to(
            device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        self.stage2_lora_B.to(
            device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        nn.init.kaiming_uniform_(self.stage2_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.stage2_lora_B.weight)
        self.base_layer.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        delta = self.stage2_lora_B(self.stage2_lora_A(self.dropout(inputs)))
        return base + delta * self.scaling


def _replace_child(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _belongs_to(path: str, names: Iterable[str]) -> bool:
    return any(
        path == name or path.startswith(f"{name}.") or f".{name}." in path
        for name in names
    )


def apply_stage2_injection_lora(
    model: nn.Module,
    *,
    injection_modules: list[str],
    adapter_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    train_adapter_norm_and_logits: bool,
    full_trainable_modules: list[str],
) -> list[str]:
    """Freeze learned injection bases and add LoRA to attention projections/adapter proj."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    persistent = list(dict.fromkeys([*injection_modules, *adapter_modules]))
    matched_persistent: set[str] = set()
    for module_path, module in base.named_modules():
        for name in persistent:
            if module_path == name or module_path.endswith(f".{name}"):
                module.requires_grad_(False)
                matched_persistent.add(name)

    missing = sorted(set(persistent) - matched_persistent)
    if missing:
        raise ValueError(f"Stage-2 persistent modules were not found: {missing}")

    attention_leaves = {"to_q", "to_k", "to_v"}
    targets: list[str] = []
    for path, module in list(base.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, Stage2LoRALinear):
            continue
        leaf = path.rsplit(".", 1)[-1]
        is_attention_target = _belongs_to(path, injection_modules) and (
            leaf in attention_leaves or path.endswith(".to_out.0")
        )
        is_adapter_target = _belongs_to(path, adapter_modules) and leaf == "proj"
        if not (is_attention_target or is_adapter_target):
            continue
        _replace_child(
            base,
            path,
            Stage2LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        targets.append(path)

    if not targets:
        raise RuntimeError("Stage-2 LoRA did not match any injection Linear modules")

    if train_adapter_norm_and_logits:
        for path, parameter in base.named_parameters():
            if _belongs_to(path, adapter_modules) and (
                ".norm." in path
                or path.endswith(".layer_logits")
                or path.endswith(".base_layer.bias")
            ):
                parameter.requires_grad_(True)

    matched_full: set[str] = set()
    for module_path, module in base.named_modules():
        for name in full_trainable_modules:
            if module_path == name or module_path.endswith(f".{name}"):
                module.requires_grad_(True)
                matched_full.add(name)
    missing_full = sorted(set(full_trainable_modules) - matched_full)
    if missing_full:
        raise ValueError(
            f"Stage-2 full-trainable modules were not found: {missing_full}"
        )
    return sorted(targets)


def get_qwen_base_model(model: nn.Module) -> nn.Module:
    current = model.module if hasattr(model, "module") else model
    return current.get_base_model() if hasattr(current, "get_base_model") else current


def apply_qwen_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    last_n_layers: int,
    target_leaf_names: list[str],
) -> tuple[nn.Module, list[str]]:
    """Freeze Qwen and apply PEFT LoRA to exact language-layer Linear paths."""
    from peft import LoraConfig, get_peft_model

    base = get_qwen_base_model(model)
    layers = base.model.language_model.layers
    if last_n_layers < 1 or last_n_layers > len(layers):
        raise ValueError(
            f"qwen_last_n_layers must be in [1, {len(layers)}], got {last_n_layers}"
        )
    allowed = set(target_leaf_names)
    selected = {
        child
        for layer in layers[-last_n_layers:]
        for name, child in layer.named_modules()
        if isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in allowed
    }
    targets = sorted(
        name
        for name, child in base.named_modules()
        if child in selected and name.rsplit(".", 1)[-1] in allowed
    )
    if not targets:
        raise RuntimeError(
            "No Qwen language Linear modules matched Stage-2 LoRA targets"
        )
    model.requires_grad_(False)
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            init_lora_weights=True,
        ),
        autocast_adapter_dtype=False,
    )
    base = get_qwen_base_model(wrapped)
    base.lm_head.requires_grad_(False)
    base.model.visual.requires_grad_(False)
    base.model.visual.eval()
    return wrapped, targets


def save_qwen_lora(model: nn.Module, path: str, *, metadata: dict[str, str]) -> None:
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    state = {
        key: value.detach().cpu().contiguous()
        for key, value in get_peft_model_state_dict(model).items()
    }
    if not state or not any("lora_A" in key for key in state):
        raise RuntimeError("Qwen Stage-2 checkpoint contains no LoRA tensors")
    save_file(state, path, metadata=metadata)


def load_qwen_lora(model: nn.Module, path: str) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    incompatible = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected:
        raise ValueError(f"Unexpected Qwen Stage-2 LoRA keys: {unexpected[:8]}")
