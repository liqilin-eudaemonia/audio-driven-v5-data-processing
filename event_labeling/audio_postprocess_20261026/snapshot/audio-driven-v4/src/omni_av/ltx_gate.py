import json
from dataclasses import replace

import torch

from .checkpoint import restore_checkpoint, save_checkpoint
from .config import PlanConfig
from .data import TextPlan, save_new
from .ltx_runtime import audit_target_window, encode_text, load_av_targets, load_renderer, render_loss, sample_latents, with_text
from .omni import load_planner
from .renderer import attach_plan
from .runner import contract, event, read_batch


def run_ltx_gate(args):
    if args.cache is None:
        raise ValueError("LTX gate requires a planner feature cache index")
    cfg = PlanConfig()
    index = json.loads(args.cache.read_text())
    if index["contract"] != contract(cfg):
        raise ValueError("Cache contract mismatch")
    data, payload = read_batch(index["paths"][0], cfg)
    plan = TextPlan(**payload["example"]["text_plan"])
    text = encode_text(cfg, plan)
    save_new(args.output / "text_conditions.pt", {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                                for k, v in text.items()})
    event("native_text_encoded", video=list(text["video"].shape), audio=list(text["audio"].shape))
    av, metadata = load_av_targets(args.data_root, payload["sample_id"], tiny=args.tiny)
    window = audit_target_window(payload, metadata)
    bundle, strategy, sampler = load_renderer(cfg)
    event("renderer_loaded", trainable=sum(p.numel() for p in bundle.parameters() if p.requires_grad))
    # Fresh planner is intentional: this is a gradient/initialization gate, not quality training.
    planner, _ = load_planner(cfg)
    bundle.planner = planner
    all_params = [p for p in bundle.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    frozen = [(p, p._version) for p in bundle.parameters() if not p.requires_grad]
    hidden_grads, losses = [], []
    for step in range(2):
        bundle.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            # Keep some targets visible; all-MASK planner tests run independently.
            mask = torch.zeros_like(data.valid)
            mask[:, ::2] = data.valid[:, ::2]
            planning = planner(data, mask)
            planning["hidden"].retain_grad()
            prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
            inputs = attach_plan(prepared, bundle.connector, planning["hidden"], planning["renderer_mask"])
            if step == 0:
                with torch.no_grad():
                    baseline = bundle.renderer(video=prepared.video, audio=prepared.audio, perturbations=None)
                    added = bundle.renderer(video=inputs.video, audio=inputs.audio, perturbations=None)
                if not all(torch.equal(a, b) for a, b in zip(baseline, added)):
                    raise RuntimeError("Zero-initialized plan branch changed native LTX output")
                del baseline, added
            av_loss, components = render_loss(bundle, strategy, inputs)
        # AV-only backward verifies the actual renderer -> Connector -> Omni LoRA path.
        av_loss.backward()
        grad = planning["hidden"].grad.float().norm().item()
        hidden_grads.append(grad)
        if step == 1:
            lora_grad = sum(p.grad.float().square().sum() for n, p in planner.named_parameters()
                            if "lora_b" in n and p.grad is not None).sqrt().item()
            if grad <= 0 or lora_grad <= 0:
                raise RuntimeError(f"No AV-to-Omni gradient: hidden={grad}, lora={lora_grad}")
        torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(p.grad is not None or p._version != version for p, version in frozen):
            raise RuntimeError("Frozen renderer/planner base changed")
        losses.append({k: float(v) for k, v in components.items()})
        event("av_step", step=step + 1, av_loss=float(av_loss), hidden_gradient=grad)
    run_contract = dict(config=contract(cfg), cache=str(args.cache.resolve()), tiny=args.tiny)
    save_checkpoint(args.output / "joint_step_000002.pt", bundle, optimizer, 2, run_contract)
    restore_checkpoint(args.output / "joint_step_000002.pt", bundle, optimizer, run_contract)
    bundle.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Sampler engineering gate uses matched GT-plan conditions, clearly recorded as Oracle.
        h = planner.hidden(data, torch.zeros_like(data.valid))
        prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
        inputs = attach_plan(prepared, bundle.connector, h, torch.cat([data.text_mask, data.valid], 1))
        sampled = sample_latents(bundle, inputs, steps=3)
    save_new(args.output / "sampled_latents.pt", {k: v.cpu() for k, v in sampled.items()})
    return dict(engineering_pass=True, quality_pass=None, sample_id=payload["sample_id"],
        zero_init_exact=True, frozen_audit_pass=True, hidden_gradients=hidden_grads,
        av_to_planning_lora_gradient=lora_grad, losses=losses, av_metadata=metadata, target_window=window,
        sampling="3-step Euler Oracle engineering gate; no quality conclusion",
        sampled_shapes={k: list(v.shape) for k, v in sampled.items()},
        max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
