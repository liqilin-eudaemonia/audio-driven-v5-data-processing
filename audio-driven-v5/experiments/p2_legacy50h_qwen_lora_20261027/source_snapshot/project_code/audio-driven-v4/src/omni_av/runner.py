"""Local preparation, planner training and engineering gates. No remote launches."""
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import VERSION
from .checkpoint import initialize_delta, restore_checkpoint, save_checkpoint, trainable_state
from .config import PlanConfig
from .data import (Observation, TextPlan, file_hash, gt_text_plan, save_new, select_training_examples, write_json_new)
from .omni import (encode_prefix, extract_targets, generate_reply, load_planner, make_layout, prepare_observation)
from .planner import PlanningBatch


def event(name, **kwargs):
    print(json.dumps(dict(event=name, **kwargs), ensure_ascii=False), flush=True)


def contract(config):
    root = Path(config.omni_path)
    return dict(version=VERSION, config=config.to_dict(),
                model_config_sha256=file_hash(root / "config.json"),
                model_index_sha256=file_hash(root / "model.safetensors.index.json"))


def create_cache(args, planner, processor):
    examples = (json.loads(args.manifest.read_text()) if args.manifest
                else select_training_examples(args.data_root, args.samples, args.budget))
    if not examples or any(x["split"] != "train" for x in examples):
        raise ValueError("Preparation requires a nonempty explicit Train set")
    ids = [x["observation"]["sample_id"] for x in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    write_json_new(args.output / "samples.json", examples)
    entries = []
    for i, example in enumerate(examples):
        obs = Observation(**example["observation"])
        text = TextPlan(**example["text_plan"])
        gt = json.loads(Path(example["gt_path"]).read_text())
        if text != gt_text_plan(gt, obs.sample_id):
            raise ValueError("Training text does not match this sample's structured GT")
        if not (0 < example["duration_s"] <= obs.output_duration_s):
            raise ValueError("Training target must fit the external output budget")
        prepared = prepare_observation(processor, obs, planner.config)
        prefix = encode_prefix(planner, processor, prepared, text)
        grid = prepared["image_grid_thw"][0]
        merge = processor.image_processor.merge_size
        layout = make_layout(prefix, obs.output_duration_s,
                             (int(grid[1]) // merge, int(grid[2]) // merge), planner.config)
        batch = extract_targets(planner, processor, layout, example["target_video"],
                                example["target_audio"], example["duration_s"], obs.output_duration_s)
        path = args.output / f"features_{i:04d}.pt"
        payload = dict(contract=contract(planner.config), sample_id=obs.sample_id, split=example["split"],
                       tensors={k: v.cpu() for k, v in vars(batch).items()}, example=example,
                       gt_plan_sha256=file_hash(example["gt_path"]))
        save_new(path, payload)
        entries.append(str(path.resolve()))
        event("features_saved", sample_id=obs.sample_id, prefix=list(batch.prefix.shape),
              target=list(batch.targets.shape), valid=int(batch.valid.sum()))
    write_json_new(args.output / "cache_index.json", dict(contract=contract(planner.config), paths=entries))
    return entries, examples


def read_batch(path, config):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload["contract"] != contract(config):
        raise ValueError("Feature-cache model or preprocessing mismatch")
    batch = PlanningBatch(**payload["tensors"])
    batch.validate()
    if payload["split"] != "train":
        raise ValueError("Training runner refuses a non-Train cache")
    return batch.to("cuda:0"), payload


def run_planner(args):
    cfg = PlanConfig()
    planner, processor = load_planner(cfg)
    event("omni_loaded", trainable=sum(p.numel() for p in planner.parameters() if p.requires_grad))
    if args.cache is None:
        paths, examples = create_cache(args, planner, processor)
    else:
        saved = json.loads(args.cache.read_text())
        if saved["contract"] != contract(cfg):
            raise ValueError("Cache index contract mismatch")
        paths = saved["paths"]
        examples = None
    if not paths:
        raise ValueError("Empty training cache")
    if args.command == "prepare":
        return dict(engineering_pass=True, quality_pass=None, prepared=len(paths))
    if args.steps < 1:
        raise ValueError("Training steps must be positive")
    data, payload = read_batch(paths[0], cfg)
    obs = Observation(**payload["example"]["observation"])
    prepared = prepare_observation(processor, obs, cfg)
    native_before = None
    if args.command == "omni-gate":
        native_before, raw = generate_reply(planner, processor, prepared)
        write_json_new(args.output / "native_before.json", dict(raw=raw, plan=asdict(native_before)))
        event("native_reply_generated", reply=native_before.reply)
    optimizer = torch.optim.AdamW([p for p in planner.parameters() if p.requires_grad], lr=1e-4)
    run_contract = dict(model=contract(cfg), cache_paths=paths, lr=1e-4)
    start_step = restore_checkpoint(args.resume, planner, optimizer, run_contract) if args.resume else 0
    frozen = [(n, p, p._version) for n, p in planner.named_parameters() if not p.requires_grad]
    metrics = []
    with (args.output / "train.jsonl").open("x") as log:
        for step in range(start_step + 1, start_step + args.steps + 1):
            data, _ = read_batch(paths[(step - 1) % len(paths)], cfg)
            planner.train()
            optimizer.zero_grad(set_to_none=True)
            before = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                result = planner(data)
            result["loss"].backward()
            grads = {}
            for key in ("lora_b", "video_flow", "audio_flow", "mask_token"):
                selected = [p.grad.float().square().sum() for n, p in planner.named_parameters()
                            if key in n and p.grad is not None]
                grads[key] = torch.stack(selected).sum().sqrt().item() if selected else 0
            if not all(x > 0 and torch.isfinite(torch.tensor(x)) for x in grads.values()):
                raise RuntimeError(f"Missing/nonfinite training gradients: {grads}")
            norm = torch.nn.utils.clip_grad_norm_([p for p in planner.parameters() if p.requires_grad],
                                                1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(p.grad is not None or p._version != v for _, p, v in frozen):
                raise RuntimeError("Frozen base changed during planner training")
            item = dict(step=step, loss=result["loss"].item(),
                        losses={k: v.item() for k, v in result["losses"].items()}, grads=grads,
                        grad_norm=float(norm), seconds=time.monotonic() - before)
            metrics.append(item)
            log.write(json.dumps(item) + "\n")
            log.flush()
            event("planner_step", **item)
    checkpoint = args.output / f"planner_step_{metrics[-1]['step']:06d}.pt"
    save_checkpoint(checkpoint, planner, optimizer, metrics[-1]["step"], run_contract)
    # Verify restore without destructive mutation: restore same state, compare RNG and parameters.
    expected_rng = torch.rand(8)
    expected_state = trainable_state(planner)
    restored = restore_checkpoint(checkpoint, planner, optimizer, run_contract)
    if not torch.equal(torch.rand(8), expected_rng):
        raise RuntimeError("Checkpoint RNG restoration differs")
    if any(not torch.equal(expected_state[k], v) for k, v in trainable_state(planner).items()):
        raise RuntimeError("Checkpoint parameter restoration differs")
    native_equal = None
    planner.eval()
    if native_before is not None:
        native_after, raw = generate_reply(planner, processor, prepared)
        native_equal = native_before == native_after
        write_json_new(args.output / "native_after.json", dict(raw=raw, plan=asdict(native_after)))
        if not native_equal:
            raise RuntimeError("Native reply changed after planning-only updates")
        # Autonomous inference layout: only observation, generated text and source image determine it.
        prefix = encode_prefix(planner, processor, prepared, native_after)
        grid = prepared["image_grid_thw"][0]
        merge = processor.image_processor.merge_size
        layout = make_layout(prefix, obs.output_duration_s,
                             (int(grid[1]) // merge, int(grid[2]) // merge), cfg)
    else:
        data, payload = read_batch(paths[0], cfg)
        layout = replace(data, targets=torch.zeros_like(data.targets))
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        predicted = planner.plan(layout, steps=3, denoising_steps=2)
    save_new(args.output / "predicted_plan.pt", dict(sample_id=obs.sample_id,
        hidden=predicted["hidden"].cpu(), mask=predicted["renderer_mask"].cpu(),
        features=predicted["features"].cpu(), contract=contract(cfg),
        text_plan=asdict(native_before) if native_before else payload["example"]["text_plan"],
        source="autonomous_text_and_plan" if native_before else "gt_text_predicted_plan"))
    return dict(engineering_pass=True, quality_pass=None, steps=args.steps, restored_step=restored,
                native_reply_unchanged=native_equal, frozen_audit_pass=True,
                inference_remaining=predicted["remaining"], checkpoint=str(checkpoint),
                max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, metrics=metrics)


def run_infer_plan(args):
    if args.observation is None or args.planner_checkpoint is None:
        raise ValueError("infer-plan requires --observation and --planner-checkpoint")
    obs = Observation(**json.loads(args.observation.read_text()))
    cfg = PlanConfig()
    planner, processor = load_planner(cfg)
    saved = torch.load(args.planner_checkpoint, map_location="cpu", weights_only=True)
    prefix = "planner." if any(k.startswith("planner.") for k in saved["trainable"]) else ""
    initialize_delta(args.planner_checkpoint, planner, contract(cfg), prefix)
    planner.eval()
    prepared = prepare_observation(processor, obs, cfg)
    plan, raw = generate_reply(planner, processor, prepared)
    prefix_data = encode_prefix(planner, processor, prepared, plan)
    grid = prepared["image_grid_thw"][0]
    merge = processor.image_processor.merge_size
    layout = make_layout(prefix_data, obs.output_duration_s,
                         (int(grid[1]) // merge, int(grid[2]) // merge), cfg)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        result = planner.plan(layout)
    save_new(args.output / "plan.pt", dict(contract=contract(cfg), sample_id=obs.sample_id,
        text_plan=asdict(plan), raw=raw, observation=asdict(obs), source="autonomous_text_and_plan",
        hidden=result["hidden"].cpu(), mask=result["renderer_mask"].cpu(),
        features=result["features"].cpu(), coordinates=layout.coordinates.cpu(),
        modalities=layout.modalities.cpu()))
    return dict(engineering_pass=True, quality_pass=None, gt_access=False,
                reply=plan.reply, remaining=result["remaining"])


def run(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(45)
    write_json_new(args.output / "run.json", dict(version=VERSION, command=args.command,
        physical_gpu=args.gpu, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    try:
        if args.command == "infer-plan":
            result = run_infer_plan(args)
        elif args.command in {"train-renderer", "train-joint"}:
            from .training import run_av_training
            result = run_av_training(args)
        elif args.command == "ltx-gate":
            from .ltx_gate import run_ltx_gate
            result = run_ltx_gate(args)
        else:
            result = run_planner(args)
        write_json_new(args.output / "completed.json", result)
        event("completed", **result)
    except Exception:
        write_json_new(args.output / "failed.json", dict(traceback=traceback.format_exc()))
        raise
