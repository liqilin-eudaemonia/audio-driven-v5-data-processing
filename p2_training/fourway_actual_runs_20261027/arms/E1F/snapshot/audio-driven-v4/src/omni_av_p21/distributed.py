"""P2.1 DDP: synchronize Extractor and native text-attention LoRA together."""
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
import random
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from omni_av.data import write_json_new
from omni_av.ltx_runtime import with_text
from omni_av_p2.checkpoint import capture_rng
from .checkpoint import export_deployment, initialize_from_p2, load, save
from .config import read_config
from omni_av_p2.data import contract
from omni_av_p2.ltx import load_renderer
from .model import TrainingModel
from omni_av_p2.model import Extractor
from omni_av_p2.launch import agree, establish_output, source_signature
from omni_av_p2.validation import check_indexes
from .validation import evaluate
from omni_av_p2.performance import TrainingSamples, configure_checkpointing, global_sample_index, sample_costs, balanced_sample_indices


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    writer = None
    output_ready = False
    try:
        establish_output(args.output, rank)
        output_ready = True
        if args.checkpoint_dir is not None and args.checkpoint_dir != args.output:
            establish_output(args.checkpoint_dir, rank)
        if rank == 0 and args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.output / "tensorboard"))
        result = train(args, rank, world, device, writer)
        if rank == 0:
            write_json_new(args.output / "completed.json", result)
            print(json.dumps(dict(event="completed", **result)), flush=True)
    except Exception:
        if output_ready:
            write_json_new(args.output / f"failed_rank{rank}.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


def save_distributed(path, raw, optimizer, step, identity, rank, world):
    states = [None] * world
    dist.all_gather_object(states, capture_rng())
    if rank == 0:
        save(path, raw, optimizer, step, identity, rank_rng=states)
        export_deployment(path, path.with_name(path.stem + "_deployment.pt"))
        write_json_new(path.with_suffix(".complete.json"), dict(step=step, world_size=world))
    dist.barrier()


def train(args, rank, world, device, writer=None):
    checkpoint_dir = args.checkpoint_dir or args.output
    cfg, lora = read_config(args.config)
    index = json.loads(args.cache.read_text())
    dev_index = json.loads(args.validation_cache.read_text()) if args.validation_cache else None
    check_indexes(index, dev_index, contract(cfg))
    identity = dict(contract=contract(cfg), recipe=args.recipe, entries=index["entries"],
                    tiny=args.tiny, seed=args.seed, world_size=world, accumulation=args.accumulation,
                    trainer="p21_ddp_v1", fused_adam=args.fused_adam, lora=lora.to_dict(),
                    initialization_policy="p2_extractor_before_renderer")
    if args.balance_workload:
        identity["balance_workload"] = True
    if dev_index is not None:
        identity.update(validation_entries=dev_index["entries"], validation_seed=args.validation_seed)
    agree(dict(identity=identity, source=source_signature(), torch_version=str(torch.__version__),
               local_world=int(os.environ["LOCAL_WORLD_SIZE"]), steps=args.steps, save_every=args.save_every,
               validate_every=args.validate_every, resume=str(args.resume), init_p2=str(args.init_p2),
               checkpoint_every=args.checkpoint_every, decoder_checkpointing=args.decoder_checkpointing))
    agree(dict(startup_save_step=args.startup_save_step))
    if rank == 0:
        write_json_new(args.output / "run.json", dict(config=cfg.to_dict(), lora=lora.to_dict(), identity=identity,
            args={k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
            global_batch=world * args.accumulation))
    # Match P2's seed and initialization order exactly. Loading LTX and adding
    # adapters must not consume RNG before the shared Extractor is initialized.
    extractor = Extractor(cfg, auxiliary=args.recipe == "joint").to(device)
    initial_extractor_digest = parameter_digest(extractor)
    renderer, strategy, sampler = load_renderer(cfg, device=device)
    raw = TrainingModel(cfg, lora, renderer, strategy, auxiliary=args.recipe == "joint", extractor=extractor).to(device)
    configure_checkpointing(renderer, raw.extractor, args.checkpoint_every, args.decoder_checkpointing)
    raw.train()
    if args.init_p2:
        source_step = initialize_from_p2(args.init_p2, raw, contract(cfg), args.recipe)
        if rank == 0:
            write_json_new(args.output / "p2_initialization.json", dict(path=str(args.init_p2),
                source_step=source_step, optimizer_resumed=False, new_step_count_starts_at=0))
    agree(dict(targets=raw.targets, config=cfg.to_dict(), lora=lora.to_dict()))
    if rank == 0:
        write_json_new(args.output / "trainable_parameters.json", dict(targets=raw.targets,
            parameters={n: p.numel() for n, p in raw.named_parameters()},
            renderer_base_frozen=True, independent_omni_attention=False,
            fresh_initial_extractor_sha256=initial_extractor_digest,
            initialization_policy="p2_extractor_before_renderer"))
    optimizer = torch.optim.AdamW(raw.optimizer_groups(), fused=args.fused_adam)
    start = load(args.resume, raw, identity, optimizer, rank=rank)["step"] if args.resume else 0
    model = DistributedDataParallel(raw, device_ids=[device.index], broadcast_buffers=False,
                                    gradient_as_bucket_view=True) if world > 1 else raw
    frozen = [(p, p._version) for p in renderer.parameters() if not p.requires_grad]
    samples = TrainingSamples(index["entries"], cfg, args.data_root, device, args.tiny,
                              args.allow_terminal_trim, args.cache_samples)
    dev_samples = (TrainingSamples(dev_index["entries"], cfg, args.data_root, device, args.tiny,
                    args.allow_terminal_trim, 0, split="dev") if dev_index else None)
    validations = []

    def validate(step):
        if dev_samples is None:
            return
        result = evaluate(raw, renderer, strategy, sampler, dev_samples, cfg, args.recipe,
                          rank, world, device, args.validation_seed)
        result["step"] = step
        validations.append(result)
        if rank == 0:
            write_json_new(args.output / f"validation_step_{step:06d}.json", result)
            if writer:
                for key in ("loss", "av", "video_feature", "audio_feature", "av_video", "av_audio"):
                    writer.add_scalar("dev/" + key, result[key], step)
                writer.flush()
            print(json.dumps(dict(event="p21_dev", **result)), flush=True)

    validate(start)
    costs = sample_costs(index["entries"], args.data_root) if args.balance_workload else None
    durations, checkpoints = [], []
    last, saved = start, start
    # Startup/loading and checkpoint writes are reported separately from measured updates.
    dist.barrier()
    started = time.perf_counter()
    with (args.output / f"train_rank{rank}.jsonl").open("x") as log:
        for step in range(start + 1, start + args.steps + 1):
            stop = torch.tensor(int((args.output / "STOP").exists()), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(5, device=device, dtype=torch.float64)
            sample_ids, noise_seeds, shapes, timing = [], [], [], []
            schedule = balanced_sample_indices(step, world, args.accumulation, costs)[rank] if costs else None
            for micro in range(args.accumulation):
                ordinal = schedule[micro] if schedule else global_sample_index(step, micro, rank, world, args.accumulation)
                condition, targets, text, payload, av = samples[ordinal]
                # Matched global sample numbers yield matched AV/aux noise in serial and DDP.
                seed = args.seed + ordinal + 1
                torch.manual_seed(seed)
                generator = torch.Generator(device=device).manual_seed(seed + 100000)
                sync = model.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
                        total, lav, lv, la, video_loss, audio_loss = model(
                            condition, targets if args.recipe == "joint" else None, prepared, generator)
                    if not torch.isfinite(total):
                        raise FloatingPointError("Nonfinite training loss")
                    events[1].record()
                    (total / args.accumulation).backward()
                events[2].record()
                timing.append(events)
                totals += torch.stack([total.detach(), lav.detach(), lv.detach(), la.detach(),
                                       video_loss.detach()]).double() / args.accumulation
                sample_ids.append(payload["sample_id"])
                shapes.append(list(av["latents"]["latents"].shape))
                noise_seeds.append(seed)
                # Release previous activation graph before the next forward.
                del prepared, total, lav, lv, la, video_loss, audio_loss
            layer_grads = {str(layer): float(norm.weight.grad.float().norm())
                           for layer, norm in zip(cfg.hidden_layers, raw.extractor.connector.memory.norms)}
            raw.assert_parameter_ownership()
            lora_grad = float(torch.stack([p.grad.float().square().sum() for p in raw.adapters.parameters()
                                          if p.grad is not None]).sum().sqrt())
            connector_grad = float(torch.stack([p.grad.float().square().sum() for p in raw.extractor.connector.parameters()
                                               if p.grad is not None]).sum().sqrt())
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if any(p.requires_grad or p.grad is not None or p._version != v for p, v in frozen):
                raise RuntimeError("Frozen renderer base was updated")
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - tick
            duration = torch.tensor(seconds, device=device, dtype=torch.float64)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            dist.all_reduce(totals)
            totals /= world
            durations.append(float(duration))
            row = dict(step=step, rank=rank, seconds=float(duration), loss=float(totals[0]),
                       av=float(totals[1]), video_feature=float(totals[2]), audio_feature=float(totals[3]),
                       av_video=float(totals[4]), av_audio=float(totals[1] - totals[4]),
                       grad_norm=float(norm), layer_gradient_norms=layer_grads,
                       lora_grad_norm=lora_grad, connector_grad_norm=connector_grad,
                       forward_seconds=sum(e[0].elapsed_time(e[1]) for e in timing) / 1000,
                       backward_including_sync_seconds=sum(e[1].elapsed_time(e[2]) for e in timing) / 1000,
                       sample_ids=sample_ids, noise_seeds=noise_seeds, video_latent_shapes=shapes,
                       peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if rank == 0:
                print(json.dumps(dict(event="p21_ddp_step", **row)), flush=True)
                if writer:
                    for key in ("loss", "av", "video_feature", "audio_feature", "av_video", "av_audio", "grad_norm", "lora_grad_norm", "connector_grad_norm"):
                        writer.add_scalar("train/" + key, row[key], step)
                    writer.add_scalar("train/samples_seen", step * world * args.accumulation, step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                    writer.add_scalar("train/lora_lr", optimizer.param_groups[1]["lr"], step)
                    writer.add_scalar("time/step_seconds_max_rank", float(duration), step)
                    writer.add_scalar("time/samples_per_second", world * args.accumulation / float(duration), step)
                    writer.add_scalar("memory/rank0_peak_allocated_gb", row["peak_allocated_gb"], step)
                    writer.flush()
            last = step
            if step % args.save_every == 0 or step == start + args.steps or step == args.startup_save_step:
                path = checkpoint_dir / f"extractor_step_{step:06d}.pt"
                save_distributed(path, raw, optimizer, step, identity, rank, world)
                checkpoints.append(str(path))
                saved = step
            if step % args.validate_every == 0 or step == start + args.steps:
                validate(step)
        if last != saved:
            path = checkpoint_dir / f"extractor_step_{last:06d}.pt"
            save_distributed(path, raw, optimizer, last, identity, rank, world)
            checkpoints.append(str(path))
    if dev_samples and validations[-1]["step"] != last:
        validate(last)
    hashes = [None] * world
    dist.all_gather_object(hashes, parameter_digest(raw))
    if len(set(hashes)) != 1:
        raise RuntimeError("Trainable parameters differ across ranks")
    peaks = [None] * world
    dist.all_gather_object(peaks, torch.cuda.max_memory_allocated(device) / 1e9)
    measured = durations[1:] if len(durations) > 1 else durations
    mean = sum(measured) / len(measured) if measured else None
    return dict(training_completed_steps=last - start, world_size=world,
                global_batch=world * args.accumulation, checkpoints=checkpoints, validations=validations,
                frozen_renderer_base=True, text_attention_lora=True, ranks_identical=True, parameter_sha256=hashes[0],
                step_seconds=durations, steady_step_seconds=mean,
                steady_samples_per_second=world * args.accumulation / mean if mean else None,
                elapsed_including_save_s=time.perf_counter() - started, peak_allocated_gb=peaks,
                stopped=(args.output / "STOP").exists(), quality_pass=None)
