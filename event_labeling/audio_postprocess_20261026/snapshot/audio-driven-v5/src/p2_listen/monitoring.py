"""Detached global training metrics and one TensorBoard writer per run."""
import json
import os
from pathlib import Path
import tempfile
import time
import torch
import torch.distributed as dist
from .config import tensorboard_config


def require_tensorboard(c):
    # Check on every rank before allocating the renderer or starting collectives.
    if tensorboard_config(c)["enabled"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("TensorBoard is enabled; install requirements-p2-monitoring.txt in the training environment") from error


def truncate_jsonl(path, step):
    """Drop uncommitted rows after a checkpoint, including a torn final line."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    kept = []
    previous = -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"Corrupt log before final line: {path}")
        if row["step"] <= step:
            if row["step"] <= previous:
                raise ValueError(f"Committed log steps are not increasing: {path}")
            previous = row["step"]
            kept.append(line + "\n")
    fd, temp = tempfile.mkstemp(prefix=path.name + ".resume.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.writelines(kept)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def rank_zero_action(action, *, rank, world):
    """A writer/open/flush failure must reach peers before their next update."""
    error = None
    result = None
    if rank == 0:
        try:
            result = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if world > 1:
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error:
        raise RuntimeError("Training monitor failed: " + error)
    return result


def global_scalars(record, *, local_samples, media_seconds, timings, optimizer,
                   event_weight, arm, step, dataset_size, device, world=1):
    """AV means are sample-weighted; Event already uses global counts in training."""
    if local_samples < 1 or dataset_size < 1:
        raise ValueError("Positive sample counts required")
    if world < 1 or (world > 1 and (not dist.is_initialized() or dist.get_world_size() != world)):
        raise ValueError("Metric world differs from initialized process group")
    values = {"base": record["base"], "av": record["av"]}
    for modality in ("video", "audio"):
        values["av_" + modality] = record.get("av_loss_by_modality", {}).get(modality)
        values["feature_" + modality] = (record.get("feature_losses") or {}).get(modality)
    sums = torch.tensor([local_samples, media_seconds] +
        [v for value in values.values() for v in (0. if value is None else value * local_samples,
                                                   0. if value is None else local_samples)],
        dtype=torch.float64, device=device)
    timing_keys = ("step_seconds", "data_seconds", "compute_seconds", "checkpoint_seconds",
                   "allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib")
    maxima = torch.tensor([timings[k] for k in timing_keys], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    numbers = sums.tolist(); maximum = dict(zip(timing_keys, maxima.tolist()))
    count, seconds = numbers[:2]
    result = {}
    for index, key in enumerate(values):
        total, valid = numbers[2 + index * 2:4 + index * 2]
        if valid == count:
            result["loss/" + key] = total / count
    result["loss/event"] = record["event"]
    result["loss/event_weighted"] = event_weight * record["event"]
    result["loss/optimization_objective"] = result["loss/base"] + result["loss/event_weighted"]
    # In E0 the event objective only updates the detached probe, not the generator.
    result["loss/generator_objective"] = result["loss/base"] + (result["loss/event_weighted"] if arm == "E1" else 0.)
    for key, value in record["event_loss_per_class"].items():
        if value is not None:
            result["event_loss/" + key] = value
    for key, value in record["event_loss_by_modality"].items():
        result["event_contribution/" + key] = value
    for key, value in record["event_known_bins"].items():
        result["supervision/known_bins/" + key] = value
    result["supervision/active_classes"] = sum(value > 0 for value in record["event_known_bins"].values())
    result.update({"grad_norm/" + k + "_preclip": v for k, v in record["grad_norm"].items()})
    result.update({"learning_rate/" + group["name"]: group["lr"] for group in optimizer.param_groups})
    result.update({"time/" + k + "_max_rank": maximum[k] for k in timing_keys[:4]})
    result.update({"memory/" + k + "_max_rank": maximum[k] for k in timing_keys[4:]})
    elapsed = maximum["step_seconds"]
    if elapsed > 0:
        result["throughput/samples_per_second"] = count / elapsed
        result["throughput/media_seconds_per_second"] = seconds / elapsed
    result.update({"progress/global_batch": count, "progress/samples_seen": step * count,
                   "progress/epochs_seen": step * count / dataset_size})
    return result


class TrainingMonitor:
    def __init__(self, output, c, start=0, resume=False):
        from torch.utils.tensorboard import SummaryWriter
        self.output = Path(output)
        settings = tensorboard_config(c)
        self.writer = SummaryWriter(str(self.output / "tensorboard"),
                                    purge_step=start + 1 if resume else None,
                                    flush_secs=settings["flush_secs"], max_queue=100)
        self.writer.add_text("run/config", "```json\n" + json.dumps(c, indent=2, ensure_ascii=False) + "\n```", start)

    def write(self, step, scalars, flush=False):
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, time=time.time(), scalars=scalars)) + "\n")
        if flush:
            self.writer.flush()

    def close(self):
        self.writer.close()


def open_monitor(output, c, *, start, resume, rank, world):
    def initialize():
        if resume:
            for file in [*Path(output).glob("train_rank*.jsonl"), Path(output) / "metrics.jsonl"]:
                truncate_jsonl(file, start)
        return TrainingMonitor(output, c, start, resume) if tensorboard_config(c)["enabled"] else None
    return rank_zero_action(initialize, rank=rank, world=world)
