"""P3-only caches: frozen input embeddings and native, unpooled AV teachers."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from omni_av.data import Observation, TextPlan, file_hash, read_audio, save_new, write_json_new
from omni_av.lora import native_reply_mode
from omni_av.omni import encode_prefix, move_inputs, prepare_observation, sampled_frames, observation_prompt_contract
from . import VERSION
from .model import Layout


def contract(cfg):
    return dict(version=VERSION, config=asdict(cfg), prompt=observation_prompt_contract(),
                model_config=file_hash(Path(cfg.omni_path) / "config.json"),
                model_index=file_hash(Path(cfg.omni_path) / "model.safetensors.index.json"))


def load_native(cfg, device):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.omni_path, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(cfg.omni_path,
        dtype=torch.bfloat16, device_map={"": device}, local_files_only=True, attn_implementation="sdpa")
    model.requires_grad_(False).eval()
    if model.config.text_config.hidden_size != cfg.hidden_size:
        raise ValueError("Native target/hidden width mismatch")
    return SimpleNamespace(thinker=model, config=cfg), processor


def select_examples(path, split, count, cfg):
    rows = json.loads(Path(path).read_text())
    if not rows or any(x["split"] != split for x in rows):
        raise ValueError("Unexpected manifest split")
    ids = [x["observation"]["sample_id"] for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate manifest IDs")
    candidates = [x for x in rows if 0 < x["duration_s"] <= cfg.max_duration
                  and x["text_plan"]["reply"].strip()]
    candidates.sort(key=lambda x: hashlib.sha256(
        f'{cfg.seed}:{x["observation"]["sample_id"]}'.encode()).hexdigest())
    if len(candidates) < count:
        raise ValueError("Insufficient reply-bearing samples in duration stratum")
    return candidates[:count]


def fixed_plan(example):
    """Existing labels, explicitly teacher-forced; no new manual annotation."""
    gt = json.loads(Path(example["gt_path"]).read_text())["avatar_plan"]
    if gt["sample_id"] != example["observation"]["sample_id"]:
        raise ValueError("GT identity mismatch")
    plan = TextPlan(**example["text_plan"])
    if plan.reply != gt["response"].get("text", ""):
        raise ValueError("GT reply mismatch")
    timeline = dict(duration_seconds=example["duration_s"], time_unit="ms",
                    vocal_track=gt.get("vocal_track", []), floor_track=gt.get("floor_track", []))
    serialized = plan.serialize() + "\nFixed target timeline: " + json.dumps(timeline, ensure_ascii=False)
    return SimpleNamespace(serialize=lambda: serialized), serialized


@torch.no_grad()
def extract(runtime, processor, example):
    cfg, model = runtime.config, runtime.thinker
    device = next(model.parameters()).device
    duration = float(example["duration_s"])
    obs = Observation(**example["observation"])
    if obs.speaker_end_s > duration:
        raise ValueError("Observation extends beyond target support")
    plan, serialized = fixed_plan(example)
    prefix = encode_prefix(runtime, processor, prepare_observation(processor, obs, cfg), plan)
    prefix.pop("text_mask")
    frames = sampled_frames(example["target_video"], duration, cfg.vision_fps, cfg.vision_max_pixels)
    vi = processor.video_processor(videos=[frames], return_tensors="pt", do_sample_frames=False,
        max_pixels=cfg.vision_max_pixels, min_pixels=28 * 28 * 4)
    vi = move_inputs(dict(vi), device)
    wave = read_audio(example["target_audio"], end_s=duration)
    ai = processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt",
                                    return_attention_mask=True, padding="max_length")
    with native_reply_mode(model):
        video = model.get_video_features(vi["pixel_values_videos"], vi["video_grid_thw"])
        audio = model.get_audio_features(ai["input_features"].to(device, torch.bfloat16),
                                        ai["attention_mask"].to(device))
    # Native audio: mel hop 10 ms, convolution stride 2, pooling stride 2 (~25 Hz).
    _, expected_audio = model.audio_tower._get_feat_extract_output_lengths(ai["attention_mask"].sum(-1))
    if len(audio) != int(expected_audio.item()):
        raise ValueError("Native audio output length mismatch")
    hop = processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate
    if not math.isclose(hop, .01):
        raise ValueError("Unexpected native mel hop")
    tg, gh, gw = map(int, vi["video_grid_thw"][0].tolist())
    merge = processor.image_processor.merge_size
    patch_t = processor.video_processor.temporal_patch_size
    if patch_t != 2 or gh % merge or gw % merge:
        raise ValueError("Unexpected native video grid")
    coords = [[(t * patch_t + .5) / cfg.vision_fps, y, x]
              for t in range(tg) for y in range(gh // merge) for x in range(gw // merge)]
    if len(coords) != len(video):
        raise ValueError("Native video output length mismatch")
    coords += [[(4 * t + 2) * hop, 0, 0] for t in range(len(audio))]
    targets = torch.cat([video, audio])[None]
    layout = Layout(**prefix, coordinates=torch.tensor([coords], device=device, dtype=torch.float32),
        modalities=torch.tensor([[0] * len(video) + [1] * len(audio)], device=device),
        valid=torch.ones(targets.shape[:2], dtype=torch.bool, device=device))
    layout.validate()
    if not torch.isfinite(targets).all() or targets.shape[-1] != cfg.hidden_size:
        raise ValueError("Invalid teachers")
    return dict(layout={k: v.cpu() for k, v in vars(layout).items()}, targets=targets.cpu(),
        sample_id=obs.sample_id, split=example["split"], fixed_plan=serialized,
        duration_s=duration, video_tokens=len(video), audio_tokens=len(audio),
        gt_sha256=file_hash(example["gt_path"]), source=example,
        note="Final visual temporal patch may repeat the last frame, as native preprocessing does.")


def prepare(cfg, train_manifest, dev_manifest, output, device):
    train = select_examples(train_manifest, "train", cfg.train_samples, cfg)
    dev = select_examples(dev_manifest, "dev", cfg.dev_samples, cfg)
    if {x["observation"]["sample_id"] for x in train} & {x["observation"]["sample_id"] for x in dev}:
        raise ValueError("Train/Dev identity overlap")
    output.mkdir(parents=True, exist_ok=False)
    metadata = contract(cfg)
    write_json_new(output / "contract.json", metadata)
    write_json_new(output / "selection.json", dict(train=train, dev=dev))
    runtime, processor = load_native(cfg, device)
    index = dict(contract=metadata, train=[], dev=[])
    for row in train + dev:
        sample = extract(runtime, processor, row)
        path = output / f'{row["split"]}_{sample["sample_id"]}.pt'
        save_new(path, sample)
        index[row["split"]].append(dict(path=str(path.resolve()), sha256=file_hash(path),
                                      sample_id=sample["sample_id"]))
        print(json.dumps(dict(prepared=sample["sample_id"], video=sample["video_tokens"],
                              audio=sample["audio_tokens"])), flush=True)
    write_json_new(output / "index.json", index)


def load_cache(index_path, cfg):
    index = json.loads(Path(index_path).read_text())
    # JSON round-trip normalizes tuple/list config values.
    if index["contract"] != json.loads(json.dumps(contract(cfg))):
        raise ValueError("Cache contract mismatch; prepare a new cache")
    result = {}
    for split in ("train", "dev"):
        rows = []
        for record in index[split]:
            if file_hash(record["path"]) != record["sha256"]:
                raise ValueError("Cache content changed")
            row = torch.load(record["path"], map_location="cpu", weights_only=True)
            if row["split"] != split or row["sample_id"] != record["sample_id"]:
                raise ValueError("Cache identity mismatch")
            rows.append(row)
        result[split] = rows
    train_ids = {x["sample_id"] for x in result["train"]}
    dev_ids = {x["sample_id"] for x in result["dev"]}
    if not train_ids or not dev_ids or train_ids & dev_ids:
        raise ValueError("Invalid Train/Dev separation")
    return result
